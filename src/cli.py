from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .config import load_config
from .drivers import build_drivers
from .llm import ChatMessage, HybridLLMDriver, TieredLLMDriver
import httpx
from tools.info_search import get_info as info_get_info
from difflib import SequenceMatcher
import time as _time
from tools.weather_tool import get_weather as weather_get, speak_weather as weather_say
from .text_normalizer import normalize_for_speech
from .conversation import get_memory, reset_memory
from .context_awareness import (
    TimeContext, get_context, get_urgency_detector, 
    build_context_aware_prompt, get_contextual_acknowledgment, reset_context
)
from .user_identity import (
    get_registry, get_session, get_current_user, set_current_user,
    identify_from_audio, format_user_greeting, get_user_context_for_prompt,
    reset_session
)
from .memory import (
    get_user_memory, get_current_user_memory, maybe_learn_from_statement,
    extract_fact_from_statement
)
from .intent import classify as classify_intent, Intent
from .reminders import (
    add_reminder, list_reminders, parse_reminder,
    register_speak, start_background_thread as start_reminder_thread,
)
from .tools_registry import (
    TOOLS_SCHEMA, ToolContext, dispatch_tool_call,
)
try:
    # optional local fallback if API returns no data
    from assistant.finance.yahoo import get_quote as _local_get_quote  # type: ignore
except Exception:  # pragma: no cover
    _local_get_quote = None  # type: ignore

from .commands.finance import money_to_words, resolve_symbol_from_text

from .commands.info_search import (
    is_lore_query, shorten_summary, extract_info_query,
    normalize_mishearings, clean_topic, _fuzzy_has,
)

def answer_from_lore(user_input: str, hit: dict) -> str | None:
    meta = hit.get("meta", {}) or {}
    q = (user_input or "").lower()
    species = meta.get("species", {}) or {}
    corp = meta.get("corporation", {}) or {}
    # specific: Xenomorph weaknesses
    if ("weak" in q or "vulnerab" in q or "kill" in q or "hurt" in q or "damage" in q) and ("xenomorph" in q or "alien" in q):
        x = species.get("xenomorph") or {}
        if isinstance(x, dict):
            b = x.get("biology") or {}
            if isinstance(b, dict) and b.get("weakness"):
                return f"Documented weaknesses: {b.get('weakness')}."
    # divisions of Weyland-Yutani
    if "division" in q or "divisions" in q:
        divs = corp.get("divisions") if isinstance(corp, dict) else None
        if isinstance(divs, dict) and divs:
            names = [k.replace("_", " ") for k in divs.keys()]
            return "Divisions include: " + ", ".join(names) + "."
    # what are aliens also known as / xenoporph → xenomorph designation
    if any(k in q for k in ["alien", "xenomorph"]) or _fuzzy_has("xenomorph", q):
        x = species.get("xenomorph") or {}
        desig = (x.get("designation") or "Xenomorph XX121") if isinstance(x, dict) else "Xenomorph XX121"
        bio = ""
        if isinstance(x, dict):
            b = x.get("biology") or {}
            if isinstance(b, dict) and b.get("blood"):
                bio = f" Their blood is {b.get('blood')}."
        return f"Also referred to as {desig}.{bio}".strip()
    # types of life found → list species keys
    if any(k in q for k in ["types of", "what are the types", "life found", "species"]):
        keys = []
        if isinstance(species, dict):
            keys = [k.replace("_", " ") for k in species.keys()]
        return ", ".join(keys) if keys else None
    # fallback: first concise line from note body
    txt = (hit.get("text") or "").strip()
    for line in txt.splitlines():
        s = line.strip().lstrip("- ")
        if s and len(s.split()) >= 4:
            return s
    return None


def _select_best_hit_for_query(hits: list[dict], user_input: str) -> dict:
    if not hits:
        return {}
    q = (user_input or "").lower()
    # Extract simple name tokens from query (first/last)
    import re as _re3
    words = [w for w in _re3.findall(r"[a-zA-Z]+", q) if len(w) > 1]
    best = hits[0]
    best_score = -1.0
    for h in hits:
        meta = h.get("meta", {}) or {}
        name = (meta.get("name") or "").lower()
        # also consider filename alias
        try:
            p = (h.get("path") or "").split("/")[-1].split("\\")[-1]
            stem = p.rsplit(".", 1)[0].replace("_", " ")
            name_alt = stem.lower()
        except Exception:
            name_alt = ""
        # token overlap
        overlap = 0
        for w in words:
            if w and (w in name or (name_alt and w in name_alt)):
                overlap += 1
        # fuzzy similarity against name and aliases
        aliases = meta.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        cand_names = [name, name_alt] + [str(a).lower() for a in aliases if a]
        ratio = max((SequenceMatcher(None, " ".join(words), cn).ratio() for cn in cand_names if cn), default=0.0)
        score = overlap + ratio  # simple combined score
        if score > best_score:
            best_score = score
            best = h
    # If no overlap with any name tokens, return empty to force safer fallback
    if best_score <= 0.4:  # slightly lower threshold; still require some signal
        # Secondary heuristic: if any candidate's name contains at least two query tokens (e.g., first + last), pick it
        import re as _re_alt
        tokens = [w for w in _re_alt.findall(r"[a-zA-Z]+", q) if len(w) > 1]
        def token_overlap(name_text: str) -> int:
            nt = (name_text or "").lower()
            return sum(1 for w in tokens if w in nt)
        best2 = None
        best2_ov = 0
        for h in hits:
            meta = h.get("meta", {}) or {}
            nm = (meta.get("name") or "").lower()
            if not nm:
                try:
                    p = (h.get("path") or "").split("/")[-1].split("\\")[-1]
                    nm = p.rsplit(".", 1)[0].replace("_", " ").lower()
                except Exception:
                    nm = ""
            ov = token_overlap(nm)
            if ov > best2_ov:
                best2_ov = ov
                best2 = h
        # Require reasonably confident two-token overlap like first+last name
        if best2 and best2_ov >= 2:
            return best2
        return {}
    return best


def answer_from_rag(user_input: str, hits: list[dict]) -> str | None:
    if not hits:
        return None
    q = (user_input or "").lower()
    h0 = _select_best_hit_for_query(hits, user_input)
    if not h0:
        return None
    meta = h0.get("meta", {}) or {}
    name = meta.get("name")
    if not name:
        try:
            p = (h0.get("path") or "").split("/")[-1].split("\\")[-1]
            stem = p.rsplit(".", 1)[0].replace("_", " ")
            if stem:
                name = stem.title()
        except Exception:
            name = None
    relations = meta.get("relations", {}) or {}
    education = meta.get("education", {}) or {}
    location = meta.get("location")
    ethnicity = meta.get("ethnicity")

    if any(k in q for k in ["dating", "girlfriend", "boyfriend", "partner"]):
        d = relations.get("dating")
        if d:
            subj = name or "They"
            val = str(d).strip().lower()
            if val in {"none", "n/a", "no", "null", "not applicable"}:
                return f"{subj} is not currently dating anyone."
            return f"{subj} is dating {d}."
        txt = (h0.get("text") or "")
        for line in txt.splitlines():
            ll = line.strip().lower()
            if ll.startswith("- ") and "dating" in ll:
                try:
                    after = line.split("dating", 1)[1].strip(" :.-")
                    subj = name or "They"
                    if after:
                        return f"{subj} is dating {after}."
                except Exception:
                    pass
    if any(k in q for k in ["study", "studying", "major", "program", "degree"]):
        prog = education.get("program")
        uni = education.get("university")
        subj = name or "They"
        if prog and uni:
            return f"{subj} is studying {prog} at {uni}."
        if prog:
            return f"{subj} is studying {prog}."
        if uni:
            return f"{subj} attends {uni}."
        txt = (h0.get("text") or "")
        for line in txt.splitlines():
            ll = line.strip().lower()
            if ll.startswith("- ") and ("student" in ll or "studying" in ll or "major" in ll):
                return line.strip("- ").strip()
    if any(k in q for k in ["where", "based", "live", "location"]):
        if location:
            subj = name or "They"
            return f"{subj} is based in {location}."
        txt = (h0.get("text") or "")
        for line in txt.splitlines():
            if line.lower().startswith("- based "):
                return line.lstrip("- ").strip()
    if any(k in q for k in ["ethnicity", "race", "heritage", "background"]):
        if ethnicity:
            subj = name or "They"
            return f"{subj}'s ethnicity is {ethnicity}."
        # heuristic from text lines
        txt = (h0.get("text") or "")
        for line in txt.splitlines():
            ll = line.strip().lower()
            if ll.startswith("- ") and ("ethnicity" in ll or "heritage" in ll):
                return line.strip("- ").strip()
    return None

def is_person_qa_query(user_input: str) -> bool:
    q = (user_input or "").lower()
    keywords = [
        "ethnicity", "race", "heritage", "background",
        "dating", "girlfriend", "boyfriend", "partner",
        "study", "studying", "major", "program", "degree",
        "where", "based", "location", "live",
    ]
    return any(k in q for k in keywords)

def fallback_line_from_text(q: str, text: str) -> str | None:
    ql = (q or "").lower()
    # Determine which keywords are actually asked
    asked = []
    key_groups = [
        ("ethnicity", ["ethnicity", "heritage", "background", "race"]),
        ("dating", ["dating", "girlfriend", "boyfriend", "partner"]),
        ("study", ["study", "studying", "major", "program", "student", "degree"]),
        ("location", ["where", "based", "location", "live"]),
    ]
    active = set()
    for tag, keys in key_groups:
        if any(k in ql for k in keys):
            active.add(tag)
    def score_line(line: str) -> int:
        ll = line.strip().lower()
        s = 0
        if "ethnicity" in active and any(k in ll for k in ["ethnicity", "heritage", "background", "race"]):
            s += 2
        if "dating" in active and any(k in ll for k in ["dating", "girlfriend", "boyfriend", "partner"]):
            s += 2
        if "study" in active and any(k in ll for k in ["student", "studying", "major", "program", "degree"]):
            s += 2
        if "location" in active and any(k in ll for k in ["based", "location", "live"]):
            s += 2
        return s
    best_line = None
    best_score = 0
    for line in (text or "").splitlines():
        sc = score_line(line)
        if sc > best_score:
            best_score = sc
            best_line = line.strip("- ").strip()
    return best_line if best_score > 0 else None

def extract_name_from_hit(h: dict) -> str | None:
    meta = h.get("meta", {}) or {}
    name = meta.get("name")
    if name:
        return str(name)
    try:
        p = (h.get("path") or "").split("/")[-1].split("\\")[-1]
        stem = p.rsplit(".", 1)[0].replace("_", " ")
        return stem.title() if stem else None
    except Exception:
        return None

def summarize_person_hit(h: dict) -> str:
    meta = h.get("meta", {}) or {}
    name = meta.get("name") or extract_name_from_hit(h) or "This person"
    relations = meta.get("relations", {}) or {}
    education = meta.get("education", {}) or {}
    location = meta.get("location")
    parts: list[str] = []
    if education.get("program") and education.get("university"):
        parts.append(f"studies {education.get('program')} at {education.get('university')}")
    elif education.get("program"):
        parts.append(f"studies {education.get('program')}")
    elif education.get("university"):
        parts.append(f"attends {education.get('university')}")
    if relations.get("dating") and str(relations.get("dating")).strip().lower() not in {"none","n/a","no","null","not applicable"}:
        parts.append(f"is dating {relations.get('dating')}")
    if location:
        parts.append(f"is based in {location}")
    if not parts:
        # fallback to first meaningful line of text
        txt = (h.get("text") or "").strip()
        for line in txt.splitlines():
            s = line.strip().lstrip("- ")
            if s and len(s.split()) >= 4:
                parts.append(s)
                break
    desc = ", ".join(parts)
    if not desc:
        return str(name)
    if not str(name).lower().startswith(("they ", "he ", "she ")):
        return f"{name} {desc}."
    return f"{name} {desc}."

def has_pronoun_reference(text: str) -> bool:
    t = (text or "").lower()
    for w in ["his", "her", "their", "him", "her", "them"]:
        if f" {w} " in f" {t} ":
            return True
    return False


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Modular AI assistant CLI")
    parser.add_argument("--config", default="configs/app.yaml", help="Path to YAML config")
    parser.add_argument("--prompt", default="Say hello!", help="User prompt to send to LLM")
    parser.add_argument("--text", default=None, help="Direct text to TTS (overrides LLM output if provided)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM and TTS the provided --text or --prompt")
    parser.add_argument(
        "--voice-out",
        default=None,
        help="Optional output WAV path; if not set, audio is returned to stdout",
    )
    parser.add_argument("--mic", action="store_true", help="Record from microphone and transcribe before sending to LLM")
    parser.add_argument("--auto", action="store_true", help="Handsfree: VAD-based capture and streaming STT")
    parser.add_argument("--mic-seconds", type=int, default=5, help="Duration to record from mic if --mic is set")
    parser.add_argument("--ptt", action="store_true", help="Temporary push-to-talk: press Enter to start/stop recording")
    parser.add_argument("--ptt-seconds", type=int, default=5, help="Fallback duration if stdin is not interactive")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    llm, tts, stt = build_drivers(cfg)
    # Persistent HTTP client reused across all RAG/finance/weather calls
    _http = httpx.Client(timeout=5.0)

    if args.ptt:
        try:
            import sounddevice as sd
            import numpy as np
            import sys
            import threading
            import re
            import time
        except ImportError:
            print("sounddevice is required for --ptt. Install dependencies.")
            return 1
        import os as _os_ptt
        if _os_ptt.name == 'nt':
            try:
                import msvcrt
            except ImportError:
                msvcrt = None  # type: ignore[assignment]
        else:
            msvcrt = None  # type: ignore[assignment]
        if stt is None:
            print("No STT configured in configs/app.yaml under 'stt'.")
            return 1

        # --- Initialize StreamingSTT (Deepgram primary, Whisper fallback) ---
        import asyncio as _asyncio_stt
        from mother.audio.stt import StreamingSTT as _StreamingSTT
        _streaming_stt = None
        try:
            _streaming_stt = _StreamingSTT()
            _asyncio_stt.run(_streaming_stt.init())
            _stt_engine = "Deepgram" if _streaming_stt._deepgram_available else "Faster-Whisper"
            print(f"[STT] StreamingSTT ready (engine: {_stt_engine})")
        except Exception as _stt_err:
            print(f"[STT] StreamingSTT init failed ({_stt_err}) — using legacy STT")
            _streaming_stt = None

        samplerate = 16000
        recording = []  # type: ignore[var-annotated]
        is_recording = False
        stream = None
        armed_recording = False  # when True, immediately start recording on next loop

        def on_enter():
            nonlocal is_recording, stream
            if not is_recording:
                print("Listening... (press Enter to stop)")
                is_recording = True
                recording.clear()
                stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                                        callback=lambda indata, frames, time, status: recording.append(indata.copy()))
                stream.start()
            else:
                is_recording = False
                if stream:
                    stream.stop(); stream.close(); stream = None

        # ---- TTS helpers -------------------------------------------------------
        # _synthesize: text → (data, sr) via persistent Piper subprocess.
        #   No temp file, no new process spawn per sentence.
        # _play_audio:  plays (data, sr) with 5ms fade and Enter-interrupt support.
        # speak_now:    convenience wrapper used by all non-LLM code paths.
        # -----------------------------------------------------------------------
        import io as _io_tts
        import soundfile as _sf_tts
        import sounddevice as _sd_tts
        import numpy as _np_tts
        import time as _time_tts

        def _synthesize(text: str):
            """Synthesize text using the persistent Piper backend.
            Returns (data_array, sample_rate) or (None, None) on error.
            Falls back to file-based synthesis if in-memory path fails.
            """
            try:
                wav = tts.synthesize_to_bytes(text)
                if not wav or len(wav) < 100:
                    raise ValueError("Empty or too-short WAV data")
                return _sf_tts.read(_io_tts.BytesIO(wav), dtype="float32")
            except Exception as _e:
                # Fallback: synthesize via temp file (always produces valid WAV)
                try:
                    import tempfile as _tf
                    with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp:
                        _tmp_path = _tmp.name
                    tts.synthesize_to_file(text, _tmp_path)
                    result = _sf_tts.read(_tmp_path, dtype="float32")
                    import os as _os_tts
                    _os_tts.unlink(_tmp_path)
                    return result
                except Exception as _e2:
                    print(f"[TTS] Synthesis error: {_e2}")
                    return None, None

        def _play_audio(data, sr) -> bool:
            """Play audio array with fade-in/out and Enter-interrupt support.
            Returns True if the user interrupted playback.
            """
            if data is None or sr is None:
                return False
            try:
                n = max(1, int(sr * 0.005))  # 5 ms fade
                if data.ndim == 1:
                    n = min(n, max(1, data.shape[0] // 4))
                    ramp = _np_tts.linspace(0.0, 1.0, n, dtype=data.dtype)
                    data[:n] *= ramp
                    data[-n:] *= ramp[::-1]
                else:
                    n = min(n, max(1, data.shape[0] // 4))
                    ramp = _np_tts.linspace(0.0, 1.0, n, dtype=data.dtype)[:, None]
                    data[:n, :] *= ramp
                    data[-n:, :] *= ramp[::-1]
                _np_tts.clip(data, -1.0, 1.0, out=data)
            except Exception:
                pass
            _sd_tts.play(data, sr)
            interrupted = False
            dur_s = max(0.0, float(len(data)) / float(sr) if sr else 0.0)
            t_start = _time_tts.monotonic()
            while (_time_tts.monotonic() - t_start) < dur_s:
                if msvcrt and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == "\r":
                        _sd_tts.stop()
                        interrupted = True
                        break
                _time_tts.sleep(0.01)
            if not interrupted:
                try:
                    _sd_tts.wait()
                except Exception:
                    pass
            return interrupted

        def speak_now(text_to_say: str) -> bool:
            """Synthesize and play text. Returns True if user interrupted."""
            if not text_to_say or not text_to_say.strip():
                return False
            text_to_say = normalize_for_speech(text_to_say)
            text_to_say = text_to_say.replace('\x00', '').replace('\ufffd', '')
            if not text_to_say.strip():
                return False
            # Serve from pre-synthesized cache when available
            cached = _phrase_cache.get(text_to_say.strip())
            if cached is not None:
                c_data, c_sr = cached
                return _play_audio(c_data, c_sr)
            data, sr = _synthesize(text_to_say)
            if data is None:
                return False
            return _play_audio(data, sr)

        # Pre-synthesize frequently-used short phrases so zero TTS latency for acks
        _CANNED_PHRASES = [
            "One moment.",
            "Working on it.",
            "Understood.",
            "Acknowledged.",
            "Let me check.",
            "Stand by.",
            "On it.",
            "I'm on it.",
            "Right away.",
            "Of course.",
            "I don't have that information.",
            "Sorry, I couldn't get that.",
        ]
        _phrase_cache: dict = {}
        for _phrase in _CANNED_PHRASES:
            _d, _s = _synthesize(normalize_for_speech(_phrase))
            if _d is not None:
                _phrase_cache[_phrase] = (_d, _s)

        # Fresh session
        reset_context()
        reset_session()
        
        # Check if any users are enrolled
        registry = get_registry()
        enrolled_users = registry.list_users()
        
        if enrolled_users:
            # Time-aware greeting with user check
            greeting = TimeContext.get_greeting() + " Speak so I can identify you."
            print(f"\n[MOTHER] {greeting}")
            speak_now(greeting)
        else:
            # No users enrolled - suggest enrollment
            greeting = TimeContext.get_greeting()
            print(f"\n[MOTHER] {greeting}")
            print("[Note: No users enrolled. Run 'python -m src.enroll_user' to enroll.]")
            speak_now(greeting)
        
        # --- Wake word detector (background, concurrent with PTT) ---
        _wake_detector = None
        try:
            from mother.audio.wake_word import WakeWordDetector as _WakeWordDetector

            def _on_wake_word(keyword: str, score: float):
                nonlocal armed_recording
                print(f"\n[MOTHER] Wake word detected ({keyword}, score: {score:.2f}) — listening...")
                # Play short ascending tone (200ms)
                try:
                    _tone_sr = 16000
                    _t = np.linspace(0, 0.2, int(_tone_sr * 0.2), dtype=np.float32)
                    _freq = np.linspace(600, 1200, len(_t))  # ascending 600→1200 Hz
                    _tone = 0.3 * np.sin(2 * np.pi * _freq * _t / _tone_sr)
                    _sd_tts.play(_tone, _tone_sr)
                except Exception:
                    pass
                armed_recording = True

            _wake_detector = _WakeWordDetector(
                sensitivity=float(_os_ptt.environ.get("WAKE_WORD_SENSITIVITY", "0.5")),
                on_detected=_on_wake_word,
            )
            _wake_detector.start()
            print(f"[WakeWord] Listening for '{_wake_detector.model_name}' (say it or press Enter)")
        except Exception as _ww_err:
            print(f"[WakeWord] Init failed ({_ww_err}) — Enter key only")
            _wake_detector = None

        print("\nPress Enter to start; Enter again to stop. Ctrl+C to exit.")
        # Start reminder background thread and register speak callback
        register_speak(lambda t: speak_now(t))
        start_reminder_thread()
        # Warm the RAG server cache at startup (non-blocking, short timeout)
        try:
            _http.get(
                f"{getattr(cfg, 'rag', None).api_base or 'http://127.0.0.1:8123'}/warmup",  # type: ignore[attr-defined]
                timeout=0.4,
            )
        except Exception:
            pass
        try:
            # If stdin isn't interactive, fall back to one-shot timed capture
            if not sys.stdin or not sys.stdin.isatty():
                import sounddevice as sd
                import numpy as np
                duration = max(1, int(args.ptt_seconds))
                print(f"No interactive stdin detected. Recording {duration}s...")
                audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
                sd.wait()
                mic_text = stt.transcribe_pcm(audio[:, 0], samplerate)
                print(f"You said: {mic_text}")

                user_input = normalize_mishearings(mic_text)
                # On-demand RAG enrichment
                notes_context = ""
                if getattr(cfg, "rag", None) and cfg.rag.enabled and cfg.rag.on_demand:
                    try:
                        resp = _http.get(
                            f"{cfg.rag.api_base}/search",
                            params={"q": user_input, "k": cfg.rag.k},
                            timeout=cfg.rag.timeout_ms / 1000.0,
                        )
                        if resp.status_code == 200:
                            hits = resp.json()[: cfg.rag.k]
                            parts = [f"- {h.get('text','')} (src: {h.get('path','')})" for h in hits]
                            notes_context = "\n".join(parts)
                    except Exception:
                        pass
                sys_prompt = cfg.llm.system_prompt
                if notes_context:
                    sys_prompt += f"\n\nNotes context (prefer these facts; keep reply concise):\n{notes_context}"
                messages = [
                    ChatMessage(role="system", content=sys_prompt),
                    ChatMessage(role="user", content=user_input),
                ]

                import re as _re
                boundary = _re.compile(r"[.!?](?:\s|$)|\n")
                sentence_buffer = ""
                for chunk in llm.stream_chat(
                    messages,
                    temperature=cfg.llm.temperature,
                    max_tokens=cfg.llm.max_tokens,
                ):
                    print(chunk, end="", flush=True)
                    sentence_buffer += chunk
                    while True:
                        m = boundary.search(sentence_buffer)
                        if not m:
                            break
                        seg = sentence_buffer[: m.end()].strip()
                        sentence_buffer = sentence_buffer[m.end() :]
                        if seg:
                            speak_now(seg)
                print()
                tail = sentence_buffer.strip()
                if tail:
                    speak_now(tail)
                return 0

            def drain_keys():
                while msvcrt.kbhit():
                    _ = msvcrt.getwch()

            def wait_enter_keydown(min_delay: float = 0.0):
                start = time.monotonic()
                while True:
                    if armed_recording:
                        return  # wake word fired — break out immediately
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch == "\r" and (time.monotonic() - start) >= min_delay:
                            return
                    time.sleep(0.01)

            def wait_for_release_then_next_enter(idle_ms: float = 350.0):
                """Ignore auto-repeat: require a quiet period, then a fresh Enter.

                idle_ms: duration with no key events to consider the key released.
                Also breaks out if armed_recording is set (wake word triggered).
                """
                idle_s = idle_ms / 1000.0
                last_event = time.monotonic()
                # Drain any repeating key events until there's a quiet gap
                while True:
                    if armed_recording:
                        return
                    if msvcrt.kbhit():
                        _ = msvcrt.getwch()
                        last_event = time.monotonic()
                    if time.monotonic() - last_event >= idle_s:
                        break
                    time.sleep(0.01)
                # Now wait for the next deliberate Enter press
                while True:
                    if armed_recording:
                        return
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch == "\r":
                            return
                    time.sleep(0.01)

            while True:
                try:
                    if armed_recording:
                        # consume the flag so we start capturing immediately
                        armed_recording = False
                    else:
                        wait_enter_keydown()  # start by keydown
                except EOFError:
                    # Fall back to timed capture once
                    import sounddevice as sd
                    import numpy as np
                    duration = max(1, int(args.ptt_seconds))
                    print(f"EOF on stdin. Recording {duration}s...")
                    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
                    sd.wait()
                    mic_text = stt.transcribe_pcm(audio[:, 0], samplerate)
                    print(f"You said: {mic_text}")

                    user_input = mic_text
                    messages = [
                        ChatMessage(role="system", content=cfg.llm.system_prompt),
                        ChatMessage(role="user", content=user_input),
                    ]

                    import re as _re
                    boundary = _re.compile(r"[.!?](?:\s|$)|\n")
                    sentence_buffer = ""
                    for chunk in llm.stream_chat(
                        messages,
                        temperature=cfg.llm.temperature,
                        max_tokens=cfg.llm.max_tokens,
                    ):
                        print(chunk, end="", flush=True)
                        sentence_buffer += chunk
                        while True:
                            m = boundary.search(sentence_buffer)
                            if not m:
                                break
                            seg = sentence_buffer[: m.end()].strip()
                            sentence_buffer = sentence_buffer[m.end() :]
                            if seg:
                                speak_now(seg)
                    print()
                    tail = sentence_buffer.strip()
                    if tail:
                        speak_now(tail)
                    return 0
                t0 = time.monotonic()
                on_enter()

                # Wait for key release (quiet gap), then next deliberate Enter
                wait_for_release_then_next_enter(350)
                on_enter()

                if not recording:
                    print("No audio captured.")
                    continue
                audio = np.concatenate(recording, axis=0).squeeze()

                # Launch speaker identification in background — runs in parallel with STT
                # Both only need the audio array so they can overlap completely.
                session = get_session()
                _spk_result: list = [None, 0.0]  # [user_id, confidence]
                _spk_thread = None
                if not session.is_identified() and registry.list_users():
                    def _spk_identify(_a=audio, _sr=samplerate):
                        sid, sconf = identify_from_audio(_a, _sr)
                        _spk_result[0] = sid
                        _spk_result[1] = sconf
                    _spk_thread = threading.Thread(target=_spk_identify, daemon=True)
                    _spk_thread.start()

                t_stt0 = time.monotonic()
                if audio.size > 0 and _streaming_stt is not None:
                    try:
                        mic_text = _asyncio_stt.run(
                            _streaming_stt.transcribe_audio(audio, samplerate)
                        )
                    except Exception as _stt_e:
                        print(f"[STT] Streaming failed ({_stt_e}) — falling back")
                        mic_text = stt.transcribe_pcm(audio, samplerate)
                elif audio.size > 0:
                    mic_text = stt.transcribe_pcm(audio, samplerate)
                else:
                    mic_text = ""
                t_stt1 = time.monotonic()
                print(f"You said: {mic_text}")

                user_input = mic_text

                # Collect speaker identification result (STT took 2-5s; should be done)
                if _spk_thread is not None:
                    _spk_thread.join(timeout=0.5)
                    speaker_id, speaker_conf = _spk_result[0], _spk_result[1]
                    if speaker_id and speaker_conf > 0.75:
                        profile = registry.get_user(speaker_id)
                        if profile:
                            set_current_user(speaker_id, confidence=speaker_conf, method="voice")
                            print(f"[Identified: {profile.display_name} ({speaker_conf*100:.0f}%)]")
                            # Restore this user's conversation history
                            get_memory().load(speaker_id)
                    elif speaker_conf > 0.5:
                        print(f"[Voice match uncertain ({speaker_conf*100:.0f}%)]")
                
                # Analyze urgency from text and audio
                urgency_detector = get_urgency_detector()
                text_urgent, text_score = urgency_detector.analyze_text_urgency(user_input)
                audio_urgent, audio_score, _ = urgency_detector.analyze_audio_urgency(
                    audio, samplerate, 
                    duration_seconds=len(audio) / samplerate,
                    word_count=len(user_input.split()) if user_input else 0
                )
                urgency_score = max(text_score, audio_score)
                context = get_context()
                context.update_interaction(urgency_score)
                
                # Quick acknowledgment for urgent requests
                if urgency_score > 0.5:
                    ack = get_contextual_acknowledgment(urgency_score)
                    if ack:
                        print(f"[MOTHER] {ack}")
                        speak_now(ack)
                # Check for explicit user identification ("this is Oliver", "I'm Oliver")
                # Skip if already identified with good confidence from voice
                try:
                    low = (user_input or "").lower()
                except Exception:
                    low = ""
                
                import re as _re_user
                # Common false positives to ignore (words that follow "I am" but aren't names)
                _FALSE_POSITIVE_NAMES = {
                    "already", "not", "going", "trying", "here", "there", "ready",
                    "sorry", "happy", "sad", "fine", "good", "okay", "ok", "sure",
                    "just", "also", "still", "now", "back", "done", "enrolled",
                    "looking", "asking", "wondering", "thinking", "speaking", "talking"
                }
                user_id_match = _re_user.search(r"(?:this is|i'?m|i am|my name is)\s+(\w+)", low)
                if user_id_match:
                    claimed_name = user_id_match.group(1)
                    # Skip false positives
                    if claimed_name in _FALSE_POSITIVE_NAMES:
                        pass  # Not a real name introduction, continue normal flow
                    else:
                        # Check if this matches an enrolled user
                        found_user = False
                        for uid in registry.list_users():
                            profile = registry.get_user(uid)
                            if profile and (claimed_name == uid or claimed_name == profile.display_name.lower()):
                                set_current_user(uid, confidence=1.0, method="explicit")
                                response = f"Hello, {profile.display_name}. I've switched to your profile."
                                print(f"[MOTHER] {response}")
                                speak_now(response)
                                found_user = True
                                drain_keys(); print("Listening... (press Enter to stop)"); break
                        if found_user:
                            continue
                        # Not found - only offer to enroll if it looks like a real name (capitalized, 2+ chars)
                        # Re-match against original (non-lowercased) input to check capitalisation
                        _orig_match = _re_user.search(r"(?:this is|i'?m|i am|my name is)\s+(\w+)", user_input)
                        _orig_name = _orig_match.group(1) if _orig_match else claimed_name
                        if len(_orig_name) >= 2 and _orig_name[0].isupper():
                            response = f"I don't have a profile for {_orig_name}. Would you like to enroll?"
                            print(f"[MOTHER] {response}")
                            speak_now(response)
                            drain_keys(); print("Listening... (press Enter to stop)"); continue
                
                # Identity questions - "who am I", "do you know who I am"
                if any(phrase in low for phrase in ["who am i", "do you know who i am", "can you identify me", "who is speaking"]):
                    current_user = get_current_user()
                    if current_user:
                        response = f"You are {current_user.display_name}. I identified you by your voice."
                    else:
                        response = "I haven't been able to identify you yet. Please speak so I can try to match your voice, or say your name."
                    print(f"[MOTHER] {response}")
                    speak_now(response)
                    drain_keys(); print("Listening... (press Enter to stop)"); continue
                
                # Memory commands - "what do you know/remember about me"
                if any(phrase in low for phrase in ["what do you know about me", "what do you remember", "what have you learned"]):
                    memory = get_current_user_memory()
                    if memory:
                        summary = memory.get_memory_summary(max_facts=8, max_episodic=5)
                        if summary and summary != "No memories stored yet.":
                            response = f"Here's what I know: {summary[:300]}..."
                        else:
                            response = "I haven't learned much about you yet. Tell me about yourself!"
                    else:
                        response = "I need to identify you first to access your memories."
                    print(f"[MOTHER] {response}")
                    speak_now(response)
                    drain_keys(); print("Listening... (press Enter to stop)"); continue
                
                # Explicit "remember" commands
                remember_match = _re_user.search(r"remember (?:that )?(?:my )?(.+)", low)
                if remember_match:
                    memory = get_current_user_memory()
                    if memory:
                        to_remember = remember_match.group(1).strip()
                        # Try to extract as structured fact
                        extracted = extract_fact_from_statement(f"my {to_remember}")
                        if extracted:
                            key, value, category = extracted
                            memory.set_fact(key, value, category=category, source="explicit")
                            response = f"Got it. I'll remember that your {key} is {value}."
                        else:
                            # Store as episodic memory
                            memory.add_episodic(to_remember, tags=["explicit"], confidence=1.0, source="explicit_request")
                            response = f"I'll remember that."
                    else:
                        response = "I need to identify you first before I can remember things for you."
                    print(f"[MOTHER] {response}")
                    speak_now(response)
                    drain_keys(); print("Listening... (press Enter to stop)"); continue
                
                # Background passive learning — never blocks the main response path
                if get_current_user():
                    _learn_thread = threading.Thread(
                        target=lambda: [print(f"[Memory] {i}") for i in maybe_learn_from_statement(user_input, llm_fn=llm.chat)],
                        daemon=True,
                    )
                    _learn_thread.start()

                # Classify intent — keyword-only in hot path (LLM fallback removed; ~1s saved)
                _intent = classify_intent(user_input)

                # Pre-fetch: compute conv context and kick off RAG in background so it
                # runs in parallel with the fast-path dispatch checks below.
                conv_mem = get_memory()
                _conv_summary = conv_mem.summarize_for_context(max_chars=200)
                _rag_query = f"{user_input} {_conv_summary}".strip() if _conv_summary else user_input
                _rag_bg_hits: list = []
                _rag_bg_lock = threading.Lock()
                _rag_bg_thread = None
                if getattr(cfg, "rag", None) and cfg.rag.enabled and cfg.rag.on_demand:
                    def _rag_bg_fetch(_q=_rag_query):
                        try:
                            r = _http.get(
                                f"{cfg.rag.api_base}/search",
                                params={"q": _q, "k": min(cfg.rag.k, 2)},
                                timeout=cfg.rag.timeout_ms / 1000.0,
                            )
                            if r.status_code == 200:
                                with _rag_bg_lock:
                                    _rag_bg_hits.extend(r.json()[:min(cfg.rag.k, 2)])
                        except Exception:
                            pass
                    _rag_bg_thread = threading.Thread(target=_rag_bg_fetch, daemon=True)
                    _rag_bg_thread.start()

                # Reminder commands
                if _intent in (Intent.REMINDER_SET, Intent.REMINDER_LIST):
                    _cu = get_current_user()
                    if _intent == Intent.REMINDER_LIST:
                        _pending = list_reminders(_cu.user_id if _cu else None)
                        if _pending:
                            _lines = [r.get("text", "?") for r in _pending[:5]]
                            response = "Your reminders: " + "; ".join(_lines) + "."
                        else:
                            response = "You have no pending reminders."
                        speak_now(response); drain_keys(); print("Listening... (press Enter to stop)"); continue
                    else:  # REMINDER_SET
                        parsed = parse_reminder(user_input)
                        if parsed:
                            _rtext, _rtime = parsed
                            _uid = _cu.user_id if _cu else "unknown"
                            _rid = add_reminder(_uid, _rtext, _rtime)
                            _when = _rtime.strftime("%I:%M %p").lstrip("0")
                            response = f"Reminder set for {_when}: {_rtext}."
                        else:
                            response = "I couldn't parse that reminder. Try: remind me at 3 PM to call the dentist."
                        speak_now(response); drain_keys(); print("Listening... (press Enter to stop)"); continue

                # Finance voice triggers (price/news) – fast path, only when asked
                # Early finance handling so it doesn't get captured by info search
                if _intent in (Intent.FINANCE_QUOTE, Intent.FINANCE_NEWS) or \
                   ("price" in low or "quote" in low or "trading at" in low or "stock price" in low):
                    symbol = resolve_symbol_from_text(user_input)
                    if not symbol:
                        # Quick clarify on low-certainty finance keyword without symbol
                        if speak_now("Which symbol did you mean?"):
                            armed_recording = True
                            drain_keys(); print("Listening... (press Enter to stop)"); continue
                        drain_keys(); print("Listening... (press Enter to stop)"); continue
                    if symbol:
                        try:
                            resp = _http.get(
                                f"{getattr(cfg, 'rag', None).api_base or 'http://127.0.0.1:8123'}/finance/quote",  # type: ignore[attr-defined]
                                params={"symbol": symbol},
                                timeout=0.5,
                            )
                            data = resp.json() if resp.status_code == 200 else {}
                        except Exception:
                            data = {}
                        if (not data or data.get("regularMarketPrice") is None) and _local_get_quote:
                            try:
                                data = _local_get_quote(symbol) or {}
                            except Exception:
                                pass
                        price = data.get("regularMarketPrice")
                        curr = data.get("currency") or "USD"
                        name = data.get("shortName") or symbol
                        say = f"{name} is {money_to_words(float(price), curr)}." if price is not None else "Price unavailable."
                        speak_now(say)
                        drain_keys(); print("Listening... (press Enter to stop)"); continue
                # Weather trigger (simple keyword + optional known locations)
                if _intent == Intent.WEATHER or ("weather" in low) or ("temperature" in low) or ("forecast" in low):
                    lat, lon = 43.0389, -87.9065  # Milwaukee default
                    loc_map = {
                        "milwaukee": (43.0389, -87.9065),
                        "madison": (43.0731, -89.4012),
                        "wisconsin": (44.5000, -89.5000),
                    }
                    for key, coords in loc_map.items():
                        if key in low:
                            lat, lon = coords
                            break
                    try:
                        data = weather_get(lat, lon, fahrenheit=True, mph=True)
                    except Exception:
                        data = {"error": "weather failed"}
                    if not data.get("error"):
                        temp = data.get("temperature")
                        wind = data.get("windspeed")
                        speak_now(f"The temperature is {round(temp)} degrees Fahrenheit with wind {round(wind)} miles per hour.")
                    else:
                        speak_now("Sorry, I couldn't get the weather.")
                    drain_keys(); print("Listening... (press Enter to stop)"); continue
                # Info search trigger (Wikipedia/DDG first; RAG fallback; if empty → general LLM)
                # Intent classifier can route here directly even without a keyword match
                t_rag0 = time.monotonic()
                info_q = extract_info_query(normalize_mishearings(user_input))
                if not info_q and _intent == Intent.INFO_SEARCH:
                    info_q = user_input.strip()
                if info_q:
                    # context-carry: resolve pronoun queries to last person within 60s
                    if has_pronoun_reference(info_q):
                        if not hasattr(main, "_last_person_name") or (_time.time() - getattr(main, "_last_person_ts", 0)) > 60:
                            # no valid context; continue normal flow
                            pass
                        else:
                            info_q = info_q.replace("his", getattr(main, "_last_person_name")).replace("her", getattr(main, "_last_person_name")).replace("their", getattr(main, "_last_person_name"))
                    try:
                        import concurrent.futures as _cf
                        with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                            _fut = _pool.submit(info_get_info, info_q)
                            try:
                                info = _fut.result(timeout=4.0)
                            except _cf.TimeoutError:
                                info = {"error": "timeout"}
                    except Exception:
                        info = {"error": "No info found"}
                    if not info.get("error"):
                        # lore answers should be extra concise
                        mx = 160 if is_lore_query(info_q) else 320
                        if speak_now(shorten_summary(info.get("summary", ""), max_chars=mx)):
                            # If interrupted, arm immediate recording
                            armed_recording = True
                            drain_keys(); print("Listening... (press Enter to stop)"); continue
                        drain_keys(); print("Listening... (press Enter to stop)"); continue
                    # Fallback to local RAG snippet
                    local_hits = []
                    if getattr(cfg, "rag", None):
                        try:
                            # If the name looks like a person, constrain to people/ subdir to avoid cross-person bleed
                            path_contains = None
                            import re as _re4
                            if _re4.search(r"\b(oliver|kieran|adil|grace|mckenzie)\b", info_q.lower()):
                                path_contains = "notes/people/"
                            # lore catch: route to lore folder
                            elif is_lore_query(info_q) or is_lore_query(user_input):
                                path_contains = "notes/lore/"
                            r = _http.get(
                                f"{cfg.rag.api_base}/search",  # type: ignore[union-attr]
                                params={k: v for k, v in ({"q": info_q, "k": 4, "path_contains": path_contains} if path_contains else {"q": info_q, "k": 4}).items()},
                                timeout=(getattr(cfg, "rag", None).timeout_ms or 500) / 1000.0,  # type: ignore[attr-defined]
                            )
                            if r.status_code == 200:
                                local_hits = r.json() or []
                        except Exception:
                            local_hits = []
                    t_rag1 = time.monotonic()
                    if local_hits:
                        # pick the correct person hit
                        best_hit = _select_best_hit_for_query(local_hits, user_input)
                        if not best_hit:
                            # Fall through to general LLM handling below
                            info_q = None  # signal to continue to general LLM
                        # lore vs person answers
                        if is_lore_query(user_input) or is_lore_query(info_q or ""):
                            ans = answer_from_lore(user_input, best_hit) or shorten_summary(best_hit.get("text", ""), max_chars=160)
                        else:
                            ans = answer_from_rag(user_input, [best_hit])
                        # update context person based on the best hit
                        who = extract_name_from_hit(best_hit)
                        if who:
                            setattr(main, "_last_person_name", who)
                            setattr(main, "_last_person_ts", _time.time())
                        # If it's not a person-style QA, avoid rambling: ask to repeat
                        if not ans and not is_person_qa_query(user_input):
                            # Provide a concise bio instead of reading entire note
                            if speak_now(summarize_person_hit(best_hit)):
                                armed_recording = True
                                drain_keys(); print("Listening... (press Enter to stop)"); continue
                            drain_keys(); print("Listening... (press Enter to stop)"); continue
                        if not ans and is_person_qa_query(user_input):
                            ans = fallback_line_from_text(user_input, best_hit.get("text", ""))
                        if speak_now(ans or shorten_summary(best_hit.get("text", ""), max_chars=160 if is_lore_query(user_input) else 320)):
                            armed_recording = True
                            drain_keys(); print("Listening... (press Enter to stop)"); continue
                        drain_keys(); print("Listening... (press Enter to stop)"); continue
                    else:
                        # Nothing useful from web or RAG: fall through to general LLM
                        info_q = None
                # If info path produced nothing usable, proceed to general LLM below
                # Finance news/top stories
                if ("finance" in low) and ("news" in low or "stories" in low or "headlines" in low):
                    titles: list[str] = []
                    def _fetch(sym: str) -> list[str]:
                        try:
                            r = _http.get(
                                f"{getattr(cfg, 'rag', None).api_base or 'http://127.0.0.1:8123'}/finance/news",  # type: ignore[attr-defined]
                                params={"symbol": sym, "count": 3},
                                timeout=0.6,
                            )
                            items = r.json() if r.status_code == 200 else []
                            return [i.get("title", "") for i in items if isinstance(i, dict)]
                        except Exception:
                            return []
                    for sym in ("^GSPC", "^DJI", "^IXIC"):
                        titles = _fetch(sym)
                        if titles:
                            break
                    if speak_now("; ".join(titles[:3]) or "No finance news available."):
                        armed_recording = True
                        drain_keys(); print("Listening... (press Enter to stop)"); continue
                    drain_keys()
                    print("Listening... (press Enter to stop)")
                    continue
                # Collect RAG result from background fetch started after intent classification
                notes_context = ""
                if _rag_bg_thread is not None:
                    if _rag_bg_thread.is_alive():
                        _rag_bg_thread.join(timeout=0.08)  # already ~500ms in flight; usually done
                    with _rag_bg_lock:
                        if _rag_bg_hits:
                            parts = [f"- {h.get('text','')} (src: {h.get('path','')})" for h in _rag_bg_hits]
                            notes_context = "\n".join(parts)
                # Build context-aware system prompt with user identity
                sys_prompt = build_context_aware_prompt(
                    cfg.llm.system_prompt,
                    user_text=user_input,
                    urgency_score=urgency_score
                )

                # Note: conversation history is already in conv_mem.get_messages() below —
                # no need to re-inject a lossy summary into the system prompt.

                # Add user-specific context and memory
                current_user = get_current_user()
                if current_user:
                    user_context = get_user_context_for_prompt(current_user)
                    if user_context:
                        sys_prompt += f"\n\nUser context: {user_context}"

                    # Add relevant memories to context
                    user_mem = get_current_user_memory()
                    if user_mem:
                        memory_context = user_mem.get_context_for_prompt(user_input, max_items=3)
                        if memory_context:
                            sys_prompt += f"\n\nMemory context: {memory_context}"

                if notes_context:
                    sys_prompt += f"\n\nNotes context (prefer these facts; keep reply concise):\n{notes_context}"

                # Build messages with conversation history for context
                conv_mem.add_user(user_input)
                messages = conv_mem.get_messages(sys_prompt)

                import re as _re
                import queue as _tts_pq
                boundary = _re.compile(r"[.!?](?:\s|$)|\n")
                sentence_buffer = ""
                full_response = []

                # ----- Background TTS synthesis pipeline -----
                # Sentences are enqueued here as the LLM streams tokens.
                # A background thread synthesizes them concurrently so that
                # by the time the LLM finishes, audio is already ready to play.
                _synth_in: _tts_pq.Queue = _tts_pq.Queue()
                _synth_out: _tts_pq.Queue = _tts_pq.Queue(maxsize=2)

                def _synth_pipeline_worker():
                    while True:
                        txt = _synth_in.get()
                        if txt is None:
                            _synth_out.put(None)
                            return
                        d, s = _synthesize(normalize_for_speech(txt).replace('\x00', '').replace('\ufffd', ''))
                        _synth_out.put((d, s) if d is not None else None)

                _synth_th = threading.Thread(target=_synth_pipeline_worker, daemon=True)
                _synth_th.start()

                # Build ToolContext for this turn
                _tool_ctx = ToolContext(
                    http_client=_http,
                    rag_base=getattr(cfg.rag, "api_base", "http://127.0.0.1:8123") if getattr(cfg, "rag", None) else "http://127.0.0.1:8123",
                    rag_timeout=getattr(cfg.rag, "timeout_ms", 500) / 1000.0 if getattr(cfg, "rag", None) else 0.5,
                    user_memory=get_current_user_memory(),
                    current_user=get_current_user(),
                    add_reminder_fn=add_reminder,
                )

                # Tiered routing: classify complexity → set tier
                if isinstance(llm, TieredLLMDriver):
                    from mother.llm.classifier import classify_complexity
                    _tier = classify_complexity(user_input, _intent.name)
                    llm.set_tier(_tier)
                    print(f"[LLM] Routing to {_tier} ({llm.current_model})")
                # Legacy hybrid routing
                elif isinstance(llm, HybridLLMDriver):
                    if _intent == Intent.GENERAL:
                        llm.route_to_cloud()
                    else:
                        llm.route_to_local()

                # Stream LLM tokens — enqueue complete sentences for background synthesis
                # Tools schema is passed when the model supports function calling
                t_llm0 = time.monotonic()
                _tool_call_intercepted = False
                for chunk in llm.stream_chat(
                    messages,
                    temperature=cfg.llm.temperature,
                    max_tokens=cfg.llm.max_tokens,
                    tools=TOOLS_SCHEMA,
                ):
                    # Detect tool-call sentinel emitted by llm.py
                    if chunk.startswith("__TOOL_CALL__:"):
                        _tool_call_intercepted = True
                        import json as _tc_json
                        try:
                            _tc_data = _tc_json.loads(chunk[len("__TOOL_CALL__:"):])
                            _tc_calls = _tc_data.get("message", {}).get("tool_calls", [])
                            for _tc in _tc_calls:
                                _tc_name = _tc.get("function", {}).get("name", "")
                                _tc_args = _tc.get("function", {}).get("arguments", {})
                                if isinstance(_tc_args, str):
                                    try:
                                        _tc_args = _tc_json.loads(_tc_args)
                                    except Exception:
                                        _tc_args = {}
                                _tc_result = dispatch_tool_call(_tc_name, _tc_args, _tool_ctx)
                                print(f"\n[Tool: {_tc_name}] {_tc_result}")
                                # Feed result back as a tool-result message then re-prompt
                                import copy as _copy
                                _followup_msgs = _copy.copy(messages)
                                _followup_msgs.append(ChatMessage(role="tool", content=_tc_result))
                                _followup_msgs.append(ChatMessage(role="user", content="Summarise this for me in one or two sentences."))
                                for _fc in llm.stream_chat(_followup_msgs, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens):
                                    print(_fc, end="", flush=True)
                                    full_response.append(_fc)
                                    if cfg.tts.stream_sentences:
                                        sentence_buffer += _fc
                                        while True:
                                            _bm = boundary.search(sentence_buffer)
                                            if not _bm:
                                                break
                                            _seg = sentence_buffer[:_bm.end()].strip()
                                            sentence_buffer = sentence_buffer[_bm.end():]
                                            if _seg:
                                                _synth_in.put(_seg)
                        except Exception as _tc_err:
                            print(f"\n[Tool dispatch error] {_tc_err}")
                        continue
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
                    if cfg.tts.stream_sentences:
                        sentence_buffer += chunk
                        while True:
                            m = boundary.search(sentence_buffer)
                            if not m:
                                break
                            seg = sentence_buffer[: m.end()].strip()
                            sentence_buffer = sentence_buffer[m.end():]
                            if seg:
                                _synth_in.put(seg)  # non-blocking; LLM keeps streaming
                print()
                t_llm1 = time.monotonic()
                conv_mem.add_assistant("".join(full_response))
                # Persist conversation history per-user (non-blocking fire-and-forget)
                _cu = get_current_user()
                if _cu:
                    try:
                        conv_mem.save(_cu.user_id)
                    except Exception:
                        pass

                # Enqueue remaining tail sentence (if any), then sentinel
                if cfg.tts.stream_sentences and sentence_buffer.strip():
                    _synth_in.put(sentence_buffer.strip())
                elif not cfg.tts.stream_sentences:
                    full_text = "".join(full_response).strip()
                    if full_text:
                        _synth_in.put(full_text)
                _synth_in.put(None)  # sentinel → worker exits after draining

                # Play synthesized audio. Most sentences will already be ready
                # because synthesis ran concurrently with LLM generation.
                t_tts0 = time.monotonic()
                _tts_interrupted = False
                while not _tts_interrupted:
                    item = _synth_out.get()  # blocks only if synthesis not yet done
                    if item is None:
                        break
                    _d, _s = item
                    if _d is not None and _play_audio(_d, _s):
                        _tts_interrupted = True
                        armed_recording = True
                        # drain pre-synthesised queue so worker thread can exit
                        while True:
                            try:
                                _synth_out.get_nowait()
                            except _tts_pq.Empty:
                                break
                t_tts1 = time.monotonic()

                # Latency summary (llm= is now pure generation time, tts= is playback)
                try:
                    total = time.monotonic() - t0
                    _rag_s = (t_rag1 - t_rag0) if 't_rag1' in locals() else 0.0
                    print(f"[latency] stt={t_stt1-t_stt0:.3f}s rag={_rag_s:.3f}s llm={t_llm1-t_llm0:.3f}s tts={t_tts1-t_tts0:.3f}s total={total:.3f}s")
                except Exception:
                    pass
                # Re-arm
                drain_keys()
                print("Listening... (press Enter to stop)")
        except KeyboardInterrupt:
            print()  # newline after Ctrl+C
            return 0

    elif args.auto:
        try:
            import sounddevice as sd
            import numpy as np
            import time
            import webrtcvad
        except ImportError:
            print("webrtcvad and sounddevice are required for --auto. Install dependencies.")
            return 1
        if stt is None:
            print("No STT configured in configs/app.yaml under 'stt'.")
            return 1

        samplerate = 16000
        vad = webrtcvad.Vad(3)  # 0-3, 3=aggressive
        frame_ms = 20
        frame_samples = int(samplerate * frame_ms / 1000)
        silence_ms = getattr(cfg.stt, "vad_silence_ms", 350)
        max_silence_frames = max(1, int(silence_ms / frame_ms))
        silence_count = 0
        # Simple RMS/energy gate to suppress quiet background as speech
        energy_threshold = 0.015
        print("Speak anytime. Auto-stops on brief silence. Ctrl+C to exit.")

        def frames_from_stream():
            with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16") as stream:
                buf = b""
                while True:
                    data, _ = stream.read(frame_samples)
                    pcm = (data[:, 0]).astype("int16").tobytes()
                    yield pcm

        def audio_chunks():
            nonlocal silence_count
            import numpy as np
            for pcm in frames_from_stream():
                samples = np.frombuffer(pcm, dtype="int16").astype("float32") / 32768.0
                rms = float(np.sqrt(np.mean(np.square(samples))) if samples.size else 0.0)
                vad_speech = vad.is_speech(pcm, samplerate)
                is_speech = vad_speech and (rms >= energy_threshold)
                if is_speech:
                    silence_count = 0
                else:
                    silence_count = min(10**9, silence_count + 1)
                yield samples
                if silence_count >= max_silence_frames:
                    return

        # Get first partial transcription chunk(s) and start LLM
        partials = []
        for text in stt.stream_transcribe(audio_chunks(), samplerate, language=cfg.stt.language):
            if text:
                partials.append(text)
            if len(partials) >= 1:
                break
        user_input = " ".join(partials).strip()
        if not user_input:
            print("No speech detected.")
            return 0
        print(f"You said (partial): {user_input}")

        # Build memory-enriched system prompt (same pipeline as PTT mode)
        auto_sys_prompt = cfg.llm.system_prompt
        _auto_user = get_current_user()
        if _auto_user:
            # Passive learning from this utterance
            for _item in maybe_learn_from_statement(user_input):
                print(f"[Memory] {_item}")
            _auto_user_ctx = get_user_context_for_prompt(_auto_user)
            if _auto_user_ctx:
                auto_sys_prompt += f"\n\nUser context: {_auto_user_ctx}"
            _auto_mem = get_current_user_memory()
            if _auto_mem:
                _auto_mem_ctx = _auto_mem.get_context_for_prompt(user_input, max_items=5)
                if _auto_mem_ctx:
                    auto_sys_prompt += f"\n\nMemory context: {_auto_mem_ctx}"

        messages = [
            ChatMessage(role="system", content=auto_sys_prompt),
            ChatMessage(role="user", content=user_input),
        ]

        import re as _re
        boundary = _re.compile(r"[.!?](?:\s|$)|\n")
        sentence_buffer = ""

        def speak_now(text_to_say: str) -> None:
            import os, tempfile, soundfile as sf, sounddevice as sd
            # Normalize text for natural speech
            text_to_say = normalize_for_speech(text_to_say)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            p = tmp.name
            try:
                tmp.close()
                tts.synthesize_to_file(text_to_say, p)
                data, sr = sf.read(p, dtype="float32")
                # Small fade to reduce clicks and clamp
                try:
                    import numpy as _np
                    n = max(1, int(sr * 0.005))
                    if data.ndim == 1:
                        n = min(n, max(1, data.shape[0] // 4))
                        ramp = _np.linspace(0.0, 1.0, n, dtype=data.dtype)
                        data[:n] *= ramp
                        data[-n:] *= ramp[::-1]
                    else:
                        n = min(n, max(1, data.shape[0] // 4))
                        ramp = _np.linspace(0.0, 1.0, n, dtype=data.dtype)[:, None]
                        data[:n, :] *= ramp
                        data[-n:, :] *= ramp[::-1]
                    _np.clip(data, -1.0, 1.0, out=data)
                except Exception:
                    pass
                sd.play(data, sr); sd.wait()
            finally:
                try: os.remove(p)
                except Exception: pass

        for chunk in llm.stream_chat(
            messages, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens,
        ):
            print(chunk, end="", flush=True)
            if cfg.tts.stream_sentences:
                sentence_buffer += chunk
                while True:
                    m = boundary.search(sentence_buffer)
                    if not m:
                        break
                    seg = sentence_buffer[: m.end()].strip()
                    sentence_buffer = sentence_buffer[m.end() :]
                    if seg:
                        speak_now(seg)
        print()
        tail = sentence_buffer.strip()
        if tail and cfg.tts.stream_sentences:
            speak_now(tail)
        return 0

    elif args.mic:
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as e:
            print("sounddevice is required for --mic. Install dependencies.")
            return 1
        if stt is None:
            print("No STT configured in configs/app.yaml under 'stt'.")
            return 1
        duration = max(1, int(args.mic_seconds))
        samplerate = 16000
        print(f"Recording {duration}s...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        mic_text = stt.transcribe_pcm(audio[:, 0], samplerate)
        print(f"You said: {mic_text}")
        args.text = mic_text
        args.skip_llm = False  # fall through to normal LLM path

    if args.skip_llm:
        final_text = args.text if args.text is not None else args.prompt
    else:
        import re as _re
        user_input = args.text if args.text is not None else args.prompt
        # Finance triggers in non-PTT path
        low = (user_input or "").lower()
        # Weather in non-PTT
        low = (user_input or "").lower()
        if ("weather" in low) or ("temperature" in low) or ("forecast" in low):
            lat, lon = 43.0389, -87.9065
            loc_map = {"milwaukee": (43.0389, -87.9065), "madison": (43.0731, -89.4012), "wisconsin": (44.5000, -89.5000)}
            for key, coords in loc_map.items():
                if key in low:
                    lat, lon = coords
                    break
            try:
                data = weather_get(lat, lon, fahrenheit=True, mph=True)
            except Exception:
                data = {"error": "weather failed"}
            if not data.get("error"):
                final_text = f"The temperature is {round(data.get('temperature'))} degrees Fahrenheit with wind {round(data.get('windspeed'))} miles per hour."
            else:
                final_text = "Sorry, I couldn't get the weather."
            if args.voice_out:
                out_path = Path(args.voice_out); out_path.parent.mkdir(parents=True, exist_ok=True)
                tts.synthesize_to_file(final_text, str(out_path)); print(f"Wrote audio to {out_path}")
            else:
                import os, tempfile, soundfile as sf, sounddevice as sd
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); p = tmp.name; tmp.close()
                try:
                    tts.synthesize_to_file(final_text, p); data_w, sr = sf.read(p, dtype="float32"); sd.play(data_w, sr); sd.wait(); print("Spoke response")
                finally:
                    try: os.remove(p)
                    except Exception: pass
            return 0
        # Info search in non-PTT (Wikipedia/DDG first; RAG fallback)
        info_q = extract_info_query(user_input)
        if info_q:
            if has_pronoun_reference(info_q):
                if hasattr(main, "_last_person_name") and (_time.time() - getattr(main, "_last_person_ts", 0)) <= 60:
                    info_q = info_q.replace("his", getattr(main, "_last_person_name")).replace("her", getattr(main, "_last_person_name")).replace("their", getattr(main, "_last_person_name"))
            try:
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                    _fut = _pool.submit(info_get_info, info_q)
                    try:
                        info = _fut.result(timeout=4.0)
                    except _cf.TimeoutError:
                        info = {"error": "timeout"}
            except Exception:
                info = {"error": "No info found"}
            if not info.get("error"):
                mx = 160 if is_lore_query(info_q) else 320
                final_text = shorten_summary(info.get("summary", ""), max_chars=mx)
                if args.voice_out:
                    out_path = Path(args.voice_out); out_path.parent.mkdir(parents=True, exist_ok=True)
                    tts.synthesize_to_file(final_text, str(out_path)); print(f"Wrote audio to {out_path}")
                else:
                    import os, tempfile, soundfile as sf, sounddevice as sd
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); p = tmp.name; tmp.close()
                    try:
                        tts.synthesize_to_file(final_text, p); data, sr = sf.read(p, dtype="float32"); sd.play(data, sr); sd.wait(); print("Spoke response")
                    finally:
                        try: os.remove(p)
                        except Exception: pass
                return 0
            # Fallback to local RAG snippet
            local_hits = []
            if getattr(cfg, "rag", None):
                try:
                    import re as _re4
                    path_contains = None
                    if _re4.search(r"\b(oliver|kieran|adil|grace|mckenzie)\b", info_q.lower()):
                        path_contains = "notes/people/"
                    params = {"q": info_q, "k": 4}
                    if path_contains:
                        params["path_contains"] = path_contains
                    r = _http.get(
                        f"{cfg.rag.api_base}/search",  # type: ignore[union-attr]
                        params=params,
                        timeout=(getattr(cfg, "rag", None).timeout_ms or 500) / 1000.0,  # type: ignore[attr-defined]
                    )
                    if r.status_code == 200:
                        local_hits = r.json() or []
                except Exception:
                    local_hits = []
            if local_hits:
                # pick the correct hit (person or lore)
                best_hit = _select_best_hit_for_query(local_hits, user_input)
                if not best_hit:
                    final_text = "I didn't catch that. Please repeat."
                    if args.voice_out:
                        out_path = Path(args.voice_out); out_path.parent.mkdir(parents=True, exist_ok=True)
                        tts.synthesize_to_file(final_text, str(out_path)); print(f"Wrote audio to {out_path}")
                    else:
                        import os, tempfile, soundfile as sf, sounddevice as sd
                        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); p = tmp.name; tmp.close()
                        try:
                            tts.synthesize_to_file(final_text, p); data, sr = sf.read(p, dtype="float32"); sd.play(data, sr); sd.wait(); print("Spoke response")
                        finally:
                            try: os.remove(p)
                            except Exception: pass
                    return 0
                if is_lore_query(user_input):
                    ans = answer_from_lore(user_input, best_hit)
                else:
                    ans = answer_from_rag(user_input, [best_hit])
                    who = extract_name_from_hit(best_hit)
                    if who:
                        setattr(main, "_last_person_name", who)
                        setattr(main, "_last_person_ts", _time.time())
                # If not clearly a person QA, avoid rambling: ask to repeat
                if not ans and not is_person_qa_query(user_input):
                    final_text = answer_from_lore(user_input, best_hit) or summarize_person_hit(best_hit)
                else:
                    if not ans and is_person_qa_query(user_input):
                        ans = fallback_line_from_text(user_input, best_hit.get("text", ""))
                    final_text = ans or shorten_summary(best_hit.get("text", ""), max_chars=160 if is_lore_query(user_input) else 320) or "No info found."
                if args.voice_out:
                    out_path = Path(args.voice_out); out_path.parent.mkdir(parents=True, exist_ok=True)
                    tts.synthesize_to_file(final_text, str(out_path)); print(f"Wrote audio to {out_path}")
                else:
                    import os, tempfile, soundfile as sf, sounddevice as sd
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); p = tmp.name; tmp.close()
                    try:
                        tts.synthesize_to_file(final_text, p); data, sr = sf.read(p, dtype="float32"); sd.play(data, sr); sd.wait(); print("Spoke response")
                    finally:
                        try: os.remove(p)
                        except Exception: pass
                return 0
            else:
                # Nothing useful from web or RAG: likely misheard — ask to repeat
                final_text = "I didn't catch that. Please repeat."
                if args.voice_out:
                    out_path = Path(args.voice_out); out_path.parent.mkdir(parents=True, exist_ok=True)
                    tts.synthesize_to_file(final_text, str(out_path)); print(f"Wrote audio to {out_path}")
                else:
                    import os, tempfile, soundfile as sf, sounddevice as sd
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); p = tmp.name; tmp.close()
                    try:
                        tts.synthesize_to_file(final_text, p); data, sr = sf.read(p, dtype="float32"); sd.play(data, sr); sd.wait(); print("Spoke response")
                    finally:
                        try: os.remove(p)
                        except Exception: pass
                return 0
        if ("price" in low or "quote" in low or "trading at" in low or "stock price" in low) or (("finance" in low) and ("news" in low or "stories" in low or "headlines" in low)):
            # Determine if quote or news
            if ("price" in low or "quote" in low or "trading at" in low or "stock price" in low):
                symbol = resolve_symbol_from_text(user_input)
                say = "Price unavailable."
                if symbol:
                    try:
                        resp = _http.get(
                            f"{getattr(cfg, 'rag', None).api_base or 'http://127.0.0.1:8123'}/finance/quote",  # type: ignore[attr-defined]
                            params={"symbol": symbol},
                            timeout=0.5,
                        )
                        data = resp.json() if resp.status_code == 200 else {}
                    except Exception:
                        data = {}
                    if (not data or data.get("regularMarketPrice") is None) and _local_get_quote:
                        try:
                            data = _local_get_quote(symbol) or {}
                        except Exception:
                            pass
                    price = data.get("regularMarketPrice")
                    curr = data.get("currency") or "USD"
                    name = data.get("shortName") or symbol
                    if price is not None:
                        say = f"{name} is {money_to_words(float(price), curr)}."
                final_text = say
                if args.voice_out:
                    out_path = Path(args.voice_out); out_path.parent.mkdir(parents=True, exist_ok=True)
                    tts.synthesize_to_file(final_text, str(out_path)); print(f"Wrote audio to {out_path}")
                else:
                    import os, tempfile, soundfile as sf, sounddevice as sd
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); p = tmp.name; tmp.close()
                    try:
                        tts.synthesize_to_file(final_text, p); data, sr = sf.read(p, dtype="float32"); sd.play(data, sr); sd.wait(); print("Spoke response")
                    finally:
                        try: os.remove(p)
                        except Exception: pass
                return 0
            else:
                titles: list[str] = []
                def _fetch(sym: str) -> list[str]:
                    try:
                        r = _http.get(
                            f"{getattr(cfg, 'rag', None).api_base or 'http://127.0.0.1:8123'}/finance/news",  # type: ignore[attr-defined]
                            params={"symbol": sym, "count": 3},
                            timeout=0.6,
                        )
                        items = r.json() if r.status_code == 200 else []
                        return [i.get("title", "") for i in items if isinstance(i, dict)]
                    except Exception:
                        return []
                for sym in ("^GSPC", "^DJI", "^IXIC"):
                    titles = _fetch(sym)
                    if titles:
                        break
                final_text = ". ".join(titles[:3]) or "No finance news available."
                if args.voice_out:
                    out_path = Path(args.voice_out); out_path.parent.mkdir(parents=True, exist_ok=True)
                    tts.synthesize_to_file(final_text, str(out_path)); print(f"Wrote audio to {out_path}")
                else:
                    import os, tempfile, soundfile as sf, sounddevice as sd
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); p = tmp.name; tmp.close()
                    try:
                        tts.synthesize_to_file(final_text, p); data, sr = sf.read(p, dtype="float32"); sd.play(data, sr); sd.wait(); print("Spoke response")
                    finally:
                        try: os.remove(p)
                        except Exception: pass
                return 0

        # Command trigger
        if getattr(cfg, "rag", None) and cfg.rag.enabled and cfg.rag.command_trigger and user_input.lower().startswith("search notes about"):
            topic = user_input.split("about", 1)[-1].strip() or user_input
            try:
                resp = _http.get(
                    f"{cfg.rag.api_base}/search",
                    params={"q": topic, "k": cfg.rag.k},
                    timeout=cfg.rag.timeout_ms / 1000.0,
                )
                hits = resp.json() if resp.status_code == 200 else []
            except Exception:
                hits = []
            lines = [f"- {h.get('text','')} (src: {h.get('path','')})" for h in hits]
            print("\n".join(lines) or "No results.")
            return 0

        # On-demand RAG enrichment
        notes_context = ""
        if getattr(cfg, "rag", None) and cfg.rag.enabled and cfg.rag.on_demand:
            try:
                resp = _http.get(
                    f"{cfg.rag.api_base}/search",
                    params={"q": user_input, "k": cfg.rag.k},
                    timeout=cfg.rag.timeout_ms / 1000.0,
                )
                if resp.status_code == 200:
                    hits = resp.json()[: cfg.rag.k]
                    parts = [f"- {h.get('text','')} (src: {h.get('path','')})" for h in hits]
                    if parts:
                        notes_context = "\n".join(parts)
            except Exception:
                pass

        sys_prompt = cfg.llm.system_prompt
        if notes_context:
            sys_prompt += f"\n\nNotes context (prefer these facts; keep reply concise):\n{notes_context}"

        messages = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_input),
        ]

        # Stream tokens; if not saving to file, speak per-sentence as they arrive
        sentence_buffer = ""
        boundary = _re.compile(r"[.!?](?:\s|$)|\n")
        final_chunks: List[str] = []

        def speak_now(text_to_say: str) -> None:
            import os
            import tempfile
            import soundfile as sf
            import sounddevice as sd

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            try:
                tmp.close()
                tts.synthesize_to_file(text_to_say, tmp_path)
                data, sr = sf.read(tmp_path, dtype="float32")
                sd.play(data, sr)
                sd.wait()
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        got_content = False
        for chunk in llm.stream_chat(
            messages,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        ):
            print(chunk, end="", flush=True)
            final_chunks.append(chunk)
            if chunk:
                got_content = True
            if not args.voice_out and cfg.tts.stream_sentences:
                sentence_buffer += chunk
                while True:
                    m = boundary.search(sentence_buffer)
                    if not m:
                        break
                    seg = sentence_buffer[: m.end()].strip()
                    sentence_buffer = sentence_buffer[m.end() :]
                    if seg:
                        speak_now(seg)
        print()
        final_text = "".join(final_chunks)

        # Fallback with RAG if nothing returned
        if getattr(cfg, "rag", None) and cfg.rag.enabled and cfg.rag.fallback_retry and not got_content:
            try:
                resp = _http.get(
                    f"{cfg.rag.api_base}/search",
                    params={"q": user_input, "k": cfg.rag.k},
                    timeout=cfg.rag.timeout_ms / 1000.0,
                )
                hits = resp.json() if resp.status_code == 200 else []
            except Exception:
                hits = []
            parts = [f"- {h.get('text','')} (src: {h.get('path','')})" for h in hits]
            if parts:
                retry_prompt = cfg.llm.system_prompt + "\n\nNotes context:\n" + "\n".join(parts)
                messages = [
                    ChatMessage(role="system", content=retry_prompt),
                    ChatMessage(role="user", content=user_input),
                ]
                final_chunks = []
                for chunk in llm.stream_chat(messages, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens):
                    print(chunk, end="", flush=True)
                    final_chunks.append(chunk)
                print()
                final_text = "".join(final_chunks)

    if args.voice_out:
        out_path = Path(args.voice_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tts.synthesize_to_file(final_text, str(out_path))
        print(f"Wrote audio to {out_path}")
    else:
        # If we already played sentence-by-sentence, nothing more to do
        if args.skip_llm:
            # For skip-LLM path, speak once directly
            import os
            import tempfile
            import soundfile as sf
            import sounddevice as sd

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            try:
                tmp.close()
                tts.synthesize_to_file(final_text, tmp_path)
                data, samplerate = sf.read(tmp_path, dtype="float32")
                sd.play(data, samplerate)
                sd.wait()
                print("Spoke response")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


