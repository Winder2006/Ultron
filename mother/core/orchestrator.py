"""Central async event loop for MOTHER.

Replaces cli.py as the main coordinator for all subsystems:
- Wake word detection (background thread)
- Push-to-talk via Enter key (async)
- STT transcription (Deepgram streaming / Whisper fallback)
- Speaker identification (parallel with STT)
- Intent classification + fast-path dispatch
- LLM tiered routing with streaming
- Sentence-boundary TTS pipeline (background synthesis)
- Passive memory learning (background thread)
- Reminder background thread
- Optional: WebSocket broadcast to dashboard via event bus

Usage:
    python -m mother --ptt          # push-to-talk (default)
    python -m mother --auto         # VAD auto-capture
    python -m mother --prompt "..."  # single prompt, no mic
    python -m mother --server       # API server mode (Phase 6)
"""
from __future__ import annotations

import asyncio
import io
import json
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from mother.config.settings import load_config, AppConfig
from mother.core.intent import classify as classify_intent, Intent
from mother.core.logging_config import get_logger
from mother.core.router import RequestRouter
from mother.handlers.fast_path import dispatch as fast_path_dispatch
from mother.llm.drivers import ChatMessage, LLMDriver, TieredLLMDriver, HybridLLMDriver
from mother.llm.tools import TOOLS_SCHEMA, ToolContext, dispatch_tool_call
from mother.tts.normalizer import normalize_for_speech

logger = get_logger("mother.orchestrator")

SAMPLE_RATE = 16000
SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s|$)|\n")


class Orchestrator:
    """Async event loop coordinating all MOTHER subsystems."""

    def __init__(self, config_path: str = "configs/app.yaml"):
        self.cfg = load_config(config_path)
        self._llm: Optional[LLMDriver] = None
        self._tts = None
        self._stt = None
        self._streaming_stt = None
        self._http = None
        self._wake_detector = None
        self._router = RequestRouter()

        # State
        self._armed_recording = False
        self._phrase_cache: dict = {}

        # Event bus for dashboard broadcast (optional)
        self._event_bus: Optional[asyncio.Queue] = None

    # ── Initialization ───────────────────────────────────────────────────────

    async def init(self):
        """Initialize all drivers and subsystems."""
        import httpx
        from mother.llm.factory import build_drivers

        self._llm, self._tts, self._stt = build_drivers(self.cfg)
        self._http = httpx.Client(timeout=5.0)

        # StreamingSTT
        try:
            from mother.audio.stt import StreamingSTT
            self._streaming_stt = StreamingSTT()
            await self._streaming_stt.init()
            engine = "Deepgram" if self._streaming_stt._deepgram_available else "Faster-Whisper"
            logger.info("StreamingSTT ready (engine: %s)", engine)
        except Exception as e:
            logger.warning("StreamingSTT init failed (%s) — using legacy STT", e)
            self._streaming_stt = None

        # Pre-synthesize canned phrases
        self._warm_phrase_cache()

        # Reminders
        try:
            from mother.core.reminders import register_speak, start_background_thread
            register_speak(lambda t: self._speak_sync(t))
            start_background_thread()
            logger.info("Reminder thread started")
        except Exception as e:
            logger.warning("Reminder init failed: %s", e)

        logger.info("Orchestrator initialized (LLM=%s, TTS=%s)",
                     self.cfg.llm.provider, self.cfg.tts.provider)

    def _warm_phrase_cache(self):
        """Pre-synthesize frequently used phrases for zero-latency acks."""
        phrases = [
            "One moment.", "Working on it.", "Understood.", "Acknowledged.",
            "Let me check.", "Stand by.", "On it.", "Right away.",
        ]
        for phrase in phrases:
            try:
                data, sr = self._synthesize(normalize_for_speech(phrase))
                if data is not None:
                    self._phrase_cache[phrase] = (data, sr)
            except Exception:
                pass

    # ── Audio helpers ────────────────────────────────────────────────────────

    def _synthesize(self, text: str):
        """Synthesize text → (numpy_array, sample_rate) or (None, None)."""
        import soundfile as sf
        try:
            wav = self._tts.synthesize_to_bytes(text)
            if not wav or len(wav) < 100:
                raise ValueError("Empty WAV")
            return sf.read(io.BytesIO(wav), dtype="float32")
        except Exception:
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                self._tts.synthesize_to_file(text, tmp_path)
                result = sf.read(tmp_path, dtype="float32")
                os.unlink(tmp_path)
                return result
            except Exception as e:
                logger.warning("TTS synthesis error: %s", e)
                return None, None

    def _play_audio(self, data, sr) -> bool:
        """Play audio with fade and Enter-interrupt. Returns True if interrupted."""
        import sounddevice as sd
        if data is None:
            return False
        try:
            n = max(1, int(sr * 0.005))
            if data.ndim == 1:
                n = min(n, max(1, data.shape[0] // 4))
                ramp = np.linspace(0.0, 1.0, n, dtype=data.dtype)
                data[:n] *= ramp
                data[-n:] *= ramp[::-1]
            else:
                n = min(n, max(1, data.shape[0] // 4))
                ramp = np.linspace(0.0, 1.0, n, dtype=data.dtype)[:, None]
                data[:n, :] *= ramp
                data[-n:, :] *= ramp[::-1]
            np.clip(data, -1.0, 1.0, out=data)
        except Exception:
            pass
        sd.play(data, sr)
        interrupted = False
        dur = max(0.0, float(len(data)) / float(sr)) if sr else 0.0
        t0 = time.monotonic()
        while (time.monotonic() - t0) < dur:
            if self._check_key_interrupt():
                sd.stop()
                interrupted = True
                break
            time.sleep(0.01)
        if not interrupted:
            try:
                sd.wait()
            except Exception:
                pass
        return interrupted

    def _check_key_interrupt(self) -> bool:
        """Check for Enter key press (non-blocking). Windows msvcrt."""
        if sys.platform == "win32":
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\r":
                    return True
        return False

    def _speak_sync(self, text: str) -> bool:
        """Synthesize + play. Returns True if interrupted."""
        if not text or not text.strip():
            return False
        text = normalize_for_speech(text).replace('\x00', '').replace('\ufffd', '')
        if not text.strip():
            return False
        cached = self._phrase_cache.get(text.strip())
        if cached:
            return self._play_audio(cached[0].copy(), cached[1])
        data, sr = self._synthesize(text)
        return self._play_audio(data, sr)

    # ── Transcription ────────────────────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio via StreamingSTT or legacy fallback."""
        if self._streaming_stt:
            try:
                return await self._streaming_stt.transcribe_audio(audio, SAMPLE_RATE)
            except Exception:
                pass
        if self._stt:
            return await asyncio.to_thread(self._stt.transcribe_pcm, audio, SAMPLE_RATE)
        return ""

    # ── Speaker ID ───────────────────────────────────────────────────────────

    async def _identify_speaker(self, audio: np.ndarray) -> tuple[Optional[str], float]:
        """Run speaker identification in a thread."""
        try:
            from mother.identity.speaker import identify_from_audio
            return await asyncio.to_thread(identify_from_audio, audio, SAMPLE_RATE)
        except Exception:
            return None, 0.0

    # ── Core processing pipeline ─────────────────────────────────────────────

    async def process_utterance(self, audio: np.ndarray) -> str:
        """Full pipeline: STT + speaker ID → intent → fast-path or LLM → TTS.

        Returns the final response text.
        """
        t0 = time.monotonic()

        # Parallel: STT + speaker ID
        stt_task = self._transcribe(audio)
        spk_task = self._identify_speaker(audio)
        text, (speaker_id, speaker_conf) = await asyncio.gather(stt_task, spk_task)

        t_stt = time.monotonic()
        print(f"You said: {text}")

        # Apply speaker identification
        if speaker_id and speaker_conf > 0.75:
            try:
                from mother.identity.speaker import get_registry, set_current_user
                profile = get_registry().get_user(speaker_id)
                if profile:
                    set_current_user(speaker_id, confidence=speaker_conf, method="voice")
                    print(f"[Identified: {profile.display_name} ({speaker_conf*100:.0f}%)]")
                    from mother.memory.conversation import get_memory
                    get_memory().load(speaker_id)
            except Exception:
                pass

        if not text or not text.strip():
            return ""

        return await self.process_text(text, t_start=t0, t_stt=t_stt - t0)

    async def process_text(
        self,
        text: str,
        *,
        t_start: float = 0,
        t_stt: float = 0,
        speak: bool = True,
    ) -> str:
        """Process text input: intent → fast-path or LLM → TTS.

        Returns the final response text.
        """
        from mother.identity.speaker import get_current_user
        from mother.memory.manager import (
            get_current_user_memory, maybe_learn_from_statement,
        )
        from mother.memory.conversation import get_memory as get_conv_memory
        from mother.core.context_awareness import build_context_aware_prompt

        # Background: passive learning
        cu = get_current_user()
        if cu:
            threading.Thread(
                target=lambda: list(maybe_learn_from_statement(text, llm_fn=self._llm.chat)),
                daemon=True,
            ).start()

        # Classify intent
        intent = classify_intent(text)

        # Route
        has_vision = False  # TODO: wire vision state from MQTT
        route_type, tier = self._router.route(text, intent, has_vision_context=has_vision)
        logger.info("Intent=%s route=%s tier=%s", intent.name, route_type, tier)

        # ── Fast-path ──
        if route_type == "fast_path":
            response = fast_path_dispatch(
                intent, text,
                memory_manager=get_current_user_memory(),
                current_user=cu,
            )
            if response:
                print(f"[MOTHER] {response}")
                if speak:
                    self._speak_sync(response)
                return response

        # ── LLM path ──
        # Build enriched system prompt
        sys_prompt = build_context_aware_prompt(
            self.cfg.llm.system_prompt,
            user_text=text,
        )

        # User context + memory
        if cu:
            from mother.identity.speaker import get_user_context_for_prompt
            user_ctx = get_user_context_for_prompt(cu)
            if user_ctx:
                sys_prompt += f"\n\nUser context: {user_ctx}"
            user_mem = get_current_user_memory()
            if user_mem:
                mem_ctx = user_mem.get_context_for_prompt(text, max_items=3)
                if mem_ctx:
                    sys_prompt += f"\n\nMemory context: {mem_ctx}"

        # RAG enrichment — notes + codebase self-awareness (both best-effort)
        rag_block = await self._fetch_rag(text)
        if rag_block:
            sys_prompt += f"\n\n{rag_block}"

        # Build messages with conversation history
        conv = get_conv_memory()
        conv.add_user(text)
        messages = conv.get_messages(sys_prompt)

        # Set tier
        if isinstance(self._llm, TieredLLMDriver) and tier:
            self._llm.set_tier(tier)
            print(f"[LLM] Routing to {tier} ({self._llm.current_model})")
        elif isinstance(self._llm, HybridLLMDriver):
            if intent == Intent.GENERAL:
                self._llm.route_to_cloud()
            else:
                self._llm.route_to_local()

        # Tool context
        tool_ctx = ToolContext(
            http_client=self._http,
            rag_base=getattr(self.cfg.rag, "api_base", "http://127.0.0.1:8123") if self.cfg.rag else "http://127.0.0.1:8123",
            rag_timeout=getattr(self.cfg.rag, "timeout_ms", 500) / 1000.0 if self.cfg.rag else 0.5,
            user_memory=get_current_user_memory(),
            current_user=cu,
        )

        # Stream LLM with background TTS synthesis
        t_llm0 = time.monotonic()
        response = await self._stream_llm_with_tts(
            messages, tool_ctx, speak=speak,
        )
        t_llm1 = time.monotonic()

        # Save to conversation memory
        conv.add_assistant(response)
        if cu:
            try:
                conv.save(cu.user_id)
            except Exception:
                pass

        # Latency log
        total = time.monotonic() - t_start if t_start else 0
        if t_start:
            print(f"[latency] stt={t_stt:.3f}s llm={t_llm1 - t_llm0:.3f}s total={total:.3f}s")

        return response

    async def _stream_llm_with_tts(
        self,
        messages: List[ChatMessage],
        tool_ctx: ToolContext,
        *,
        speak: bool = True,
    ) -> str:
        """Stream LLM tokens with concurrent sentence-boundary TTS synthesis.

        Returns the full response text.
        """
        # Background TTS synthesis pipeline
        synth_in: queue.Queue = queue.Queue()
        synth_out: queue.Queue = queue.Queue(maxsize=2)

        def synth_worker():
            while True:
                txt = synth_in.get()
                if txt is None:
                    synth_out.put(None)
                    return
                clean = normalize_for_speech(txt).replace('\x00', '').replace('\ufffd', '')
                data, sr = self._synthesize(clean)
                synth_out.put((data, sr) if data is not None else None)

        synth_thread = threading.Thread(target=synth_worker, daemon=True)
        synth_thread.start()

        # Stream LLM
        full_response: list[str] = []
        sentence_buffer = ""
        tool_call_intercepted = False

        def _do_stream():
            return list(self._llm.stream_chat(
                messages,
                temperature=self.cfg.llm.temperature,
                max_tokens=self.cfg.llm.max_tokens,
                tools=TOOLS_SCHEMA,
            ))

        tokens = await asyncio.to_thread(_do_stream)

        for chunk in tokens:
            # Tool call sentinel
            if chunk.startswith("__TOOL_CALL__:"):
                tool_call_intercepted = True
                try:
                    tc_data = json.loads(chunk[len("__TOOL_CALL__:"):])
                    tc_calls = tc_data.get("message", {}).get("tool_calls", [])
                    for tc in tc_calls:
                        tc_name = tc.get("function", {}).get("name", "")
                        tc_args = tc.get("function", {}).get("arguments", {})
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args)
                            except Exception:
                                tc_args = {}
                        result = dispatch_tool_call(tc_name, tc_args, tool_ctx)
                        print(f"\n[Tool: {tc_name}] {result}")

                        # Re-prompt with tool result
                        import copy
                        followup = copy.copy(messages)
                        followup.append(ChatMessage(role="tool", content=result))
                        followup.append(ChatMessage(role="user", content="Summarise this for me in one or two sentences."))

                        for fc in self._llm.stream_chat(
                            followup,
                            temperature=self.cfg.llm.temperature,
                            max_tokens=self.cfg.llm.max_tokens,
                        ):
                            print(fc, end="", flush=True)
                            full_response.append(fc)
                            sentence_buffer_inner = ""
                            sentence_buffer_inner += fc
                            # Stream sentences from tool followup
                            if self.cfg.tts.stream_sentences:
                                while True:
                                    m = SENTENCE_BOUNDARY.search(sentence_buffer_inner)
                                    if not m:
                                        break
                                    seg = sentence_buffer_inner[:m.end()].strip()
                                    sentence_buffer_inner = sentence_buffer_inner[m.end():]
                                    if seg:
                                        synth_in.put(seg)
                except Exception as e:
                    logger.warning("Tool dispatch error: %s", e)
                continue

            print(chunk, end="", flush=True)
            full_response.append(chunk)

            if self.cfg.tts.stream_sentences:
                sentence_buffer += chunk
                while True:
                    m = SENTENCE_BOUNDARY.search(sentence_buffer)
                    if not m:
                        break
                    seg = sentence_buffer[:m.end()].strip()
                    sentence_buffer = sentence_buffer[m.end():]
                    if seg:
                        synth_in.put(seg)

        print()  # newline after streaming

        # Enqueue remaining tail or full text
        if self.cfg.tts.stream_sentences and sentence_buffer.strip():
            synth_in.put(sentence_buffer.strip())
        elif not self.cfg.tts.stream_sentences:
            full_text = "".join(full_response).strip()
            if full_text:
                synth_in.put(full_text)
        synth_in.put(None)  # sentinel

        # Play synthesized audio — most sentences already ready from concurrency
        if speak:
            interrupted = False
            while not interrupted:
                item = synth_out.get()
                if item is None:
                    break
                data, sr = item
                if data is not None and self._play_audio(data, sr):
                    interrupted = True
                    self._armed_recording = True
                    # Drain queue so worker thread exits
                    while True:
                        try:
                            synth_out.get_nowait()
                        except queue.Empty:
                            break

        return "".join(full_response)

    async def _fetch_rag(self, text: str) -> str:
        """Fetch RAG context (notes + code) as a pre-formatted block.

        Delegates to `context_injection.fetch_context`, which queries both
        indexes in parallel. The code index is only consulted when the
        query looks code-related (cheap regex) so most queries pay a
        single HTTP round-trip, not two.
        """
        if not self.cfg.rag or not self.cfg.rag.enabled or not self.cfg.rag.on_demand:
            return ""
        try:
            from mother.core.context_injection import fetch_context
            return await fetch_context(
                text,
                api_base=self.cfg.rag.api_base,
                notes_k=min(self.cfg.rag.k, 2),
                code_k=getattr(self.cfg.rag, "code_k", 3),
                timeout_ms=self.cfg.rag.timeout_ms,
                enable_notes=True,
                enable_code=getattr(self.cfg.rag, "code_enabled", True),
            )
        except Exception:
            return ""

    # ── PTT mode ─────────────────────────────────────────────────────────────

    async def run_ptt(self):
        """Push-to-talk loop: Enter to start recording, Enter to stop."""
        import sounddevice as sd
        from mother.core.context_awareness import TimeContext, reset_context
        from mother.identity.speaker import get_registry, reset_session

        reset_context()
        reset_session()

        # Greeting
        registry = get_registry()
        enrolled = registry.list_users()
        if enrolled:
            greeting = TimeContext.get_greeting() + " Speak so I can identify you."
        else:
            greeting = TimeContext.get_greeting()
            print("[Note: No users enrolled. Run 'python -m mother.identity.enroll' to enroll.]")
        print(f"\n[MOTHER] {greeting}")
        self._speak_sync(greeting)

        # Wake word detector
        self._start_wake_word()

        print("\nPress Enter to start; Enter again to stop. Ctrl+C to exit.\n")

        while True:
            try:
                # Wait for Enter key or wake word
                await self._wait_for_trigger()

                # Record
                recording: list[np.ndarray] = []
                is_recording = True
                print("Listening... (press Enter to stop)")

                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    callback=lambda indata, frames, t, status: recording.append(indata.copy()),
                )
                stream.start()

                # Wait for Enter to stop
                await self._wait_for_stop()
                stream.stop()
                stream.close()

                if not recording:
                    print("No audio captured.")
                    continue

                audio = np.concatenate(recording, axis=0).squeeze()

                # Process the utterance
                await self.process_utterance(audio)

                self._drain_keys()
                print("Listening... (press Enter to stop)")

            except KeyboardInterrupt:
                print()
                break

    async def _wait_for_trigger(self):
        """Wait for Enter key press or wake word trigger."""
        while True:
            if self._armed_recording:
                self._armed_recording = False
                return
            if self._check_key_interrupt():
                return
            await asyncio.sleep(0.01)

    async def _wait_for_stop(self):
        """Wait for Enter key to stop recording (with auto-repeat debounce)."""
        # Wait for quiet period first (debounce auto-repeat)
        idle_s = 0.35
        last_event = time.monotonic()
        while True:
            if self._armed_recording:
                return
            if self._check_key_interrupt():
                last_event = time.monotonic()
            if time.monotonic() - last_event >= idle_s:
                break
            await asyncio.sleep(0.01)

        # Now wait for deliberate Enter
        while True:
            if self._armed_recording:
                return
            if self._check_key_interrupt():
                return
            await asyncio.sleep(0.01)

    def _start_wake_word(self):
        """Start the wake word detector in background."""
        try:
            import os
            from mother.audio.wake_word import WakeWordDetector
            import sounddevice as sd

            def on_wake(keyword: str, score: float):
                print(f"\n[MOTHER] Wake word detected ({keyword}, score: {score:.2f}) — listening...")
                try:
                    t = np.linspace(0, 0.2, int(SAMPLE_RATE * 0.2), dtype=np.float32)
                    freq = np.linspace(600, 1200, len(t))
                    tone = 0.3 * np.sin(2 * np.pi * freq * t / SAMPLE_RATE)
                    sd.play(tone, SAMPLE_RATE)
                except Exception:
                    pass
                self._armed_recording = True

            sensitivity = float(os.environ.get("WAKE_WORD_SENSITIVITY", "0.5"))
            self._wake_detector = WakeWordDetector(
                sensitivity=sensitivity,
                on_detected=on_wake,
            )
            self._wake_detector.start()
            print(f"[WakeWord] Listening for '{self._wake_detector.model_name}'")
        except Exception as e:
            logger.warning("Wake word init failed (%s) — Enter key only", e)

    def _drain_keys(self):
        """Drain buffered key presses (Windows)."""
        if sys.platform == "win32":
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()

    # ── Single prompt mode ───────────────────────────────────────────────────

    async def run_prompt(self, text: str):
        """Process a single text prompt and speak the response."""
        await self.process_text(text)

    # ── Server mode ──────────────────────────────────────────────────────────

    async def run_server(self, host: str = "0.0.0.0", port: int = 8300):
        """Start the FastAPI server (delegates to api.server)."""
        import uvicorn
        config = uvicorn.Config(
            "mother.api.server:app",
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
