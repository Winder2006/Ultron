"""WebSocket voice endpoint for MOTHER.

Protocol:
    Client sends binary frames: raw PCM float32 mono 16kHz audio chunks.
    Client sends JSON text frames for control: {"action": "start"}, {"action": "stop"}.
    Server sends JSON text frames: events as they happen.

Events sent to client:
    {"event": "stt",       "text": "...", "final": true/false}
    {"event": "intent",    "intent": "WEATHER", "route": "fast_path"}
    {"event": "llm_token", "token": "..."}
    {"event": "llm_done",  "full_text": "..."}
    {"event": "tts_ready", "audio_b64": "...", "sample_rate": 24000}
    {"event": "speaker",   "user_id": "...", "confidence": 0.85}
    {"event": "error",     "message": "..."}
    {"event": "tool_call", "name": "...", "result": "..."}
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import queue
import re
import threading
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mother.core.logging_config import get_logger

logger = get_logger("mother.api.voice")
router = APIRouter()

# Tool result preview length for the SSE event bus / tool_call event.
# 200 chars is long enough to see the shape of the payload on the
# dashboard without blowing up the SSE frame. Notes have their own
# separate truncation (800) where full content matters.
MAX_RESULT_PREVIEW = 200

# Counter for dropped events so we don't spam the log per drop, but
# also don't go silent. Reset is best-effort — a long-running server
# will see monotonically increasing counts, which is fine for alerting.
_event_bus_drops = {"count": 0, "last_logged": 0.0}


def _emit_event(bus: Optional[asyncio.Queue], event: dict) -> None:
    """Push an observability event to the SSE bus.

    Uses put_nowait so the hot path never blocks on a slow subscriber,
    but unlike the old `except QueueFull: pass` pattern, we count drops
    and log them periodically so the operator actually knows when
    metrics are going missing.
    """
    if bus is None:
        return
    try:
        bus.put_nowait(event)
    except asyncio.QueueFull:
        _event_bus_drops["count"] += 1
        now = time.monotonic()
        # Rate-limit the warning to once every 10s so a flooded bus
        # doesn't fill the log file too.
        if now - _event_bus_drops["last_logged"] > 10.0:
            _event_bus_drops["last_logged"] = now
            logger.warning(
                "[event_bus] dropped %d events (queue full) — SSE subscriber slow or absent",
                _event_bus_drops["count"],
            )
    except Exception as e:
        logger.debug("event bus put failed: %s", e)


async def _send_event(ws: WebSocket, event: dict):
    """Send a JSON event to the WebSocket client.

    Swallows expected disconnects but logs unexpected failures — otherwise
    an error on send is invisible and the UI silently hangs on a client
    that's still open but can't receive events.
    """
    try:
        await ws.send_json(event)
    except WebSocketDisconnect:
        # Expected when the client tab is closed mid-response.
        pass
    except RuntimeError as e:
        # FastAPI raises this when the socket has already closed on
        # the other side. Not worth logging at warning level.
        logger.debug("send_json on closed socket: %s", e)
    except Exception as e:
        logger.warning("send_json failed (event=%s): %s", event.get("event"), e)


def _run_speaker_id(audio: np.ndarray, sr: int) -> tuple[Optional[str], float]:
    """Run speaker identification (blocking — call from thread)."""
    try:
        from mother.identity.speaker import identify_from_audio
        return identify_from_audio(audio, sr)
    except Exception:
        return None, 0.0


def _run_transcription(streaming_stt, stt_legacy, audio: np.ndarray, sr: int) -> str:
    """Transcribe audio via StreamingSTT or legacy fallback (blocking)."""
    if streaming_stt is not None:
        try:
            loop = asyncio.new_event_loop()
            text = loop.run_until_complete(streaming_stt.transcribe_audio(audio, sr))
            loop.close()
            return text
        except Exception:
            pass
    if stt_legacy is not None:
        try:
            return stt_legacy.transcribe_pcm(audio, sr)
        except Exception:
            pass
    return ""


def _synthesize_to_b64(tts, text: str) -> tuple[Optional[str], int]:
    """Synthesize text to base64-encoded WAV. Returns (b64_data, sample_rate)."""
    try:
        from mother.tts.normalizer import normalize_for_speech
        text = normalize_for_speech(text)
    except Exception:
        pass
    try:
        wav_bytes = tts.synthesize_to_bytes(text)
        if wav_bytes and len(wav_bytes) > 100:
            return base64.b64encode(wav_bytes).decode("ascii"), 24000
    except Exception:
        pass
    return None, 0


async def _dispatch_execute_python(
    tool_args: dict,
    *,
    ws: WebSocket,
    event_bus,
    turn_state: dict,
    current_user,
) -> str:
    """Run execute_python and emit the structured `code_exec` SSE event.

    This is THE single path for running user-visible Python — the first
    tool call, ReAct hops, and the fake-tag rescue all come through
    here, so the /exec dashboard view never stays dark while code runs.
    The subprocess round-trip (up to 60s) happens in a worker thread:
    running it inline on the event loop froze TTS streaming, SSE, and
    every other session for the duration of the exec.

    Returns the compact LLM-facing result string.
    """
    from mother.tools.code_exec import (
        execute_python_full, restore_session_state,
    )
    # First exec of this WS session — restore any pickled namespace
    # from the previous session so variables survive reconnects.
    if current_user is not None and not turn_state.get("repl_state_loaded"):
        turn_state["repl_state_loaded"] = True
        try:
            res = await asyncio.to_thread(
                restore_session_state, id(ws), current_user.user_id,
            )
            if res.get("loaded", 0) > 0:
                logger.info(
                    "[code_exec] restored %d vars for user=%s",
                    res["loaded"], current_user.user_id,
                )
        except Exception as restore_err:
            logger.debug("repl restore error: %s", restore_err)

    full = await asyncio.to_thread(
        execute_python_full, tool_args, session_key=id(ws),
    )

    # Rich event for the /exec dashboard view — full code + streams.
    _emit_event(event_bus, {
        "type": "code_exec",
        "code": full.get("code", ""),
        "stdout": full.get("stdout", ""),
        "stderr": full.get("stderr", ""),
        "value": full.get("result", ""),
        "duration_s": full.get("duration_s", 0),
        "timed_out": full.get("timed_out", False),
        "images": full.get("images", []),
        "session": id(ws),
    })

    # Compact LLM-facing string built from the same payload.
    parts: list[str] = []
    if full.get("result"):
        parts.append(f"=> {full['result']}")
    if full.get("stdout"):
        out = full["stdout"].rstrip()
        if out:
            parts.append(f"stdout:\n{out}")
    if full.get("stderr"):
        err = full["stderr"].rstrip()
        if err:
            parts.append(f"stderr:\n{err}")
    if full.get("timed_out"):
        parts.append("(timed out — REPL restarted)")
    if not parts:
        parts.append("(no output)")
    parts.append(f"[{full.get('duration_s', 0):.2f}s]")
    result = "\n".join(parts)
    if len(result) > 3000:
        result = result[:3000] + "\n... (truncated; see /exec view)"
    return result


async def _stream_fast_path_pcm(ws: WebSocket, tts, text: str) -> None:
    """Stream a fast-path reply as raw PCM chunks, same wire format the
    LLM path uses for sentences. One sentence_id per fast-path reply
    since there's no token-boundary fragmentation to worry about.

    Emits:
      {event: tts_start, sentence_id, text, sample_rate}
      {event: tts_chunk, sentence_id, pcm_b64}*
      {event: tts_end,   sentence_id}
    """
    if not text or not hasattr(tts, "synthesize_stream_pcm"):
        return
    sample_rate = 24000
    await _send_event(ws, {
        "event": "tts_start",
        "sentence_id": 1,
        "text": text,
        "sample_rate": sample_rate,
    })
    # Bridge the blocking PCM generator to asyncio via a queue so the
    # main task isn't blocked during synth. Same pattern as the LLM
    # path's _stream_one_sentence uses.
    chunk_q: asyncio.Queue = asyncio.Queue()

    def _produce():
        try:
            for chunk in tts.synthesize_stream_pcm(text):
                chunk_q._loop.call_soon_threadsafe(chunk_q.put_nowait, chunk)
        except Exception as e:
            logger.warning("fast-path TTS stream error: %s", e)
        finally:
            chunk_q._loop.call_soon_threadsafe(chunk_q.put_nowait, None)

    chunk_q._loop = asyncio.get_running_loop()
    import threading as _th
    _th.Thread(target=_produce, daemon=True).start()

    while True:
        chunk = await chunk_q.get()
        if chunk is None:
            break
        if not chunk:
            continue
        await _send_event(ws, {
            "event": "tts_chunk",
            "sentence_id": 1,
            "pcm_b64": base64.b64encode(chunk).decode("ascii"),
        })
    await _send_event(ws, {"event": "tts_end", "sentence_id": 1})


@router.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    """Main voice interaction WebSocket.

    Flow:
    1. Client sends {"action": "start"} to begin recording
    2. Client streams binary PCM chunks
    3. Client sends {"action": "stop"} to end recording
    4. Server transcribes, classifies intent, routes to handler/LLM, streams response
    5. Server sends TTS audio back as base64
    """
    await ws.accept()
    from mother.api.server import get_state
    state = get_state()
    llm = state.get("llm")
    tts = state.get("tts")
    stt = state.get("stt")
    streaming_stt = state.get("streaming_stt")
    cfg = state.get("config")
    event_bus: asyncio.Queue = state.get("event_bus")
    ws_clients: set = state.get("ws_clients")
    ws_clients.add(ws)

    # Per-connection Event: set the moment a real user turn starts
    # (recording, text prompt, or any processing). Ambient tasks check
    # this before emitting audio so a slow greeting can't talk over
    # the user mid-sentence. Event is clear()-able by the idle watcher
    # after a grace period, but we also stamp the last-activity time
    # so a slow idle iteration can't re-enable ambient speech right as
    # the user is mid-utterance.
    ambient_suppressed = asyncio.Event()
    ambient_state = {"last_activity": 0.0}
    ws.state.ambient_suppressed = ambient_suppressed  # type: ignore[attr-defined]

    # ── Barge-in support ──
    # `cancel_event` is set when the user interrupts (wake word fires
    # or push-to-talk pressed mid-utterance). The LLM producer thread
    # checks it between tokens and bails; the TTS worker checks it in
    # its sentence loop and drops pending segments. The current-turn
    # task is also tracked so we can cancel it explicitly. The flag is
    # cleared at the start of each new turn.
    cancel_event = asyncio.Event()
    turn_state: dict = {"current_turn_task": None, "repl_state_loaded": False}
    ws.state.cancel_event = cancel_event  # type: ignore[attr-defined]
    ws.state.turn_state = turn_state  # type: ignore[attr-defined]

    # Background tasks spawned during this connection — we MUST cancel
    # all of them on disconnect or they leak into the next session
    # (reused Deepgram sockets, ghost TTS, cross-user speaker ID).
    bg_tasks: set[asyncio.Task] = set()

    def _spawn(coro) -> asyncio.Task:
        """Track a background task so we can cancel it on disconnect."""
        task = asyncio.create_task(coro)
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
        return task

    # ── Ambient speech: morning greeting if due for this user ──
    # Speak it through the same TTS path the conversational replies use,
    # so voice, filter, and framing stay consistent. Best-effort — if
    # anything here errors we silently skip. Running this before any
    # user turn also seeds `current_user` for the rest of the session.
    try:
        from mother.identity.speaker import get_or_fallback_user
        from mother.core.ambient import maybe_morning_greeting, _is_morning_window
        _greet_user = get_or_fallback_user()
        # Fold today's weather into the greeting when we're in-window.
        # Guardrails:
        #   - Only fetch weather if we're actually in the morning window —
        #     otherwise the ambient function will return None and the
        #     fetch is wasted work.
        #   - Bound the fetch with asyncio.wait_for so the greeting (and
        #     therefore the first user-facing TTS event) can't be held
        #     up by a slow weather endpoint.
        _greet_weather: Optional[str] = None
        _greet_calendar: Optional[str] = None
        if _is_morning_window():
            # Fetch weather + calendar in parallel — calendar adds the
            # day's commitments to the greeting ("three things today,
            # first is the standup at ten"). Both are bounded so a slow
            # backend can't delay the greeting.
            try:
                from mother.tools.weather_tool import get_weather as _gw
                from mother.tools.icloud_calendar import (
                    fetch_today as _fetch_today,
                    _fmt_clock,
                )

                async def _w_fetch():
                    return await asyncio.to_thread(
                        _gw, 43.0389, -87.9065,
                        fahrenheit=True, mph=True,
                    )

                async def _c_fetch():
                    return await asyncio.to_thread(_fetch_today)

                results = await asyncio.gather(
                    asyncio.wait_for(_w_fetch(), timeout=1.5),
                    asyncio.wait_for(_c_fetch(), timeout=2.5),
                    return_exceptions=True,
                )
                _w, _c = results
                if isinstance(_w, dict) and "temperature" in _w:
                    _t = round(_w.get("temperature") or 0)
                    _desc = _w.get("description") or ""
                    _greet_weather = (
                        f"It's {_t}°{' and ' + _desc if _desc else ''} outside."
                    )
                if isinstance(_c, list):
                    if not _c:
                        _greet_calendar = "Nothing on the calendar."
                    else:
                        n = len(_c)
                        first = _c[0]
                        first_when = (
                            "all day" if first.all_day
                            else _fmt_clock(first.start.astimezone())
                        )
                        if n == 1:
                            _greet_calendar = (
                                f"One thing today: {first.summary} {first_when}."
                            )
                        else:
                            _greet_calendar = (
                                f"{n} things today; first is "
                                f"{first.summary} at {first_when}."
                            )
            except Exception:
                pass
        _greet_line = maybe_morning_greeting(
            _greet_user.user_id, _greet_user.display_name,
            weather_summary=_greet_weather,
            calendar_summary=_greet_calendar,
        )
        if _greet_line and tts is not None:
            async def _speak_greeting(line: str):
                # Use the fallback-TTS path (WAV b64) to keep this simple
                # — it's one short utterance, not worth the streaming
                # machinery. Runs in a thread because the TTS call is
                # blocking. If the user starts interacting before the
                # synthesis completes, we drop the emission — better
                # to feel a missed greeting than to be talked over.
                try:
                    b64, sr = await asyncio.to_thread(_synthesize_to_b64, tts, line)
                    if ambient_suppressed.is_set():
                        return
                    if b64:
                        await _send_event(ws, {"event": "ambient", "text": line})
                        await _send_event(ws, {
                            "event": "tts_ready",
                            "audio_b64": b64,
                            "sample_rate": sr,
                        })
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("ambient greeting TTS failed: %s", e)
            _spawn(_speak_greeting(_greet_line))
    except Exception as e:
        logger.debug("ambient greeting skipped: %s", e)

    # ── Ambient: idle-observation watcher ──
    # Ticks every 60s, checks the scheduler. The scheduler's own gates
    # (activity timestamp + backoff + awake-hours) decide whether to
    # fire; we just wake it up periodically.
    # Minimum quiet period before the idle watcher is allowed to
    # un-suppress ambient speech. Keeps a slow idle tick from flipping
    # the flag off *during* a user utterance just because record_activity
    # happened >60s ago but the conversation is still in flight.
    AMBIENT_QUIET_S = 30.0

    async def _idle_watcher():
        from mother.core.ambient import maybe_idle_observation
        from mother.identity.speaker import get_or_fallback_user
        try:
            while True:
                await asyncio.sleep(60)
                try:
                    # Only un-suppress if the user has been quiet for
                    # the full quiet window. This is the fix for the
                    # race where the idle tick would reset the flag
                    # while the user was still speaking.
                    last = ambient_state["last_activity"]
                    if last and (time.monotonic() - last) < AMBIENT_QUIET_S:
                        continue
                    ambient_suppressed.clear()
                    user = get_or_fallback_user()
                    line = maybe_idle_observation(user.user_id)
                    if line and tts is not None:
                        b64, sr = await asyncio.to_thread(_synthesize_to_b64, tts, line)
                        if ambient_suppressed.is_set():
                            # User started talking during synthesis —
                            # drop this idle emission.
                            continue
                        if b64:
                            await _send_event(ws, {"event": "ambient", "text": line})
                            await _send_event(ws, {
                                "event": "tts_ready",
                                "audio_b64": b64,
                                "sample_rate": sr,
                            })
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug("idle watcher iteration failed: %s", e)
        except asyncio.CancelledError:
            pass

    _spawn(_idle_watcher())

    # Live-streaming recording state:
    #   audio_queue    — PCM chunks being streamed to Deepgram in real-time
    #   audio_buffer   — same chunks kept as a backup for speaker ID + fallback STT
    #   stt_task       — background task draining Deepgram transcripts
    #   transcript_queue — final transcripts from Deepgram
    #   stt_track      — interim-tracking state for speculative dispatch.
    #                    `latest_interim` holds Deepgram's most recent partial,
    #                    `stable_count` counts consecutive frames where the
    #                    text didn't change. When the user pauses, Deepgram
    #                    keeps emitting the same interim until it commits a
    #                    final — that gives us a high-confidence signal we
    #                    can dispatch on, ~300-500ms before the final lands.
    audio_queue: Optional[asyncio.Queue] = None
    audio_buffer: list[np.ndarray] = []
    stt_task: Optional[asyncio.Task] = None
    transcript_queue: Optional[asyncio.Queue] = None
    stt_track: dict = {
        "latest_interim": "",
        "latest_interim_ts": 0.0,
        "stable_count": 0,
    }
    is_recording = False
    sample_rate = 16000

    async def _drain_transcripts(aq: asyncio.Queue, tq: asyncio.Queue):
        """Background task: pull streaming transcripts from Deepgram.

        Pushes final transcripts into `tq` (consumed by the recording-
        stop handler). Also emits interim + final `stt` events directly
        on the WebSocket so the browser sees the live transcript AND
        can auto-stop recording the moment Deepgram flags final —
        without waiting for the server-side stop-handler to propagate.

        Tracks interim stability (consecutive frames with the same text)
        so the stop handler can speculatively dispatch on a stable
        interim instead of waiting for Deepgram's final commit.
        """
        if streaming_stt is None:
            return
        try:
            async for text, is_final in streaming_stt.stream(aq):
                if not text:
                    continue
                # Forward every transcript (interim + final) to the
                # browser so the auto-stop-on-final logic fires
                # promptly. This is the event that cuts end-of-turn
                # latency from ~1.5s down to ~150ms.
                await _send_event(ws, {
                    "event": "stt",
                    "text": text,
                    "final": is_final,
                })
                if is_final:
                    await tq.put(text)
                else:
                    # Track interim stability. When Deepgram emits the
                    # same partial twice in a row, the user has clearly
                    # paused — the next thing it sends is almost
                    # always a final with this same text.
                    if text == stt_track["latest_interim"]:
                        stt_track["stable_count"] += 1
                    else:
                        stt_track["stable_count"] = 0
                    stt_track["latest_interim"] = text
                    stt_track["latest_interim_ts"] = time.monotonic()
        except Exception as e:
            logger.warning("Live STT stream error: %s", e)
        finally:
            await tq.put("")

    chunks_received = 0
    try:
        while True:
            message = await ws.receive()

            # Binary frame: audio chunk — stream to Deepgram immediately
            if "bytes" in message:
                if is_recording:
                    raw_bytes = message["bytes"]
                    # Expected format: float32 mono PCM from the browser's
                    # AudioWorklet. Validate length is aligned; if the client
                    # sends a different dtype we'd silently corrupt audio.
                    if len(raw_bytes) % 4 != 0:
                        logger.warning(
                            "[voice] received audio chunk not aligned to float32 "
                            "(%d bytes) — dropping", len(raw_bytes),
                        )
                        continue
                    chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                    audio_buffer.append(chunk)
                    chunks_received += 1
                    # Only log the FIRST chunk — per-10 logging produces
                    # noise without adding information (all subsequent
                    # chunks go through the same path).
                    if chunks_received == 1:
                        logger.info(
                            "[TIMING] rx_chunk first (queued_to_stt=%s)",
                            audio_queue is not None,
                        )
                    if audio_queue is not None:
                        await audio_queue.put(chunk)
                continue

            # Text frame: control message
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _send_event(ws, {"event": "error", "message": "Invalid JSON"})
                    continue

                action = data.get("action", "")

                if action == "start":
                    # If a previous turn is still in flight, this is a
                    # barge-in. Cancel everything before starting fresh.
                    cur = turn_state.get("current_turn_task")
                    if cur is not None and not cur.done():
                        cancel_event.set()
                        cur.cancel()
                        await _send_event(ws, {"event": "cancelled"})
                        _emit_event(event_bus, {"type": "barge_in"})
                    # Clear cancel_event so the new turn isn't pre-killed.
                    cancel_event.clear()

                    # Suppress any pending ambient emissions — the user
                    # is about to speak, we should not talk over them.
                    ambient_suppressed.set()
                    ambient_state["last_activity"] = time.monotonic()
                    audio_buffer.clear()
                    # Open fresh Deepgram streaming session
                    audio_queue = asyncio.Queue()
                    transcript_queue = asyncio.Queue()
                    if streaming_stt is not None and streaming_stt._deepgram_available:
                        stt_task = _spawn(
                            _drain_transcripts(audio_queue, transcript_queue)
                        )
                    # ── TTS pre-warm ──
                    # Fire a tiny background synth the moment recording
                    # starts. This keeps the Deepgram TTS HTTP/2
                    # connection pool hot, so the first real synth
                    # (called once the LLM's first clause is ready)
                    # doesn't pay the TLS/connection-setup tax. Costs
                    # ~1 character of Deepgram billing per turn (~$0
                    # rounded). Daemon thread, fire-and-forget — never
                    # blocks the recording flow even on TTS errors.
                    if tts is not None and hasattr(tts, "warmup"):
                        def _bg_tts_warmup(_t=tts):
                            try:
                                _t.warmup()
                            except Exception:
                                pass
                        threading.Thread(
                            target=_bg_tts_warmup, daemon=True,
                            name="tts-prewarm",
                        ).start()
                    is_recording = True
                    await _send_event(ws, {"event": "recording_started"})

                elif action == "stop":
                    is_recording = False
                    await _send_event(ws, {"event": "recording_stopped"})

                    if not audio_buffer:
                        await _send_event(ws, {"event": "error", "message": "No audio captured"})
                        continue

                    # Signal end-of-audio to Deepgram
                    if audio_queue is not None:
                        await audio_queue.put(None)

                    audio = np.concatenate(audio_buffer, axis=0)
                    audio_buffer.clear()

                    # ── Speculative dispatch on stable interim ──
                    # The browser's VAD already auto-stopped the mic
                    # after ~700ms of silence, so by the time `stop`
                    # arrives the user has been quiet for a while and
                    # Deepgram's last interim is usually identical to
                    # the final it's about to commit. We capture the
                    # latest interim, wait BRIEFLY for the final, and
                    # fall back to the interim if the final doesn't
                    # land in time. Saves 300-500ms per turn in the
                    # common case.
                    live_text = ""
                    transcript_source = "none"
                    # Capture references — we null these out below but
                    # the deferred final-capture task needs them.
                    _spec_stt_task = stt_task
                    _spec_tq = transcript_queue

                    if _spec_stt_task is not None and _spec_tq is not None:
                        t_wait = time.monotonic()
                        interim = stt_track.get("latest_interim") or ""
                        stable = stt_track.get("stable_count", 0)
                        last_interim_ts = stt_track.get("latest_interim_ts", 0.0)
                        # An interim is "complete enough" to commit on
                        # ONLY when ALL of these hold:
                        #   • ≥3 words — filters out partial fragments
                        #     ("what is the", "how many") that arrive
                        #     before the user has finished speaking.
                        #   • Deepgram emitted the same text on
                        #     consecutive frames (stable_count >= 1)
                        #     — confirms user has paused, not just
                        #     mid-utterance breath.
                        #   • The interim is old (>200ms since last
                        #     update) — extra confirmation Deepgram
                        #     hasn't seen new speech.
                        # Earlier we used (stable >= 1 OR age > 0.25),
                        # which fired on partial interims that happened
                        # to be quiet for 250ms ("What is eight" while
                        # user was about to say "times 12"). That cost
                        # us a confabulated answer. Now both checks
                        # required — slightly slower fallback, much
                        # higher correctness.
                        now = time.monotonic()
                        interim_age = now - last_interim_ts if last_interim_ts else 0.0
                        is_complete_interim = (
                            interim
                            and len(interim.split()) >= 3
                            and stable >= 1
                            and interim_age > 0.20
                        )
                        # WS path with speech_final delivers finals in
                        # ~100-300ms reliably. 700ms gives slow finals
                        # a fair shot before falling back to REST.
                        wait_timeout = 0.7

                        try:
                            live_text = await asyncio.wait_for(
                                _spec_tq.get(), timeout=wait_timeout,
                            )
                            transcript_source = "final"
                            logger.info(
                                "[TIMING] live_stt: waited=%.3fs (final) text=%r",
                                time.monotonic() - t_wait, live_text[:60],
                            )
                        except asyncio.TimeoutError:
                            if is_complete_interim:
                                live_text = interim
                                transcript_source = "interim"
                                logger.info(
                                    "[TIMING] live_stt: using interim after %.3fs "
                                    "(stable=%d age=%.2fs words=%d) text=%r",
                                    time.monotonic() - t_wait, stable,
                                    interim_age, len(interim.split()),
                                    interim[:60],
                                )
                                _emit_event(event_bus, {
                                    "type": "stt_speculative",
                                    "text": live_text[:80],
                                })
                            else:
                                logger.warning(
                                    "Live STT timed out (interim=%r stable=%d "
                                    "words=%d), falling back to batch",
                                    interim[:40] if interim else "",
                                    stable,
                                    len(interim.split()) if interim else 0,
                                )

                    # Reset interim-tracking state for next utterance.
                    stt_track["latest_interim"] = ""
                    stt_track["latest_interim_ts"] = 0.0
                    stt_track["stable_count"] = 0

                    stt_task = None
                    audio_queue = None
                    transcript_queue = None

                    # Cleanup. If we used the interim, defer the STT
                    # task cleanup so the actual final still has time
                    # to arrive — we want to capture it for divergence
                    # telemetry without blocking the LLM dispatch.
                    if transcript_source == "interim" and _spec_stt_task is not None:
                        async def _capture_final_then_cleanup(
                            tq=_spec_tq,
                            task=_spec_stt_task,
                            spec_text=live_text,
                        ):
                            try:
                                final_text = await asyncio.wait_for(
                                    tq.get(), timeout=2.0,
                                )
                                # Compare normalized — Deepgram
                                # sometimes adds trailing punctuation
                                # to the final that wasn't in the
                                # interim, which we don't want to flag
                                # as divergence.
                                norm_a = (final_text or "").strip().rstrip(".!?,").lower()
                                norm_b = (spec_text or "").strip().rstrip(".!?,").lower()
                                if final_text and norm_a != norm_b:
                                    logger.warning(
                                        "[STT] interim diverged from final: "
                                        "interim=%r final=%r",
                                        spec_text[:60], final_text[:60],
                                    )
                                    _emit_event(event_bus, {
                                        "type": "stt_interim_diverged",
                                        "interim": spec_text[:80],
                                        "final": final_text[:80],
                                    })
                            except asyncio.TimeoutError:
                                pass
                            except Exception as e:
                                logger.debug("divergence-check error: %s", e)
                            finally:
                                if not task.done():
                                    task.cancel()
                                    try:
                                        await task
                                    except (asyncio.CancelledError, Exception):
                                        pass
                        _spawn(_capture_final_then_cleanup())
                    else:
                        # Cleanup the streaming STT task — cancel AND await
                        # so the Deepgram socket is fully torn down before
                        # the next utterance opens a new one. Without this
                        # the next start can reuse a dying socket and see
                        # stale audio from the previous utterance.
                        if _spec_stt_task is not None and not _spec_stt_task.done():
                            _spec_stt_task.cancel()
                            try:
                                await _spec_stt_task
                            except (asyncio.CancelledError, Exception):
                                pass

                    # Process utterance (pass live_text to skip batch
                    # transcription). Wrap as a task so a barge-in
                    # cancel can interrupt it cleanly.
                    async def _run_utterance(a=audio, lt=live_text):
                        try:
                            await _process_utterance(
                                ws, a, sample_rate,
                                llm=llm, tts=tts, stt=stt,
                                streaming_stt=streaming_stt,
                                cfg=cfg, event_bus=event_bus,
                                vision_state=state.get("vision"),
                                pre_transcribed_text=lt,
                                spawn=_spawn,
                            )
                        finally:
                            turn_state["current_turn_task"] = None
                    utt_task = asyncio.create_task(_run_utterance())
                    turn_state["current_turn_task"] = utt_task
                    bg_tasks.add(utt_task)
                    utt_task.add_done_callback(bg_tasks.discard)

                elif action == "transcribe":
                    # One-shot: client sends complete audio in data["audio_b64"]
                    audio_b64 = data.get("audio_b64", "")
                    if audio_b64:
                        raw = base64.b64decode(audio_b64)
                        audio = np.frombuffer(raw, dtype=np.float32)
                        text = await asyncio.to_thread(
                            _run_transcription, streaming_stt, stt, audio, sample_rate
                        )
                        await _send_event(ws, {"event": "stt", "text": text, "final": True})

                elif action == "prompt":
                    # Text-only: skip STT, go straight to intent + LLM.
                    # Suppress ambient first — same rationale as for
                    # voice: don't let a pending greeting talk over
                    # the user's reply.
                    # Barge-in handling: same as start — kill any
                    # in-flight turn before spawning a new one.
                    cur = turn_state.get("current_turn_task")
                    if cur is not None and not cur.done():
                        cancel_event.set()
                        cur.cancel()
                        await _send_event(ws, {"event": "cancelled"})
                        _emit_event(event_bus, {"type": "barge_in"})
                    cancel_event.clear()

                    ambient_suppressed.set()
                    ambient_state["last_activity"] = time.monotonic()
                    text = data.get("text", "")
                    if text:
                        async def _run_prompt(t=text):
                            try:
                                await _process_text_query(
                                    ws, t,
                                    llm=llm, tts=tts, cfg=cfg,
                                    event_bus=event_bus,
                                    vision_state=state.get("vision"),
                                )
                            finally:
                                turn_state["current_turn_task"] = None
                        prompt_task = asyncio.create_task(_run_prompt())
                        turn_state["current_turn_task"] = prompt_task
                        bg_tasks.add(prompt_task)
                        prompt_task.add_done_callback(bg_tasks.discard)

                elif action == "cancel":
                    # Barge-in: user pressed mic / wake word fired
                    # mid-response. Stop everything in flight so the
                    # new utterance can land cleanly.
                    #   1. Set cancel_event so the LLM producer thread
                    #      and the TTS worker bail out promptly.
                    #   2. Cancel the current turn's task if one is
                    #      tracked.
                    #   3. Tell the client to flush its playback
                    #      buffer so already-buffered audio doesn't
                    #      keep playing after cancellation.
                    if not cancel_event.is_set():
                        cancel_event.set()
                        cur = turn_state.get("current_turn_task")
                        if cur is not None and not cur.done():
                            cur.cancel()
                        await _send_event(ws, {"event": "cancelled"})
                        _emit_event(event_bus, {"type": "barge_in"})

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        ws_clients.discard(ws)
        # Cancel every background task spawned during this connection.
        # Without this, the idle watcher, greeting task, speaker ID,
        # and STT drain all leak — they keep the Deepgram socket
        # open and can emit audio into the next user's session.
        for task in list(bg_tasks):
            if not task.done():
                task.cancel()
        # Give cancellation a bounded window to settle. If a task ignores
        # cancel (shouldn't happen with the above handlers) we log and
        # move on — blocking forever on a stuck task is worse than a leak.
        if bg_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*bg_tasks, return_exceptions=True),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[voice] %d background tasks did not cancel within 2s",
                    sum(1 for t in bg_tasks if not t.done()),
                )
            except Exception as e:
                logger.debug("bg_tasks cleanup error: %s", e)
        # Persist the user's REPL namespace BEFORE tearing the
        # subprocess down — so variables survive the next reconnect /
        # backend restart.
        try:
            from mother.tools.code_exec import (
                shutdown_repl, persist_session_state,
            )
            try:
                from mother.identity.speaker import get_or_fallback_user
                _u = get_or_fallback_user()
                if _u is not None:
                    # Threaded — pickling the namespace + the REPL wait
                    # can take seconds and this loop serves other
                    # sessions too.
                    res = await asyncio.to_thread(
                        persist_session_state, id(ws), _u.user_id,
                    )
                    if res.get("saved", 0) > 0:
                        logger.info(
                            "[code_exec] persisted %d vars for user=%s "
                            "(skipped %d)",
                            res["saved"], _u.user_id,
                            len(res.get("skipped") or []),
                        )
            except Exception as pe:
                logger.debug("repl persist error: %s", pe)
            await asyncio.to_thread(shutdown_repl, id(ws))
        except Exception as e:
            logger.debug("repl shutdown error: %s", e)


# ── Processing pipeline ──────────────────────────────────────────────────────

async def _process_utterance(
    ws: WebSocket,
    audio: np.ndarray,
    sr: int,
    *,
    llm, tts, stt, streaming_stt,
    cfg, event_bus, vision_state,
    pre_transcribed_text: str = "",
    spawn=None,
):
    """Full pipeline: STT + speaker ID (parallel) → intent → handler/LLM → TTS.

    If pre_transcribed_text is provided (from live Deepgram streaming),
    skip the batch transcription step entirely — saves 500-1500ms.
    """
    t_utterance_start = time.monotonic()

    # Skip speaker ID on the hot path. Speaker ID is slow (~2s Resemblyzer on CPU)
    # and *never* needs to block the response. Run it in a fire-and-forget task
    # that updates the session state for the NEXT utterance.
    from mother.identity.speaker import get_session, get_registry, set_current_user
    sess = get_session()
    has_enrolled_users = bool(get_registry().list_users())
    speaker_id, speaker_conf = None, 0.0

    if has_enrolled_users and not sess.is_identified():
        # Background speaker ID — doesn't block anything.
        # Tracked via the parent connection's spawn() so it gets
        # cancelled on disconnect (otherwise the ~2s Resemblyzer work
        # keeps running after the user is gone and can post an event
        # to a closed socket).
        async def _bg_speaker_id():
            try:
                sid, conf = await asyncio.to_thread(_run_speaker_id, audio, sr)
                if sid and conf > 0.75:
                    profile = get_registry().get_user(sid)
                    if profile:
                        set_current_user(sid, confidence=conf, method="voice")
                        await _send_event(ws, {
                            "event": "speaker",
                            "user_id": sid,
                            "display_name": profile.display_name,
                            "confidence": conf,
                        })
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        if spawn is not None:
            spawn(_bg_speaker_id())
        else:
            asyncio.create_task(_bg_speaker_id())

    if pre_transcribed_text:
        text = pre_transcribed_text
    else:
        text = await asyncio.to_thread(_run_transcription, streaming_stt, stt, audio, sr)

    logger.info(
        "[TIMING] utterance_transcribed: %.3fs (pre_transcribed=%s, text=%r)",
        time.monotonic() - t_utterance_start,
        bool(pre_transcribed_text),
        (text or "")[:60],
    )

    # ── Prosody analysis: read voice cues from the audio buffer ──
    # Cheap (~3-8ms numpy) so we just block on it. The resulting
    # tag is appended to the system prompt so the LLM can match
    # register: clipped if the user sounds urgent, calmer if the
    # user sounds calm. Empty string if signal is too quiet/short.
    prosody_tag = ""
    try:
        from mother.audio.prosody import analyze_to_tag
        from mother.identity.speaker import get_or_fallback_user as _gofu
        _puser = _gofu()
        _t_pros = time.monotonic()
        prosody_tag = await asyncio.to_thread(
            analyze_to_tag, audio, sr,
            _puser.user_id if _puser else None,
        )
        if prosody_tag:
            logger.info(
                "[TIMING] prosody analyze: %.3fs tag=%r",
                time.monotonic() - _t_pros, prosody_tag,
            )
            _emit_event(event_bus, {"type": "prosody", "tag": prosody_tag})
    except Exception as e:
        logger.debug("prosody analysis skipped: %s", e)

    # Emit STT result (speaker ID runs in background via _bg_speaker_id)
    await _send_event(ws, {"event": "stt", "text": text, "final": True})

    if not text or not text.strip():
        # No speech detected — tell the frontend we're done so it unblocks
        await _send_event(ws, {
            "event": "llm_done",
            "full_text": "",
            "reason": "empty_transcript",
        })
        return

    # Passive memory learning is handled inside _process_text_query so
    # both the voice and the text-prompt paths benefit from it.
    await _process_text_query(
        ws, text, llm=llm, tts=tts, cfg=cfg,
        event_bus=event_bus, vision_state=vision_state,
        prosody_tag=prosody_tag,
    )


async def _process_text_query(
    ws: WebSocket,
    text: str,
    *,
    llm, tts, cfg, event_bus, vision_state,
    prosody_tag: str = "",
):
    """Intent classification → fast-path or LLM → TTS → send back."""
    from mother.core.intent import classify as classify_intent
    from mother.core.router import RequestRouter
    from mother.llm.drivers import ChatMessage, TieredLLMDriver

    # ── Ensure we have a user identity for memory ──
    # Use get_or_fallback_user() so memory works even without voice
    # enrollment (the dashboard's typical starting state). If voice ID
    # already set a real user, that takes precedence and is returned here.
    from mother.identity.speaker import get_or_fallback_user
    current_user = get_or_fallback_user()

    # Per-connection mutable state (barge-in, REPL restore tracking).
    # Stored on ws.state by voice_websocket; fall back to a plain dict
    # if called from a context that never set it (tests / fast-path).
    turn_state: dict = getattr(ws.state, "turn_state", {"repl_state_loaded": False})

    # Record activity for the ambient scheduler — this suppresses idle
    # observations for ~15 minutes after each real user turn.
    try:
        from mother.core.ambient import record_activity
        record_activity(current_user.user_id)
    except Exception:
        pass

    # ── Passive memory learning (fire-and-forget background thread) ──
    # Runs regex + optional LLM extraction against the user's statement.
    # Writes happen inside the current user's memory dir. We start a real
    # daemon thread rather than asyncio.to_thread — the previous code
    # *returned* an awaitable and discarded it, so learning never fired.
    try:
        from mother.memory.manager import maybe_learn_from_statement
        _loop_for_bus = asyncio.get_running_loop()

        def _learn_and_emit():
            learned = list(maybe_learn_from_statement(text, llm_fn=llm.chat))
            if learned and event_bus is not None:
                def _push():
                    _emit_event(event_bus, {
                        "type": "memory_write",
                        "count": len(learned),
                        "items": learned[:8],
                    })
                try:
                    _loop_for_bus.call_soon_threadsafe(_push)
                except Exception:
                    pass

        threading.Thread(target=_learn_and_emit, daemon=True).start()
    except Exception as e:
        logger.debug("passive-learn skipped: %s", e)

    intent = classify_intent(text)
    router = RequestRouter()

    has_vision = bool(vision_state and vision_state.get("faces"))
    route_type, tier = router.route(text, intent, has_vision_context=has_vision)

    await _send_event(ws, {
        "event": "intent",
        "intent": intent.name,
        "route": route_type,
        "tier": tier,
    })

    # Broadcast to SSE event bus
    _emit_event(event_bus, {
        "type": "query",
        "text": text,
        "intent": intent.name,
        "route": route_type,
        "tier": tier,
    })

    # ── Fast-path handlers ──
    if route_type == "fast_path":
        response = await _handle_fast_path(intent, text)
        await _send_event(ws, {"event": "llm_done", "full_text": response})
        # Stream the fast-path reply through the same PCM pipeline the
        # LLM branch uses. Previously we synthesized to a full WAV and
        # sent one big tts_ready blob, which added 500-1000ms of
        # generate-then-download latency. Streaming PCM hits the
        # browser's AudioWorklet ring buffer as chunks arrive,
        # dropping first-audible-word to ~150ms.
        if response and hasattr(tts, "synthesize_stream_pcm"):
            await _stream_fast_path_pcm(ws, tts, response)
        elif response:
            # Engines without streaming (Kokoro/Piper fallback) keep
            # the legacy WAV-blob path.
            b64, sr = await asyncio.to_thread(_synthesize_to_b64, tts, response)
            if b64:
                await _send_event(ws, {"event": "tts_ready", "audio_b64": b64, "sample_rate": sr})
        return

    # ── LLM path ──
    # Build messages with context. Critical: the STATIC persona prompt
    # is held separately from per-turn dynamic content so Anthropic's
    # prompt cache can hit. The cache requires the cached message to
    # be byte-identical across requests; mixing in things that change
    # per turn (prosody, RAG, memory, vision) busts the cache and
    # costs ~300-500ms TTFT on every turn.
    #
    # The fix: two system messages. First is the static persona
    # (cacheable, ~7KB + tools schema). Second is dynamic context
    # (per-turn, NOT cached, smaller). The driver layer applies
    # cache_control to the first block only.
    system_prompt = cfg.llm.system_prompt if cfg else "You are ULTRON."
    dynamic_parts: list[str] = []

    # Inject vision context if available
    if has_vision:
        try:
            from mother.vision.context import build_vision_context
            vision_text = build_vision_context(vision_state)
            if vision_text:
                dynamic_parts.append(f"[Visual context: {vision_text}]")
        except Exception:
            pass

    # Prosody cues — appended only on voice turns. The model is told
    # that this is observation, not user-stated content, so it
    # adjusts register without quoting the tags back.
    if prosody_tag:
        dynamic_parts.append(
            f"[Voice cues: {prosody_tag}. "
            "Match the speaker's register without commenting on it. "
            "Don't say 'you sound urgent' — just be sharper if they are.]"
        )

    # RAG enrichment — notes + codebase self-awareness. Best-effort,
    # bounded by rag.timeout_ms, silently empty if the RAG service is
    # down or disabled in config.
    #
    # Skip for trivial queries — greetings and short confirmations
    # never benefit from RAG and the HTTP round-trip is 200-400ms we'd
    # rather not spend. The earlier version of this also rejected any
    # query with ≤3 words, which turned out to be too aggressive: "who
    # is shakespeare" / "what is LV-426" / "explain your architecture"
    # are all exactly 3 words and SHOULD hit RAG. So we only skip on an
    # exact-match whitelist of pure social turns. A stripped trailing
    # punctuation lets "hello!" and "thanks." match too.
    _text_low = (text or "").strip().lower().rstrip("!.?,")
    _is_trivial = _text_low in {
        "yes", "no", "okay", "ok", "sure", "thanks", "thank you",
        "hi", "hello", "hey", "good morning", "good evening",
        "good night", "goodnight", "bye", "goodbye",
        "yeah", "yep", "nope", "cool", "got it", "right",
        "mhm", "uh huh", "ah", "oh", "huh",
    }
    # RAG, personal memory, and long-conversation memory are three
    # INDEPENDENT retrievals that each cost 50-500ms. They used to run
    # serially — worse, the memory/FAISS lookups ran synchronously ON
    # the event loop, stalling TTS streaming and every other session
    # while they computed embeddings. Now: each runs as its own task
    # (sync work pushed to threads) and we gather them, so the total
    # pre-LLM context cost is the max of the three, not the sum.
    async def _rag_fetch_task() -> Optional[str]:
        if not (cfg and cfg.rag and cfg.rag.enabled and cfg.rag.on_demand
                and not _is_trivial):
            return None
        try:
            from mother.core.context_injection import fetch_context
            _rag_t0 = time.monotonic()
            rag_block = await fetch_context(
                text,
                api_base=cfg.rag.api_base,
                notes_k=min(cfg.rag.k, 2),
                code_k=getattr(cfg.rag, "code_k", 3),
                timeout_ms=cfg.rag.timeout_ms,
                enable_notes=True,
                enable_code=getattr(cfg.rag, "code_enabled", True),
            )
            if rag_block:
                # Emit observability event: summarise the retrieval so
                # the dashboard can display what context Ultron was given.
                _emit_event(event_bus, {
                    "type": "rag_hit",
                    "query": text[:80],
                    "duration_ms": int((time.monotonic() - _rag_t0) * 1000),
                    "preview": rag_block[:300],
                    "blocks": rag_block.count("\n- "),
                })
            return rag_block or None
        except Exception as e:
            logger.warning("RAG fetch failed: %s", e)
            return None

    async def _memory_ctx_task() -> Optional[str]:
        # Skip on trivial queries — greetings and acks don't benefit
        # from "user facts" context, and every retrieval costs a DB
        # hit + an embedding lookup (~50-100ms).
        if current_user is None or _is_trivial:
            return None
        try:
            from mother.memory.manager import get_user_memory
            user_mem = get_user_memory(current_user.user_id)
            if user_mem is None:
                return None
            mem_ctx = await asyncio.to_thread(
                user_mem.get_context_for_prompt, text, max_items=5,
            )
            return f"Memory context: {mem_ctx}" if mem_ctx else None
        except Exception as e:
            logger.debug("memory-context injection skipped: %s", e)
            return None

    async def _long_memory_task() -> Optional[str]:
        # Long-conversation semantic memory — search every past
        # exchange embedded in the per-user FAISS index. This is
        # what makes "what did we decide about X last week" work
        # without dragging the full transcript into the prompt.
        if current_user is None or _is_trivial:
            return None
        try:
            from mother.memory.long_memory import (
                get_long_memory, format_for_prompt as _fmt_long,
            )
            _long_mem = get_long_memory(current_user.user_id)
            _t_long = time.monotonic()
            long_hits = await asyncio.to_thread(_long_mem.search, text, k=3)
            long_block = _fmt_long(long_hits)
            if long_block:
                _emit_event(event_bus, {
                    "type": "long_memory_hit",
                    "count": len(long_hits),
                    "top_score": round(long_hits[0][0], 3) if long_hits else 0,
                    "duration_ms": int((time.monotonic() - _t_long) * 1000),
                })
            return long_block or None
        except Exception as e:
            logger.debug("long_memory retrieval skipped: %s", e)
            return None

    _rag_block, _mem_block, _long_block = await asyncio.gather(
        _rag_fetch_task(), _memory_ctx_task(), _long_memory_task(),
    )
    if _rag_block:
        dynamic_parts.append(_rag_block)

    # User identity + personal memory. Mirrors the orchestrator's
    # (process_text) context injection so the WS path has feature parity.
    if current_user is not None:
        try:
            from mother.identity.speaker import get_user_context_for_prompt
            user_ctx = get_user_context_for_prompt(current_user)
            if user_ctx:
                dynamic_parts.append(f"User context: {user_ctx}")
        except Exception as e:
            logger.debug("user-context injection skipped: %s", e)
    if _mem_block:
        dynamic_parts.append(_mem_block)
    if _long_block:
        dynamic_parts.append(_long_block)

    # Conversation memory — load from disk on first use of this process
    # so multi-turn history survives server restarts. `_conv_loaded_for`
    # tracks which user_id we've already loaded to avoid re-reading the
    # file on every turn.
    #
    # NOTE: rolling summary goes into dynamic_parts (not the cached
    # static prompt) because it changes when conversations grow.
    history_msgs: list[ChatMessage] = []
    try:
        from mother.memory.conversation import get_memory
        conv = get_memory()
        if current_user is not None and (
            getattr(conv, "_loaded_for", None) != current_user.user_id
        ):
            try:
                conv.load(current_user.user_id)
                conv._loaded_for = current_user.user_id  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("conv.load skipped: %s", e)

        # For pure greetings ("hello", "hi", "yes"), don't feed prior
        # conversation history into the LLM. The history is still
        # persisted on disk; we just hide it from this turn's prompt.
        if not _is_trivial:
            if conv.summary:
                dynamic_parts.append(
                    f"Earlier conversation (summary): {conv.summary}"
                )
            history_msgs = list(conv._history)
    except Exception:
        history_msgs = []

    # Assemble messages. First the cached static system prompt, then
    # the dynamic context as a SECOND system message (driver layer
    # merges these into one multi-block system payload, applying
    # cache_control to the static block only). Then conversation
    # history. Then the user's turn.
    messages = [ChatMessage(role="system", content=system_prompt)]
    if dynamic_parts:
        messages.append(ChatMessage(
            role="system",
            content="\n\n".join(dynamic_parts),
        ))
    messages.extend(history_msgs)
    messages.append(ChatMessage(role="user", content=text))

    # Tier is passed directly into stream_chat below (atomic with the
    # call). The legacy set_tier() path is kept as a default fallback
    # but is racy under concurrent WS sessions, so we don't use it.
    _tier_kwargs = {"tier": tier} if (isinstance(llm, TieredLLMDriver) and tier) else {}

    # Stream LLM response — real-time token-by-token via thread-safe queue
    full_response: list[str] = []
    # TTS chunking strategy: for the FIRST chunk of a response, split
    # on any clause boundary (comma, em-dash, semicolon, period). That
    # kicks off speech as soon as Ultron finishes his first phrase,
    # which typically trims 300-700ms of perceived latency on longer
    # responses. For subsequent chunks we fall back to sentence-only
    # boundaries so the middle of a thought isn't carved into choppy
    # fragments that break prosody.
    first_clause_boundary = re.compile(r"[,;:.!?—](?:\s|$)|\n")
    sentence_boundary = re.compile(r"[.!?](?:\s|$)|\n")
    sentence_buffer = ""
    _first_chunk_emitted = False
    tool_call_detected = False
    # Fake-tool-tag rescue state. When the LLM writes
    # <execute_python>...</execute_python> as TEXT instead of using the
    # real tool_calls API, we capture the body and dispatch it ourselves.
    # `_emit_hold` is a small lookahead buffer — only chars that can't
    # possibly extend into `<execute_python` get committed; the rest
    # waits for the next token. Without it, a tag split across tokens
    # (`<`, `execute`, `_python>`) leaks its leading chars into the UI.
    _fake_tool_active = False
    _fake_tool_buffer = ""
    _fake_tool_format = ""  # "execute_python" or "function_calls" — which opener matched
    _emit_hold = ""

    # Two opener formats Haiku has been observed to emit when it fails
    # to use the real tool_calls API:
    #   1. Simple <execute_python>code</execute_python>  (early form)
    #   2. Anthropic legacy <function_calls><invoke name="X">
    #      <parameter name="Y">val</parameter></invoke></function_calls>
    # We detect either, capture until the matching closer, then parse
    # and dispatch as a real tool call.
    _RESCUE_OPENERS = (
        ("<function_calls", "</function_calls>", "function_calls"),
        ("<execute_python", "</execute_python>", "execute_python"),
    )

    # Timing instrumentation
    t_request_start = time.monotonic()
    t_first_token = [0.0]
    t_first_tts_ready = [0.0]

    # Token queue filled by a background thread, drained by the async loop.
    #
    # Using asyncio.Queue here (not threading.Queue + run_in_executor)
    # is critical: when the LLM stream stalls and the watchdog cancels
    # the wait, asyncio.Queue.get() unwinds cleanly. The earlier
    # threading-Queue approach blocked an executor thread on `get()`
    # that couldn't be cancelled — every stall leaked one thread, and
    # after several stalls the executor pool filled up, blocking ALL
    # to_thread() calls (STT, prosody, speaker ID, downstream LLM
    # warmups). One bad stream took the whole backend down silently.
    _drain_loop = asyncio.get_running_loop()
    token_queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def _put_token(tok):
        """Schedule put on the asyncio queue from the producer thread.
        call_soon_threadsafe is the only safe way to touch asyncio
        primitives from a non-loop thread."""
        try:
            _drain_loop.call_soon_threadsafe(token_queue.put_nowait, tok)
        except RuntimeError:
            # Loop closed (server shutting down) — drop silently.
            pass

    # Tools: only pass to tier 2/3 (Haiku/Sonnet) — tier 1 (Cerebras
    # Llama 8B) isn't reliable at structured tool calling and will
    # produce malformed __TOOL_CALL__ sentinels. If we're about to hit
    # tier 1, we skip tools entirely; the LLM will just answer directly.
    _tools_for_call = None
    if isinstance(llm, TieredLLMDriver) and tier in ("tier2", "tier3"):
        try:
            from mother.llm.tools import TOOLS_SCHEMA as _TS
            _tools_for_call = _TS
        except Exception:
            _tools_for_call = None

    # Cancel-event lookup — set by the barge-in handler. We reach
    # through ws.state since this function is module-level.
    _cancel_event: Optional[asyncio.Event] = getattr(ws.state, "cancel_event", None)

    def _producer():
        """Run blocking LLM stream in a thread, push tokens into queue.

        If the first stream returns zero tokens (observed with tier3 Sonnet
        + tools + large conversation history — the model returns
        finish_reason=end_turn with no content and no tool call), retry
        once without tools. Better to get a direct text answer than silence.

        Bails out promptly if cancel_event fires (barge-in).
        """
        tok_count = 0
        try:
            for tok in llm.stream_chat(
                messages,
                temperature=cfg.llm.temperature if cfg else 0.7,
                max_tokens=cfg.llm.max_tokens if cfg else None,
                tools=_tools_for_call,
                **_tier_kwargs,
            ):
                if _cancel_event is not None and _cancel_event.is_set():
                    break
                _put_token(tok)
                tok_count += 1
        except Exception as e:
            logger.warning("LLM stream error: %s", e, exc_info=True)

        if tok_count == 0 and (_cancel_event is None or not _cancel_event.is_set()):
            logger.warning(
                "[LLM] empty stream — retrying without tools (tier=%s msgs=%d)",
                tier, len(messages),
            )
            try:
                for tok in llm.stream_chat(
                    messages,
                    temperature=cfg.llm.temperature if cfg else 0.7,
                    max_tokens=cfg.llm.max_tokens if cfg else None,
                    **_tier_kwargs,
                ):
                    if _cancel_event is not None and _cancel_event.is_set():
                        break
                    _put_token(tok)
                    tok_count += 1
            except Exception as e:
                logger.warning("LLM retry stream error: %s", e)

        _put_token(SENTINEL)

    t_llm_thread_start = time.monotonic()

    threading.Thread(target=_producer, daemon=True).start()

    # Streaming TTS: synthesize each sentence via Deepgram and stream raw PCM
    # chunks to the browser as they arrive (~150ms time-to-first-chunk vs 1200ms
    # for the full WAV). Browser plays chunks with Web Audio API scheduling.
    tts_pending: asyncio.Queue = asyncio.Queue()

    # Check if TTS supports PCM streaming (Deepgram does; Kokoro/Piper don't).
    has_pcm_stream = hasattr(tts, "synthesize_stream_pcm")

    def _start_sentence_synth(seg: str) -> asyncio.Queue:
        """Kick off TTS synthesis for a sentence immediately, buffering
        PCM chunks into a queue. Separated from _stream_one_sentence so
        the worker can start sentence N+1's synthesis while sentence N
        is still streaming out — Deepgram's ~150-300ms time-to-first-
        byte then overlaps playback instead of inserting an audible gap
        at every sentence boundary."""
        q: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                for chunk in tts.synthesize_stream_pcm(seg):
                    _drain_loop.call_soon_threadsafe(q.put_nowait, chunk)
            except Exception as e:
                logger.warning("TTS stream error: %s", e)
            finally:
                _drain_loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=_produce, daemon=True).start()
        return q

    async def _stream_one_sentence(
        seg: str, sentence_id: int, chunk_q: Optional[asyncio.Queue] = None,
    ):
        """Stream a single sentence's PCM to the WebSocket.

        Streams chunks through in real-time (low latency), but buffers the
        TAIL so we can trim trailing silence and apply a 5ms fade-out to the
        last emitted chunk. Leading silence gets a fade-in on the first chunk.
        This produces click-free boundaries between sentences.

        `chunk_q` may be a queue from _start_sentence_synth whose
        synthesis is already in flight (the prefetch path); when None,
        synthesis starts here.
        """
        import numpy as np
        t0 = time.monotonic()
        first_chunk_t = [0.0]

        await _send_event(ws, {
            "event": "tts_start",
            "sentence_id": sentence_id,
            "text": seg,
            "sample_rate": 24000,
        })

        if chunk_q is None:
            chunk_q = _start_sentence_synth(seg)

        sample_rate = 24000
        fade_len = int(sample_rate * 0.005)  # 5ms fade
        silence_threshold = 150               # out of 32767

        # We always hold back the LAST chunk received so we can trim/fade it
        # before sending. This preserves streaming latency for all but the
        # final chunk of each sentence (which is the one that matters).
        held_chunk: bytes | None = None
        is_first_chunk = True

        async def _emit(chunk_bytes: bytes, apply_fade_in: bool, apply_fade_out: bool):
            """Send a chunk, optionally fading its edges."""
            if not chunk_bytes:
                return
            if not (apply_fade_in or apply_fade_out):
                await _send_event(ws, {
                    "event": "tts_chunk",
                    "sentence_id": sentence_id,
                    "pcm_b64": base64.b64encode(chunk_bytes).decode("ascii"),
                })
                return
            samples = np.frombuffer(chunk_bytes, dtype=np.int16).copy()
            f_len = min(fade_len, len(samples) // 2)
            if f_len > 0:
                samples_f = samples.astype(np.float32)
                if apply_fade_in:
                    ramp = np.linspace(0.0, 1.0, f_len, dtype=np.float32)
                    samples_f[:f_len] *= ramp
                if apply_fade_out:
                    ramp = np.linspace(1.0, 0.0, f_len, dtype=np.float32)
                    samples_f[-f_len:] *= ramp
                samples = samples_f.astype(np.int16)
            await _send_event(ws, {
                "event": "tts_chunk",
                "sentence_id": sentence_id,
                "pcm_b64": base64.b64encode(samples.tobytes()).decode("ascii"),
            })

        while True:
            chunk = await chunk_q.get()
            if chunk is None:
                break
            # Mid-sentence barge-in: stop pumping PCM to the client.
            # We still need to drain the producer's queue (so the
            # underlying thread can finish and clean up), so we just
            # discard the remaining chunks.
            if _cancel_event is not None and _cancel_event.is_set():
                continue
            if first_chunk_t[0] == 0.0:
                first_chunk_t[0] = time.monotonic()
                if t_first_tts_ready[0] == 0.0:
                    t_first_tts_ready[0] = first_chunk_t[0]
                    logger.info(
                        "[TIMING] first_tts_chunk: %.3fs after req_start (sentence=%r)",
                        t_first_tts_ready[0] - t_request_start,
                        seg[:40],
                    )

            # Emit the previously held chunk (we now know it's not the last)
            if held_chunk is not None:
                await _emit(held_chunk, apply_fade_in=is_first_chunk, apply_fade_out=False)
                is_first_chunk = False
            held_chunk = chunk

        # Process the final held chunk: trim trailing silence, apply fade-out
        if held_chunk is not None:
            samples = np.frombuffer(held_chunk, dtype=np.int16)
            # Trim trailing silence
            abs_s = np.abs(samples)
            if (abs_s > silence_threshold).any():
                last_sound = len(samples) - int(np.argmax(abs_s[::-1] > silence_threshold))
                # Keep 20ms padding so we don't clip the last phoneme
                pad = int(sample_rate * 0.02)
                last_sound = min(len(samples), last_sound + pad)
                samples = samples[:last_sound]
            if len(samples) > 0:
                await _emit(
                    samples.tobytes(),
                    apply_fade_in=is_first_chunk,
                    apply_fade_out=True,
                )

        synth_dur = time.monotonic() - t0
        logger.info(
            "[TIMING] tts_sentence_done synth=%.3fs first_chunk=%.3fs (text=%r)",
            synth_dur,
            first_chunk_t[0] - t0 if first_chunk_t[0] else 0,
            seg[:40],
        )
        await _send_event(ws, {"event": "tts_end", "sentence_id": sentence_id})

    async def _fallback_tts(seg: str):
        """For engines without PCM streaming — fall back to base64 WAV."""
        t0 = time.monotonic()
        b64, sr = await asyncio.to_thread(_synthesize_to_b64, tts, seg)
        synth_dur = time.monotonic() - t0
        if b64:
            if t_first_tts_ready[0] == 0.0:
                t_first_tts_ready[0] = time.monotonic()
                logger.info(
                    "[TIMING] first_tts_ready: synth=%.3fs total=%.3fs (text=%r)",
                    synth_dur,
                    t_first_tts_ready[0] - t_request_start,
                    seg[:40],
                )
            await _send_event(ws, {"event": "tts_ready", "audio_b64": b64, "sample_rate": sr})

    sentence_counter = [0]

    async def _tts_worker():
        # One-sentence synthesis lookahead: while sentence N streams to
        # the client, sentence N+1's Deepgram request is already in
        # flight, so its TTFB is paid during playback, not as a gap.
        prefetched: Optional[tuple[str, asyncio.Queue]] = None
        saw_end = False
        while True:
            if prefetched is not None:
                seg, chunk_q = prefetched
                prefetched = None
            elif saw_end:
                break
            else:
                seg = await tts_pending.get()
                if seg is None:
                    break
                chunk_q = None
            # Drop any pending segments if barge-in fired — we don't
            # want to keep speaking after the user cut us off.
            if _cancel_event is not None and _cancel_event.is_set():
                continue
            sentence_counter[0] += 1
            if has_pcm_stream:
                if chunk_q is None:
                    chunk_q = _start_sentence_synth(seg)
                # If the next segment is already queued, start its
                # synthesis NOW, before we spend this sentence's
                # playback time streaming chunks.
                if not saw_end:
                    try:
                        nxt = tts_pending.get_nowait()
                        if nxt is None:
                            saw_end = True
                        else:
                            prefetched = (nxt, _start_sentence_synth(nxt))
                    except asyncio.QueueEmpty:
                        pass
                await _stream_one_sentence(seg, sentence_counter[0], chunk_q)
            else:
                await _fallback_tts(seg)

    tts_worker_task = asyncio.create_task(_tts_worker())

    def _queue_tts(seg: str):
        tts_pending.put_nowait(seg)

    loop = asyncio.get_running_loop()

    # ── TTFT watchdog ──
    # If the LLM stream produces no tokens within this window, the
    # upstream call is wedged (Anthropic stall, network drop, LiteLLM
    # connection-pool deadlock — all observed). On timeout we set the
    # cancel flag (the producer thread will bail next time it checks),
    # emit a user-facing error event, and break out cleanly.
    #
    # NOTE: token_queue is an asyncio.Queue (not threading.Queue) so
    # asyncio.wait_for can cancel cleanly without leaving any thread
    # blocked. The earlier threading.Queue + run_in_executor design
    # leaked an executor thread per stall, eventually exhausting the
    # default pool and freezing every other to_thread() in the app.
    TTFT_WATCHDOG_S = 20.0

    async def _get_token_or_timeout(first_token_yet: bool):
        timeout = TTFT_WATCHDOG_S if not first_token_yet else 30.0
        return await asyncio.wait_for(token_queue.get(), timeout=timeout)

    # Drain the token queue
    while True:
        # Watchdog: bail if no token arrives in time.
        try:
            token = await _get_token_or_timeout(t_first_token[0] > 0.0)
        except asyncio.TimeoutError:
            stall_kind = "first-token" if t_first_token[0] == 0.0 else "mid-stream"
            logger.warning(
                "[LLM] %s stall — aborting turn (tier=%s msgs=%d)",
                stall_kind, tier, len(messages),
            )
            if _cancel_event is not None:
                _cancel_event.set()
            await _send_event(ws, {
                "event": "error",
                "message": f"LLM stalled ({stall_kind}); turn aborted.",
            })
            _emit_event(event_bus, {
                "type": "llm_stall",
                "kind": stall_kind,
                "tier": tier,
            })
            break
        if token is SENTINEL:
            break
        # Cancel mid-stream — stop forwarding tokens to the client and
        # to TTS. The producer thread will also notice and exit.
        if _cancel_event is not None and _cancel_event.is_set():
            continue

        # Handle tool call sentinel
        if isinstance(token, str) and token.startswith("__TOOL_CALL__:"):
            tool_call_detected = True
            # Pre-bind so the except handler below can't NameError when
            # the sentinel JSON is malformed (tool_name/tool_args would
            # otherwise be unbound, and the resulting NameError killed
            # the whole turn: no llm_done, TTS worker never drained).
            tool_name = ""
            tool_args: dict = {}
            try:
                raw_payload = token[len("__TOOL_CALL__:"):]
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError as je:
                    logger.warning(
                        "[tool_call] malformed sentinel JSON: %s (payload=%r)",
                        je, raw_payload[:200],
                    )
                    raise
                tc = payload["message"]["tool_calls"][0]["function"]
                tool_name = tc["name"]
                tool_args = tc.get("arguments", {})
                if not isinstance(tool_args, dict):
                    logger.warning(
                        "[tool_call] %s got non-dict args (%r) — coercing to {}",
                        tool_name, tool_args,
                    )
                    tool_args = {}

                from mother.llm.tools import ToolContext, dispatch_tool_call
                from mother.memory.manager import get_user_memory
                from mother.core.reminders import add_reminder
                from mother.api.server import get_state as _get_state
                user_mem = (
                    get_user_memory(current_user.user_id)
                    if current_user is not None else None
                )
                # Reuse the process-wide httpx.Client so we don't open
                # a fresh TCP/TLS connection on every tool dispatch.
                _shared_http = _get_state().get("tool_http_client")
                # The REPL session key is the WS object id — one
                # persistent Python process per WS connection so
                # variables stick across turns within a session.
                ctx = ToolContext(
                    http_client=_shared_http,
                    rag_base=(cfg.rag.api_base if cfg and cfg.rag else "http://127.0.0.1:8123"),
                    rag_timeout=(cfg.rag.timeout_ms / 1000.0 if cfg and cfg.rag else 0.5),
                    user_memory=user_mem,
                    current_user=current_user,
                    add_reminder_fn=add_reminder,
                    repl_session_key=id(ws),
                )

                # execute_python takes a side-channel through the
                # shared dispatcher so the /exec view gets the
                # structured SSE event WITHOUT re-running the code.
                if tool_name == "execute_python":
                    result = await _dispatch_execute_python(
                        tool_args, ws=ws, event_bus=event_bus,
                        turn_state=turn_state, current_user=current_user,
                    )
                else:
                    # Off the event loop — tool handlers do blocking
                    # network I/O (Brave search, IMAP email, iCloud
                    # CalDAV). Running them inline froze TTS streaming
                    # and every other session for the full call.
                    result = await asyncio.to_thread(
                        dispatch_tool_call, tool_name, tool_args, ctx,
                    )

                # Emit a dashboard event for observability. D2 will
                # consume this via SSE; for now it just goes to the
                # event bus.
                _emit_event(event_bus, {
                    "type": "tool_call",
                    "name": tool_name,
                    "args": tool_args,
                    "result": result[:MAX_RESULT_PREVIEW],
                })

                await _send_event(ws, {
                    "event": "tool_call",
                    "name": tool_name,
                    "result": result,
                })

                # Two classes of tool output:
                #   TERSE: output is already a clean, speakable sentence.
                #     (calculate / current_time / convert_units / weather /
                #      get_time_in / set_reminder / list_my_tools /
                #      forget_fact / correct_fact). TTS it directly —
                #      fast, no extra LLM round-trip.
                #   BLOB: output is a data blob with search hits, note
                #     contents, or formatted news — the kind of thing
                #     that reads like machine output if spoken verbatim
                #     ("1. Dean's Office Directory // Business // ..."
                #     followed by phone numbers and emails). For those
                #     we re-prompt the LLM to synthesize one short
                #     in-character answer, using the tool result as
                #     context.
                TERSE_TOOLS = {
                    "calculate", "current_time", "get_time_in",
                    "convert_units", "get_weather", "set_reminder",
                    "list_my_tools", "forget_fact", "correct_fact",
                    "get_memory",
                }
                # COMPOUND queries always go through the ReAct
                # re-prompt loop, even for tools that would normally
                # TTS directly. Otherwise multi-part questions ("who's
                # CEO and what's the weather there") get only the
                # weather spoken back, because the TERSE shortcut never
                # asks the LLM "anything else?". Single-intent GENERAL
                # queries ("what time is it in Tokyo") keep the direct
                # shortcut — forcing the chain on ALL of GENERAL added
                # a ~400-800ms LLM round-trip to every terse tool call
                # for no information gain.
                from mother.core.intent import _is_compound_query
                _force_chain = (
                    intent.name == "GENERAL"
                    and _is_compound_query(text.lower())
                )
                is_terse = (tool_name in TERSE_TOOLS) and not _force_chain

                if is_terse:
                    await _send_event(ws, {"event": "llm_token", "token": result})
                    _queue_tts(result)
                    full_response.append(result)
                else:
                    # ── ReAct multi-hop tool chaining ──
                    # The LLM gets the tool result and may either answer
                    # OR call another tool. Up to MAX_REACT_HOPS hops
                    # before we force a synthesis. Mid-hop preamble
                    # ("let me check that") is buffered and dropped if
                    # followed by another tool call — only the final
                    # answer streams through TTS. This unlocks "search →
                    # read → search again → synthesize" chains without
                    # shipping awkward intermediate speech.
                    from mother.llm.drivers import ChatMessage as _CM
                    MAX_REACT_HOPS = 3
                    narration_msgs = list(messages)
                    narration_msgs.append(_CM(
                        role="tool",
                        content=f"{tool_name} result:\n{result}",
                    ))

                    final_text = ""
                    hop = 0
                    while True:
                        is_last_hop = hop >= MAX_REACT_HOPS
                        if is_last_hop:
                            instr = (
                                "Answer my question now using everything "
                                "you've gathered. No more tool calls. "
                                "One or two short sentences. Your voice."
                            )
                            tools_this_hop = None
                        else:
                            instr = (
                                "Use that data to answer my question. "
                                "Call another tool only if you genuinely "
                                "need more info — don't pad. When you "
                                "answer: one or two short sentences, "
                                "your voice, not raw results."
                            )
                            tools_this_hop = _tools_for_call
                        narration_msgs.append(_CM(role="user", content=instr))

                        # asyncio.Queue + call_soon_threadsafe — same
                        # rationale as the main token_queue: avoid
                        # leaking executor threads on stalls.
                        narr_q: asyncio.Queue = asyncio.Queue()
                        _hop_kw = dict(_tier_kwargs)

                        def _narr_put(t):
                            try:
                                _drain_loop.call_soon_threadsafe(
                                    narr_q.put_nowait, t,
                                )
                            except RuntimeError:
                                pass

                        def _narrate_producer(
                            msgs=list(narration_msgs),
                            tools_arg=tools_this_hop,
                            kw=_hop_kw,
                        ):
                            try:
                                for tok in llm.stream_chat(
                                    msgs,
                                    temperature=cfg.llm.temperature if cfg else 0.7,
                                    max_tokens=(cfg.llm.max_tokens if cfg else None) or 120,
                                    tools=tools_arg,
                                    **kw,
                                ):
                                    _narr_put(tok)
                            except Exception as e:
                                logger.warning("narration hop %d stream error: %s", hop, e)
                            finally:
                                _narr_put(SENTINEL)

                        threading.Thread(target=_narrate_producer, daemon=True).start()

                        hop_text = ""
                        next_tool_sentinel = None
                        # Bound the per-hop wait so a stalled narration
                        # call can't hang the whole route. 25s is enough
                        # for tier-3 with cold cache; less than the
                        # main TTFT watchdog so it always wins.
                        NARRATION_TIMEOUT_S = 25.0
                        while True:
                            try:
                                ntok = await asyncio.wait_for(
                                    narr_q.get(), timeout=NARRATION_TIMEOUT_S,
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    "[narration] hop %d stalled — aborting", hop,
                                )
                                break
                            if ntok is SENTINEL:
                                break
                            if not isinstance(ntok, str):
                                continue
                            if ntok.startswith("__TOOL_CALL__:"):
                                if not is_last_hop:
                                    next_tool_sentinel = ntok
                                    # Drop any preamble text from this
                                    # hop — it was "let me check" filler
                                    # before the model decided to chain.
                                    break
                                # On last hop, ignore additional tool
                                # calls and keep collecting text only.
                                continue
                            hop_text += ntok

                        # Pop the instruction we appended, so the next
                        # hop's instruction replaces it cleanly rather
                        # than stacking.
                        if narration_msgs and narration_msgs[-1].role == "user":
                            narration_msgs.pop()

                        if next_tool_sentinel is None:
                            # No more tool calls — this hop's text is
                            # our final answer.
                            final_text = hop_text
                            break

                        # ── Dispatch the next hop's tool ──
                        try:
                            next_payload = json.loads(
                                next_tool_sentinel[len("__TOOL_CALL__:"):]
                            )
                            next_tc = next_payload["message"]["tool_calls"][0]["function"]
                            next_name = next_tc["name"]
                            next_args = next_tc.get("arguments", {}) or {}
                            if not isinstance(next_args, dict):
                                next_args = {}
                            # execute_python on a chained hop MUST go
                            # through the shared dispatcher — routing
                            # it through the generic registry ran the
                            # code but never emitted the code_exec SSE
                            # event, so the /exec window stayed dark.
                            if next_name == "execute_python":
                                next_result = await _dispatch_execute_python(
                                    next_args, ws=ws, event_bus=event_bus,
                                    turn_state=turn_state,
                                    current_user=current_user,
                                )
                            else:
                                next_result = await asyncio.to_thread(
                                    dispatch_tool_call, next_name, next_args, ctx,
                                )

                            await _send_event(ws, {
                                "event": "tool_call",
                                "name": next_name,
                                "result": next_result,
                            })
                            _emit_event(event_bus, {
                                "type": "tool_call",
                                "name": next_name,
                                "args": next_args,
                                "result": next_result[:MAX_RESULT_PREVIEW],
                                "hop": hop + 1,
                            })

                            narration_msgs.append(_CM(
                                role="tool",
                                content=f"{next_name} result:\n{next_result}",
                            ))
                            hop += 1
                            continue
                        except Exception as he:
                            logger.warning(
                                "ReAct hop %d dispatch failed: %s — committing partial",
                                hop + 1, he,
                            )
                            # Bail out of the chain; whatever we have
                            # in hop_text is what the user gets.
                            final_text = hop_text or "I couldn't follow that up."
                            break

                    # Stream the final synthesis through TTS with the
                    # same sentence-boundary chunking as the main path.
                    if final_text:
                        full_response.append(final_text)
                        await _send_event(ws, {"event": "llm_token", "token": final_text})
                        narration_buf = final_text
                        while True:
                            pat = (
                                sentence_boundary
                                if _first_chunk_emitted
                                else first_clause_boundary
                            )
                            m = pat.search(narration_buf)
                            if not m:
                                break
                            seg_end = m.end()
                            seg_candidate = narration_buf[:seg_end].strip()
                            if not _first_chunk_emitted and len(seg_candidate) < 18:
                                break
                            sentence_buffer_tail = narration_buf[seg_end:]
                            if seg_candidate:
                                _queue_tts(seg_candidate)
                                _first_chunk_emitted = True
                            narration_buf = sentence_buffer_tail
                        tail = narration_buf.strip()
                        if tail:
                            _queue_tts(tail)
            except Exception as e:
                logger.warning("Tool call dispatch error: %s", e)
                # execute_python errors go to /exec panel — never TTS them.
                # Routing code output through speech defeats the whole
                # purpose of the exec view and causes the "speaks the code"
                # symptom.
                if tool_name == "execute_python":
                    _emit_event(event_bus, {
                        "type": "code_exec",
                        "code": tool_args.get("code", ""),
                        "stdout": "",
                        "stderr": f"Dispatch error: {e}",
                        "value": "",
                        "duration_s": 0,
                        "timed_out": False,
                        "images": [],
                        "session": id(ws),
                    })
                    err_spoken = "Execution error. Check the exec panel."
                    full_response.append(err_spoken)
                    await _send_event(ws, {"event": "llm_token", "token": err_spoken})
                    _queue_tts(err_spoken)
                else:
                    # For non-code tools, re-prompt LLM in character without tools.
                    try:
                        from mother.llm.drivers import ChatMessage as _CM
                        fallback_msgs = list(messages)
                        fallback_msgs.append(_CM(
                            role="user",
                            content=(
                                "Answer in character without using tools. "
                                "Brief — one or two short sentences."
                            ),
                        ))
                        fb_buf = ""
                        for tok in llm.stream_chat(
                            fallback_msgs,
                            temperature=cfg.llm.temperature if cfg else 0.7,
                            max_tokens=(cfg.llm.max_tokens if cfg else None) or 120,
                            **_tier_kwargs,
                        ):
                            if isinstance(tok, str) and not tok.startswith("__TOOL_CALL__:"):
                                full_response.append(tok)
                                await _send_event(ws, {"event": "llm_token", "token": tok})
                                fb_buf += tok
                        if fb_buf.strip():
                            _queue_tts(fb_buf.strip())
                    except Exception as fe:
                        logger.warning("Tool-fallback re-prompt failed: %s", fe)
                        err_msg = "I couldn't reach that."
                        full_response.append(err_msg)
                        await _send_event(ws, {"event": "llm_token", "token": err_msg})
                        _queue_tts(err_msg)
            continue

        # Suppress any trailing text tokens after a tool call. Claude
        # sometimes continues to stream a narration like "I lack live
        # web access" alongside the tool_call — we don't want that read
        # aloud because (a) the tool already produced the real answer,
        # and (b) the narration often hallucinates ("I don't have
        # access to..." even though tools ARE configured).
        if tool_call_detected:
            continue

        # ── Fake-tool-tag rescue ──
        # Haiku 4.5 sometimes emits a tool call as TEXT —
        #     <execute_python> ...code... </execute_python>
        # — instead of using the structured tool_calls API. The harness
        # has no way to dispatch text. Result: no tool runs, the user
        # hears the tags spoken, /exec stays dark. We catch the pattern
        # in the streaming buffer, suppress it from TTS, dispatch the
        # captured code as if it were a real tool call, and continue
        # with whatever post-tag text the model emits as the response.
        #
        # Two-stage detection so a partial tag prefix split across
        # tokens (`<`, `execute`, `_python>`) doesn't leak its leading
        # chars into the dashboard before the full opener arrives:
        #   1. New tokens land in `_emit_hold` (a small lookahead buffer).
        #   2. We commit only the portion of `_emit_hold` that's
        #      definitively NOT the start of `<execute_python` —
        #      anything that could still extend into a fake tag opener
        #      stays held.
        if not isinstance(token, str):
            continue

        if _fake_tool_active:
            _fake_tool_buffer += token
            # Find the closer matching whichever opener matched.
            _closer = next(
                (c for o, c, fmt in _RESCUE_OPENERS if fmt == _fake_tool_format),
                "</execute_python>",
            )
            close_pos = _fake_tool_buffer.lower().find(_closer)
            if close_pos < 0:
                continue
            # Got a closer. Extract args based on format.
            tool_calls_to_run: list[tuple[str, dict]] = []
            if _fake_tool_format == "function_calls":
                # Body looks like (whitespace flexible):
                #   <function_calls>
                #     <invoke name="execute_python">
                #       <parameter name="code">CODE</parameter>
                #     </invoke>
                #     [more invokes possible]
                #   </function_calls>
                # Parse with regex — XML parsing is overkill and the
                # code parameter often contains `<` / `>` which would
                # confuse a strict parser anyway.
                body = _fake_tool_buffer[:close_pos]
                # Each <invoke name="..."> ... </invoke> block
                _invoke_re = re.compile(
                    r'<invoke\s+name="([^"]+)">(.*?)</invoke>',
                    re.DOTALL,
                )
                _param_re = re.compile(
                    r'<parameter\s+name="([^"]+)">(.*?)</parameter>',
                    re.DOTALL,
                )
                for m_inv in _invoke_re.finditer(body):
                    tool_name = m_inv.group(1)
                    inv_body = m_inv.group(2)
                    args: dict = {}
                    for m_par in _param_re.finditer(inv_body):
                        args[m_par.group(1)] = m_par.group(2).strip()
                    tool_calls_to_run.append((tool_name, args))
            else:
                # Simple <execute_python>code</execute_python> form
                open_gt = _fake_tool_buffer.find(">")
                if 0 <= open_gt < close_pos:
                    code = _fake_tool_buffer[open_gt + 1:close_pos].strip()
                else:
                    code = _fake_tool_buffer[:close_pos].strip()
                tool_calls_to_run.append(("execute_python", {"code": code}))

            post_tag = _fake_tool_buffer[close_pos + len(_closer):]
            _fake_tool_active = False
            _fake_tool_buffer = ""
            _fake_tool_format = ""

            # Dispatch each captured tool call. Most rescues are a
            # single invoke; a multi-invoke rescue is rare but cheap
            # to handle. execute_python goes through the shared
            # dispatcher (thread offload + REPL-state restore +
            # code_exec SSE event) — same path as real tool calls.
            for tool_name, tool_args in tool_calls_to_run:
                try:
                    if tool_name == "execute_python":
                        code = tool_args.get("code") or ""
                        logger.info(
                            "[fake_tag_rescue] dispatching execute_python "
                            "(%d chars)", len(code),
                        )
                        tool_summary = await _dispatch_execute_python(
                            {"code": code}, ws=ws, event_bus=event_bus,
                            turn_state=turn_state,
                            current_user=current_user,
                        )
                        await _send_event(ws, {
                            "event": "tool_call",
                            "name": "execute_python",
                            "result": tool_summary[:MAX_RESULT_PREVIEW],
                        })
                        _emit_event(event_bus, {
                            "type": "tool_call",
                            "name": "execute_python",
                            "args": {"code": code},
                            "result": tool_summary[:MAX_RESULT_PREVIEW],
                            "rescued": True,
                        })
                    else:
                        # Generic rescue — dispatch through the regular
                        # tool registry. Covers things like fetch_url
                        # if the model emits the same XML form for it.
                        from mother.llm.tools import (
                            ToolContext as _TC,
                            dispatch_tool_call as _dispatch,
                        )
                        from mother.api.server import get_state as _gs
                        _shared_http = _gs().get("tool_http_client")
                        ctx2 = _TC(
                            http_client=_shared_http,
                            rag_base=(cfg.rag.api_base if cfg and cfg.rag else "http://127.0.0.1:8123"),
                            rag_timeout=(cfg.rag.timeout_ms / 1000.0 if cfg and cfg.rag else 0.5),
                            current_user=current_user,
                            repl_session_key=id(ws),
                        )
                        result = await asyncio.to_thread(
                            _dispatch, tool_name, tool_args, ctx2,
                        )
                        await _send_event(ws, {
                            "event": "tool_call",
                            "name": tool_name,
                            "result": result,
                        })
                        _emit_event(event_bus, {
                            "type": "tool_call",
                            "name": tool_name,
                            "args": tool_args,
                            "result": result[:MAX_RESULT_PREVIEW],
                            "rescued": True,
                        })
                except Exception as _re:
                    logger.warning(
                        "[fake_tag_rescue] dispatch %s failed: %s",
                        tool_name, _re,
                    )

            # Treat post-tag content as new streamed text — feed it
            # back into the holding buffer so the same lookahead /
            # safe-emit logic applies.
            _emit_hold = post_tag
        else:
            _emit_hold += token

        # Decide what's safe to commit from _emit_hold. Check ALL
        # known opener formats; pick the EARLIEST occurrence so we
        # don't miss a `<function_calls>` because we matched on
        # `<execute_python>` later in the buffer (or vice versa).
        _hold_lower = _emit_hold.lower()
        _best_open: int = -1
        _best_format = ""
        for _opener, _closer, _fmt in _RESCUE_OPENERS:
            _idx = _hold_lower.find(_opener)
            if _idx >= 0 and (_best_open < 0 or _idx < _best_open):
                _best_open = _idx
                _best_format = _fmt
        if _best_open >= 0:
            # Real opener detected. Anything before it is committed text;
            # everything from the opener onward enters fake-capture mode.
            _safe = _emit_hold[:_best_open]
            _capture_start = _emit_hold[_best_open:]
            _emit_hold = ""
            if _safe:
                if t_first_token[0] == 0.0:
                    t_first_token[0] = time.monotonic()
                    logger.info("[TIMING] first_token: ttft=%.3fs", t_first_token[0] - t_llm_thread_start)
                full_response.append(_safe)
                await _send_event(ws, {"event": "llm_token", "token": _safe})
                sentence_buffer += _safe
            _fake_tool_active = True
            _fake_tool_buffer = _capture_start
            _fake_tool_format = _best_format
        else:
            # No full opener yet — but the tail of _emit_hold might be
            # the start of one. Hold back any tail that matches a
            # prefix of ANY known opener; commit everything before.
            _hold_back = 0
            for _opener, _closer, _fmt in _RESCUE_OPENERS:
                for _plen in range(min(len(_opener), len(_emit_hold)), 0, -1):
                    if _hold_lower.endswith(_opener[:_plen]):
                        if _plen > _hold_back:
                            _hold_back = _plen
                        break
            _safe_len = len(_emit_hold) - _hold_back
            if _safe_len > 0:
                _safe = _emit_hold[:_safe_len]
                _emit_hold = _emit_hold[_safe_len:]
                if t_first_token[0] == 0.0:
                    t_first_token[0] = time.monotonic()
                    logger.info("[TIMING] first_token: ttft=%.3fs", t_first_token[0] - t_llm_thread_start)
                full_response.append(_safe)
                await _send_event(ws, {"event": "llm_token", "token": _safe})
                sentence_buffer += _safe
            # else: everything is held; fall through to sentence chunking
            # which will operate on whatever's already in sentence_buffer.

        while True:
            # Clause-level boundary for the first chunk only. We also
            # require the first clause to be at least ~18 characters —
            # firing on "Well," (5 chars) cuts latency but sounds
            # chopped; waiting for a real clause (~3+ words) lands
            # naturally.
            pat = sentence_boundary if _first_chunk_emitted else first_clause_boundary
            m = pat.search(sentence_buffer)
            if not m:
                break
            seg_end = m.end()
            seg_candidate = sentence_buffer[:seg_end].strip()
            if not _first_chunk_emitted and len(seg_candidate) < 18:
                # Too short to sound natural as an opening phrase.
                # Treat as if no boundary found and keep buffering until
                # either the minimum length or a sentence boundary.
                break
            seg = seg_candidate
            sentence_buffer = sentence_buffer[seg_end:]
            if seg:
                _queue_tts(seg)
                _first_chunk_emitted = True

    # Flush anything held in the lookahead buffer — the stream is
    # over so there's no fake tag coming. Treat any leftover as text.
    if _emit_hold:
        full_response.append(_emit_hold)
        await _send_event(ws, {"event": "llm_token", "token": _emit_hold})
        sentence_buffer += _emit_hold
        _emit_hold = ""

    # Flush remaining sentence buffer
    tail = sentence_buffer.strip()
    if tail:
        _queue_tts(tail)

    # Signal TTS worker to exit and wait for it to drain the queue.
    # Bounded wait so a stuck synth can't hang the whole turn — if the
    # worker doesn't drain within 30s something is wrong downstream
    # (Deepgram hung, network stall, etc.) and we cancel it explicitly.
    tts_pending.put_nowait(None)
    try:
        await asyncio.wait_for(tts_worker_task, timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("[TTS] worker drain exceeded 30s — cancelling")
        tts_worker_task.cancel()
        try:
            await tts_worker_task
        except (asyncio.CancelledError, Exception):
            pass
    except asyncio.CancelledError:
        tts_worker_task.cancel()
        raise

    # End-of-turn bookkeeping for both normal turns and tool-call turns.
    # Tool-call turns now produce a narration via the re-prompt flow
    # above, so full_response holds the in-character reply and we can
    # treat the two branches identically: send llm_done, save to conv
    # memory, persist to disk.
    final_text = "".join(full_response)
    await _send_event(ws, {"event": "llm_done", "full_text": final_text})
    assistant_content_for_memory = final_text if final_text else "(tool call)"

    # Threaded: add_user() triggers rolling-summary compaction once the
    # history window is full, and compaction makes a SYNCHRONOUS LLM
    # call (up to 6s timeout). Running that inline on the event loop
    # froze the entire server right at end-of-turn — exactly when the
    # user is most likely to fire the next question. (The comment in
    # conversation.py assumed this ran on a worker thread; it didn't.)
    def _update_conv_memory():
        try:
            from mother.memory.conversation import get_memory
            conv = get_memory()
            conv.add_user(text)
            conv.add_assistant(assistant_content_for_memory)
            if current_user is not None:
                try:
                    conv.save(current_user.user_id)
                    logger.debug(
                        "[conv] saved %d turns to user=%s",
                        len(conv._history), current_user.user_id,
                    )
                except Exception as e:
                    logger.warning("conv.save failed: %s", e)
            else:
                logger.debug("conv save skipped: current_user is None")
        except Exception as e:
            logger.warning("conv update failed: %s", e)

    await asyncio.to_thread(_update_conv_memory)

    # ── Long-conversation semantic memory ──
    # Embed the completed exchange and add it to the per-user FAISS
    # index. Done in a background thread because embedding takes
    # 30-100ms — we'd rather the user-facing "llm_done" land first.
    # Trivial greetings + empty assistant outputs get filtered inside
    # add_exchange so we don't index noise.
    if current_user is not None and final_text:
        try:
            from mother.memory.long_memory import get_long_memory
            _long_mem = get_long_memory(current_user.user_id)
            _user_text = text
            _asst_text = final_text

            def _index_exchange():
                try:
                    _long_mem.add_exchange(_user_text, _asst_text)
                except Exception as ie:
                    logger.debug("long_memory add failed: %s", ie)

            threading.Thread(target=_index_exchange, daemon=True).start()
        except Exception as e:
            logger.debug("long_memory schedule failed: %s", e)

    # Broadcast to SSE: response + latency breakdown for the dashboard
    _emit_event(event_bus, {
        "type": "response",
        "text": "".join(full_response) if full_response else "",
    })
    # Latency event — only useful when we actually got tokens/audio
    if t_first_token[0] > 0.0:
        _emit_event(event_bus, {
            "type": "latency",
            "ttft_ms": int((t_first_token[0] - t_llm_thread_start) * 1000),
            "tts_first_chunk_ms": (
                int((t_first_tts_ready[0] - t_request_start) * 1000)
                if t_first_tts_ready[0] > 0.0 else None
            ),
            "total_ms": int((time.monotonic() - t_request_start) * 1000),
            "tier": tier,
        })


async def _handle_fast_path(intent, text: str) -> str:
    """Dispatch fast-path intents to their handlers.

    Handlers return (handled: bool, response: str | None).
    We extract the response text, falling back to a generic message.
    """
    from mother.core.intent import Intent

    if intent == Intent.WEATHER:
        try:
            from mother.handlers.weather import handle_weather_command
            handled, response = await asyncio.to_thread(handle_weather_command, text)
            return response or "Weather data unavailable."
        except Exception as e:
            return f"Weather data unavailable: {e}"

    if intent in (Intent.FINANCE_QUOTE, Intent.FINANCE_NEWS):
        try:
            from mother.handlers.finance import handle_finance_command, handle_finance_news
            if intent == Intent.FINANCE_NEWS:
                handled, response = await asyncio.to_thread(handle_finance_news, text)
            else:
                handled, response = await asyncio.to_thread(handle_finance_command, text)
            return response or "Finance data unavailable."
        except Exception as e:
            return f"Finance data unavailable: {e}"

    if intent in (Intent.REMINDER_SET, Intent.REMINDER_LIST):
        try:
            from mother.core.reminders import parse_reminder, add_reminder, list_reminders
            from mother.identity.speaker import get_current_user
            cu = get_current_user()
            uid = cu.user_id if cu else "unknown"
            if intent == Intent.REMINDER_LIST:
                pending = list_reminders(uid)
                if pending:
                    lines = [r.get("text", "?") for r in pending[:5]]
                    return "Your reminders: " + "; ".join(lines) + "."
                return "You have no pending reminders."
            parsed = parse_reminder(text)
            if parsed:
                rtext, rtime = parsed
                add_reminder(uid, rtext, rtime)
                when = rtime.strftime("%I:%M %p").lstrip("0")
                return f"Reminder set for {when}: {rtext}."
            return "I couldn't parse that reminder. Try: remind me at 3 PM to call the dentist."
        except Exception as e:
            return f"Reminder error: {e}"


    if intent == Intent.MEMORY_EXPLICIT:
        try:
            from mother.handlers.memory_commands import handle_memory_query, handle_remember_command
            from mother.memory.manager import get_current_user_memory
            memory = get_current_user_memory()
            low = text.lower()
            if "remember" in low:
                handled, response = await asyncio.to_thread(
                    handle_remember_command, text, memory
                )
            else:
                handled, response = await asyncio.to_thread(
                    handle_memory_query, text, memory
                )
            return response or "I don't have that information."
        except Exception as e:
            return f"Memory error: {e}"

    return "I'm not sure how to handle that."
