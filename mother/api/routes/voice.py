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

    chunk_q._loop = asyncio.get_event_loop()
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
        if _is_morning_window():
            try:
                from mother.tools.weather_tool import get_weather as _gw
                _w = await asyncio.wait_for(
                    asyncio.to_thread(_gw, 43.0389, -87.9065,
                                      fahrenheit=True, mph=True),
                    timeout=1.5,  # brief — if weather is slow, skip it
                )
                if isinstance(_w, dict) and "temperature" in _w:
                    _t = round(_w.get("temperature") or 0)
                    _desc = _w.get("description") or ""
                    _greet_weather = f"It's {_t}°{' and ' + _desc if _desc else ''} outside."
            except (asyncio.TimeoutError, Exception):
                _greet_weather = None
        _greet_line = maybe_morning_greeting(
            _greet_user.user_id, _greet_user.display_name,
            weather_summary=_greet_weather,
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
    audio_queue: Optional[asyncio.Queue] = None
    audio_buffer: list[np.ndarray] = []
    stt_task: Optional[asyncio.Task] = None
    transcript_queue: Optional[asyncio.Queue] = None
    is_recording = False
    sample_rate = 16000

    async def _drain_transcripts(aq: asyncio.Queue, tq: asyncio.Queue):
        """Background task: pull streaming transcripts from Deepgram.

        Pushes final transcripts into `tq` (consumed by the recording-
        stop handler). Also emits interim + final `stt` events directly
        on the WebSocket so the browser sees the live transcript AND
        can auto-stop recording the moment Deepgram flags final —
        without waiting for the server-side stop-handler to propagate.
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

                    # Try to get streamed Deepgram transcript (should be ~0-200ms wait
                    # since audio was already being sent during recording)
                    live_text = ""
                    if stt_task is not None and transcript_queue is not None:
                        try:
                            t_wait = time.monotonic()
                            live_text = await asyncio.wait_for(
                                transcript_queue.get(), timeout=1.5
                            )
                            logger.info(
                                "[TIMING] live_stt: waited=%.3fs text=%r",
                                time.monotonic() - t_wait, live_text[:60],
                            )
                        except asyncio.TimeoutError:
                            logger.warning("Live STT timed out, falling back to batch")

                    # Cleanup the streaming STT task — cancel AND await
                    # so the Deepgram socket is fully torn down before
                    # the next utterance opens a new one. Without this
                    # the next start can reuse a dying socket and see
                    # stale audio from the previous utterance.
                    if stt_task is not None and not stt_task.done():
                        stt_task.cancel()
                        try:
                            await stt_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    stt_task = None
                    audio_queue = None
                    transcript_queue = None

                    # Process utterance (pass live_text to skip batch transcription)
                    await _process_utterance(
                        ws, audio, sample_rate,
                        llm=llm, tts=tts, stt=stt,
                        streaming_stt=streaming_stt,
                        cfg=cfg, event_bus=event_bus,
                        vision_state=state.get("vision"),
                        pre_transcribed_text=live_text,
                        spawn=_spawn,
                    )

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
                    ambient_suppressed.set()
                    ambient_state["last_activity"] = time.monotonic()
                    text = data.get("text", "")
                    if text:
                        await _process_text_query(
                            ws, text,
                            llm=llm, tts=tts, cfg=cfg,
                            event_bus=event_bus,
                            vision_state=state.get("vision"),
                        )

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
    )


async def _process_text_query(
    ws: WebSocket,
    text: str,
    *,
    llm, tts, cfg, event_bus, vision_state,
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
    # Build messages with context
    system_prompt = cfg.llm.system_prompt if cfg else "You are ULTRON."

    # Inject vision context if available
    if has_vision:
        try:
            from mother.vision.context import build_vision_context
            vision_text = build_vision_context(vision_state)
            if vision_text:
                system_prompt += f"\n\n[Visual context: {vision_text}]"
        except Exception:
            pass

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
    if cfg and cfg.rag and cfg.rag.enabled and cfg.rag.on_demand and not _is_trivial:
        try:
            import time as _time
            from mother.core.context_injection import fetch_context
            _rag_t0 = _time.monotonic()
            rag_block = await fetch_context(
                text,
                api_base=cfg.rag.api_base,
                notes_k=min(cfg.rag.k, 2),
                code_k=getattr(cfg.rag, "code_k", 3),
                timeout_ms=cfg.rag.timeout_ms,
                enable_notes=True,
                enable_code=getattr(cfg.rag, "code_enabled", True),
            )
            _rag_dt = _time.monotonic() - _rag_t0
            if rag_block:
                system_prompt += f"\n\n{rag_block}"
                # Emit observability event: summarise the retrieval so the
                # dashboard can display what context Ultron was given.
                _emit_event(event_bus, {
                    "type": "rag_hit",
                    "query": text[:80],
                    "duration_ms": int(_rag_dt * 1000),
                    "preview": rag_block[:300],
                    "blocks": rag_block.count("\n- "),
                })
        except Exception as e:
            logger.warning("RAG fetch failed: %s", e)

    # User identity + personal memory. Mirrors the orchestrator's
    # (process_text) context injection so the WS path has feature parity.
    if current_user is not None:
        try:
            from mother.identity.speaker import get_user_context_for_prompt
            user_ctx = get_user_context_for_prompt(current_user)
            if user_ctx:
                system_prompt += f"\n\nUser context: {user_ctx}"
        except Exception as e:
            logger.debug("user-context injection skipped: %s", e)

        # Skip memory retrieval on trivial queries — greetings and
        # acks don't benefit from "user facts" context, and every
        # retrieval costs a DB hit + an embedding lookup (~50-100ms).
        # Same whitelist as the RAG skip above.
        if not _is_trivial:
            try:
                from mother.memory.manager import get_user_memory
                user_mem = get_user_memory(current_user.user_id)
                if user_mem is not None:
                    mem_ctx = user_mem.get_context_for_prompt(text, max_items=5)
                    if mem_ctx:
                        system_prompt += f"\n\nMemory context: {mem_ctx}"
            except Exception as e:
                logger.debug("memory-context injection skipped: %s", e)

    # Conversation memory — load from disk on first use of this process
    # so multi-turn history survives server restarts. `_conv_loaded_for`
    # tracks which user_id we've already loaded to avoid re-reading the
    # file on every turn.
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
        # conversation history into the LLM. Without this, the model
        # sees the tail of the previous exchange (say, a long talk
        # about AI existential risk) and treats "hello" as a
        # continuation — producing odd "I've been processing the
        # implications of your current research" non-sequiturs. The
        # history is still persisted on disk; we just hide it from
        # THIS turn's prompt so the greeting lands clean.
        if _is_trivial:
            messages = [ChatMessage(role="system", content=system_prompt)]
            messages.append(ChatMessage(role="user", content=text))
        else:
            messages = conv.get_messages(system_prompt)
            messages.append(ChatMessage(role="user", content=text))
    except Exception:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=text),
        ]

    # Set tier if tiered driver
    if isinstance(llm, TieredLLMDriver) and tier:
        llm.set_tier(tier)

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

    # Timing instrumentation
    t_request_start = time.monotonic()
    t_first_token = [0.0]
    t_first_tts_ready = [0.0]

    # Token queue filled by a background thread, drained by the async loop
    token_queue: queue.Queue = queue.Queue()
    SENTINEL = object()

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

    def _producer():
        """Run blocking LLM stream in a thread, push tokens into queue.

        If the first stream returns zero tokens (observed with tier3 Sonnet
        + tools + large conversation history — the model returns
        finish_reason=end_turn with no content and no tool call), retry
        once without tools. Better to get a direct text answer than silence.
        """
        tok_count = 0
        try:
            for tok in llm.stream_chat(
                messages,
                temperature=cfg.llm.temperature if cfg else 0.7,
                max_tokens=cfg.llm.max_tokens if cfg else None,
                tools=_tools_for_call,
            ):
                token_queue.put(tok)
                tok_count += 1
        except Exception as e:
            logger.warning("LLM stream error: %s", e)

        if tok_count == 0:
            logger.warning(
                "[LLM] empty stream — retrying without tools (tier=%s msgs=%d)",
                tier, len(messages),
            )
            try:
                for tok in llm.stream_chat(
                    messages,
                    temperature=cfg.llm.temperature if cfg else 0.7,
                    max_tokens=cfg.llm.max_tokens if cfg else None,
                ):
                    token_queue.put(tok)
                    tok_count += 1
            except Exception as e:
                logger.warning("LLM retry stream error: %s", e)

        token_queue.put(SENTINEL)

    t_llm_thread_start = time.monotonic()

    threading.Thread(target=_producer, daemon=True).start()

    # Streaming TTS: synthesize each sentence via Deepgram and stream raw PCM
    # chunks to the browser as they arrive (~150ms time-to-first-chunk vs 1200ms
    # for the full WAV). Browser plays chunks with Web Audio API scheduling.
    tts_pending: asyncio.Queue = asyncio.Queue()

    # Check if TTS supports PCM streaming (Deepgram does; Kokoro/Piper don't).
    has_pcm_stream = hasattr(tts, "synthesize_stream_pcm")

    async def _stream_one_sentence(seg: str, sentence_id: int):
        """Stream a single sentence's PCM to the WebSocket.

        Streams chunks through in real-time (low latency), but buffers the
        TAIL so we can trim trailing silence and apply a 5ms fade-out to the
        last emitted chunk. Leading silence gets a fade-in on the first chunk.
        This produces click-free boundaries between sentences.
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

        # Bridge blocking generator to asyncio with a chunk queue
        chunk_q: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                for chunk in tts.synthesize_stream_pcm(seg):
                    chunk_q._loop.call_soon_threadsafe(chunk_q.put_nowait, chunk)
            except Exception as e:
                logger.warning("TTS stream error: %s", e)
            finally:
                chunk_q._loop.call_soon_threadsafe(chunk_q.put_nowait, None)

        chunk_q._loop = asyncio.get_event_loop()
        threading.Thread(target=_produce, daemon=True).start()

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
        while True:
            seg = await tts_pending.get()
            if seg is None:
                break
            sentence_counter[0] += 1
            if has_pcm_stream:
                await _stream_one_sentence(seg, sentence_counter[0])
            else:
                await _fallback_tts(seg)

    tts_worker_task = asyncio.create_task(_tts_worker())

    def _queue_tts(seg: str):
        tts_pending.put_nowait(seg)

    loop = asyncio.get_event_loop()

    # Drain the token queue
    while True:
        # Pull next token — await-safe via run_in_executor on a blocking get
        token = await loop.run_in_executor(None, token_queue.get)
        if token is SENTINEL:
            break

        # Handle tool call sentinel
        if isinstance(token, str) and token.startswith("__TOOL_CALL__:"):
            tool_call_detected = True
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
                import httpx
                user_mem = (
                    get_user_memory(current_user.user_id)
                    if current_user is not None else None
                )
                ctx = ToolContext(
                    http_client=httpx.Client(timeout=5.0),
                    rag_base=(cfg.rag.api_base if cfg and cfg.rag else "http://127.0.0.1:8123"),
                    rag_timeout=(cfg.rag.timeout_ms / 1000.0 if cfg and cfg.rag else 0.5),
                    user_memory=user_mem,
                    current_user=current_user,
                    add_reminder_fn=add_reminder,
                )
                result = dispatch_tool_call(tool_name, tool_args, ctx)

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
                is_terse = tool_name in TERSE_TOOLS

                if is_terse:
                    await _send_event(ws, {"event": "llm_token", "token": result})
                    _queue_tts(result)
                    full_response.append(result)
                else:
                    # Re-prompt the LLM with the tool result as context.
                    # Ultron will give a tight, in-character answer
                    # drawn from the data. Stream it through the same
                    # sentence-boundary → TTS path as normal responses.
                    from mother.llm.drivers import ChatMessage as _CM
                    narration_msgs = list(messages)
                    narration_msgs.append(_CM(
                        role="tool",
                        content=f"{tool_name} result:\n{result}",
                    ))
                    narration_msgs.append(_CM(
                        role="user",
                        content=(
                            "Answer my question using that data. "
                            "One or two short sentences. Your voice, "
                            "not a transcription of the raw results."
                        ),
                    ))

                    narr_q: queue.Queue = queue.Queue()

                    def _narrate_producer():
                        try:
                            for tok in llm.stream_chat(
                                narration_msgs,
                                temperature=cfg.llm.temperature if cfg else 0.7,
                                max_tokens=(cfg.llm.max_tokens if cfg else None) or 120,
                            ):
                                narr_q.put(tok)
                        except Exception as e:
                            logger.warning("narration stream error: %s", e)
                        finally:
                            narr_q.put(SENTINEL)

                    threading.Thread(target=_narrate_producer, daemon=True).start()

                    narration_buf = ""
                    while True:
                        ntok = await loop.run_in_executor(None, narr_q.get)
                        if ntok is SENTINEL:
                            break
                        if not isinstance(ntok, str):
                            continue
                        # Skip any further tool-call sentinels (shouldn't
                        # happen on a simple narration call but guard
                        # against recursion).
                        if ntok.startswith("__TOOL_CALL__:"):
                            continue
                        full_response.append(ntok)
                        await _send_event(ws, {"event": "llm_token", "token": ntok})
                        narration_buf += ntok
                        # Same clause-boundary chunking as the main path
                        pat = (
                            sentence_boundary
                            if _first_chunk_emitted
                            else first_clause_boundary
                        )
                        while True:
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
                # Don't leave the user hanging: we already consumed the
                # __TOOL_CALL__ sentinel, so the outer stream won't
                # produce a fallback. Re-prompt the LLM without tools
                # and TTS whatever it says — better than dead air.
                logger.warning("Tool call dispatch error: %s", e)
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

        # Regular text token — send immediately, buffer for sentence boundary
        if t_first_token[0] == 0.0:
            t_first_token[0] = time.monotonic()
            logger.info("[TIMING] first_token: ttft=%.3fs", t_first_token[0] - t_llm_thread_start)
        full_response.append(token)
        await _send_event(ws, {"event": "llm_token", "token": token})

        sentence_buffer += token
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

    if intent in (Intent.IDENTITY_CLAIM, Intent.IDENTITY_QUERY):
        return "Identity handled via voice pipeline."

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
