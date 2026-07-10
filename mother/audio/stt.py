"""Speech-to-text engines for MOTHER.

Contains:
- STTEngine / FasterWhisperSTT: Original sync interface (used by cli.py PTT loop)
- StreamingSTT: New async streaming interface with Deepgram primary + Whisper fallback
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("mother.stt")


# ---------------------------------------------------------------------------
# Original sync interface — preserved for backward compatibility with cli.py
# ---------------------------------------------------------------------------

class STTEngine:
    """Abstract base for speech-to-text engines."""

    def transcribe_wav(self, wav_path: str, language: Optional[str] = None) -> str:
        raise NotImplementedError

    def transcribe_pcm(
        self, pcm_samples, sample_rate: int, language: Optional[str] = None
    ) -> str:
        """Transcribe from an in-memory PCM array (float32/float64/int16)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, pcm_samples, sample_rate)
        try:
            return self.transcribe_wav(tmp_path, language=language)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


@dataclass
class FasterWhisperConfig:
    model_size: str = "small.en"
    device: str = "auto"
    compute_type: str = "auto"
    language: Optional[str] = None
    beam_size: int = 5


class FasterWhisperSTT(STTEngine):
    """Faster-Whisper transcription — runs locally on CPU or CUDA."""

    def __init__(self, cfg: FasterWhisperConfig) -> None:
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model_size, device=cfg.device, compute_type=cfg.compute_type
        )

    def stream_transcribe(self, audio_iter, sample_rate: int, language: Optional[str] = None):
        """Yield partial text chunks for streaming pipelines."""
        buffer = np.zeros((0,), dtype="float32")
        chunk_ms = 300
        samples_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
        for chunk in audio_iter:
            if chunk is None or len(chunk) == 0:
                continue
            buffer = np.concatenate([buffer, chunk.astype("float32")])
            if buffer.shape[0] >= samples_per_chunk:
                text = self.transcribe_pcm(buffer, sample_rate, language)
                if text:
                    yield text
                buffer = np.zeros((0,), dtype="float32")

    def transcribe_wav(self, wav_path: str, language: Optional[str] = None) -> str:
        lang = language or self.cfg.language or "en"
        beam = getattr(self.cfg, "beam_size", 5)

        def _run(vad: bool):
            return self.model.transcribe(
                wav_path,
                language=lang,
                vad_filter=vad,
                beam_size=beam,
                temperature=0.0,
            )

        try:
            segments, info = _run(vad=True)
        except Exception:
            segments, info = _run(vad=False)
        texts: list[str] = []
        for seg in segments:
            if getattr(seg, "text", None):
                texts.append(seg.text)
        return " ".join(texts).strip()


# ---------------------------------------------------------------------------
# New async streaming interface — Deepgram primary, Whisper fallback
# ---------------------------------------------------------------------------

@dataclass
class DeepgramConfig:
    model: str = "nova-3"
    language: str = "en-US"
    encoding: str = "linear16"
    sample_rate: int = 16000
    smart_format: bool = True
    interim_results: bool = True
    utterance_end_ms: int = 1000


class StreamingSTT:
    """Async streaming STT with automatic fallback.

    Primary: Deepgram Nova-3 via WebSocket (streaming, <200ms latency).
    Fallback: Faster-Whisper base.en (activates if Deepgram unreachable).

    Usage:
        stt = StreamingSTT()
        await stt.init()
        async for transcript, is_final in stt.stream(audio_queue):
            if is_final:
                process(transcript)
    """

    def __init__(
        self,
        deepgram_cfg: Optional[DeepgramConfig] = None,
        whisper_cfg: Optional[FasterWhisperConfig] = None,
    ):
        self._dg_cfg = deepgram_cfg or DeepgramConfig()
        self._wh_cfg = whisper_cfg or FasterWhisperConfig(
            model_size="base.en", device="cpu", compute_type="int8", beam_size=3
        )
        self._whisper: Optional[FasterWhisperSTT] = None
        self._deepgram_available = False
        self._dg_api_key = os.environ.get("DEEPGRAM_API_KEY", "")

    async def init(self):
        """Initialize engines. Call once at startup."""
        # Always pre-load Whisper so fallback has no cold start
        logger.info("[STT] Pre-loading Faster-Whisper %s...", self._wh_cfg.model_size)
        self._whisper = FasterWhisperSTT(self._wh_cfg)
        logger.info("[STT] Faster-Whisper loaded (fallback ready)")

        # Check Deepgram connectivity. Wrapped in to_thread so the
        # 2-second blocking socket connect can't freeze the asyncio
        # loop during server startup.
        self._deepgram_available = await asyncio.to_thread(
            self._check_deepgram_available
        )
        if self._deepgram_available:
            logger.info("[STT] Deepgram API reachable — using as primary")
        else:
            logger.warning("[STT] Deepgram unreachable — using Faster-Whisper only")

    def _check_deepgram_available(self) -> bool:
        """Quick connectivity check to Deepgram API."""
        if not self._dg_api_key:
            logger.warning("[STT] DEEPGRAM_API_KEY not set")
            return False
        try:
            sock = socket.create_connection(("api.deepgram.com", 443), timeout=2.0)
            sock.close()
            return True
        except (OSError, socket.timeout):
            return False

    async def stream(
        self, audio_queue: asyncio.Queue
    ) -> AsyncIterator[tuple[str, bool]]:
        """Yields (partial_transcript, is_final) tuples.

        Reads 16kHz linear16 PCM chunks from audio_queue.
        Automatically falls back to Whisper if Deepgram fails.
        """
        # Re-check Deepgram availability
        if self._deepgram_available and self._dg_api_key:
            try:
                async for result in self._deepgram_stream(audio_queue):
                    yield result
                return
            except Exception as e:
                logger.warning("[STT] Deepgram stream failed: %s — falling back to Whisper", e)
                self._deepgram_available = False

        # Whisper fallback: collect all audio then transcribe
        async for result in self._whisper_stream(audio_queue):
            yield result

    async def _deepgram_stream(
        self, audio_queue: asyncio.Queue
    ) -> AsyncIterator[tuple[str, bool]]:
        """Try real WebSocket streaming first; fall back to REST batch.

        WebSocket gives interim transcripts during speech and a final
        ~600ms after the user stops talking (governed by
        `utterance_end_ms`). REST waits until the user is done, then
        POSTs everything — adds 400-1400ms.

        Audio chunks are buffered as they arrive even on the WS path,
        so if the WS attempt aborts (HTTP 400, network drop, anything),
        we can replay the buffered audio through the REST fallback
        without losing the utterance.
        """
        # Buffer everything we receive — tee'd into both the WS sender
        # AND a fallback list so REST has the same audio to work with.
        ws_audio_q: asyncio.Queue = asyncio.Queue()
        buffered_chunks: list[np.ndarray] = []
        producer_done = asyncio.Event()

        async def _split_audio():
            """Pull from the source queue, push to WS queue + buffer."""
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        await ws_audio_q.put(None)
                        return
                    buffered_chunks.append(chunk)
                    await ws_audio_q.put(chunk)
            finally:
                producer_done.set()

        splitter_task = asyncio.create_task(_split_audio())

        ws_yielded_any_final = False
        try:
            async for text, is_final in self._deepgram_websocket(ws_audio_q):
                if is_final and text:
                    ws_yielded_any_final = True
                yield (text, is_final)
            if ws_yielded_any_final:
                # WS path succeeded — we're done.
                return
            # WS finished but never gave us a final transcript.
            # Could mean Deepgram closed early or audio was empty.
            # Treat it as a soft failure and try REST.
            logger.info("[STT] WS stream produced no final transcript — trying REST")
        except Exception as e:
            # If the WS already emitted a final, the caller has the
            # transcript — don't run REST, it would produce a duplicate
            # that re-triggers the whole LLM pipeline and doubles the
            # quota burn. Only fall back when we have nothing yet.
            if ws_yielded_any_final:
                logger.warning(
                    "[STT] WebSocket error after final (%s) — suppressing REST fallback", e,
                )
                return
            logger.warning(
                "[STT] WebSocket failed (%s) — falling back to REST", e,
            )
            # Drain the splitter so we don't leave it hanging
            try:
                ws_audio_q.put_nowait(None)
            except asyncio.QueueFull:
                pass

        # Wait for any remaining audio to land in the buffer
        try:
            await asyncio.wait_for(producer_done.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("[STT] splitter did not finish within 2s — proceeding with partial buffer")
        if not splitter_task.done():
            splitter_task.cancel()
        try:
            await splitter_task
        except (asyncio.CancelledError, Exception):
            pass

        # REST fallback: assemble the buffered audio + POST.
        if not buffered_chunks:
            return
        async for result in self._deepgram_rest(buffered_chunks):
            yield result

    async def _deepgram_websocket(
        self, audio_queue: asyncio.Queue
    ) -> AsyncIterator[tuple[str, bool]]:
        """Real WebSocket streaming via Deepgram SDK 6.1+ ``listen.v1.connect``.

        Critical: the SDK's params for boolean-ish flags
        (`interim_results`, `smart_format`, `punctuate`) MUST be the
        strings `"true"`/`"false"` — they go on a query string, and
        Python booleans serialize to `"True"` (capitalised) which
        Deepgram's URL parser rejects with HTTP 400. This was the bug
        that bit the first attempt.

        Runs the SDK's blocking context manager on a worker thread,
        bridging events back to asyncio via a queue.
        """
        from deepgram import DeepgramClient
        loop = asyncio.get_running_loop()
        result_q: asyncio.Queue[tuple[str, bool, bool]] = asyncio.Queue()

        cfg = self._dg_cfg
        # Convert Python bools to the SDK-required string literals.
        bool_to_str = lambda b: "true" if b else "false"

        def _runner() -> None:
            try:
                client = DeepgramClient(api_key=self._dg_api_key)
                with client.listen.v1.connect(
                    model=cfg.model,
                    language=cfg.language,
                    encoding=cfg.encoding,
                    sample_rate=cfg.sample_rate,
                    smart_format=bool_to_str(cfg.smart_format),
                    interim_results=bool_to_str(cfg.interim_results),
                    utterance_end_ms=cfg.utterance_end_ms,
                    punctuate="true",
                    # endpointing in ms — how much silence Deepgram's
                    # VAD waits before flipping speech_final=true. The
                    # default is 10ms which is too aggressive (cuts off
                    # mid-pause). 300ms is a comfortable middle ground:
                    # late enough that Deepgram doesn't fire on natural
                    # speech pauses, early enough that we don't sit
                    # waiting forever after the user stops.
                    endpointing=300,
                ) as socket:
                    import threading
                    audio_done = threading.Event()

                    def _audio_pump():
                        try:
                            while True:
                                fut = asyncio.run_coroutine_threadsafe(
                                    audio_queue.get(), loop,
                                )
                                chunk = fut.result()
                                if chunk is None:
                                    break
                                if chunk.dtype in (np.float32, np.float64):
                                    pcm16 = (
                                        np.clip(chunk, -1.0, 1.0) * 32767
                                    ).astype(np.int16)
                                else:
                                    pcm16 = chunk.astype(np.int16)
                                socket.send_media(pcm16.tobytes())
                        except Exception as e:
                            logger.warning("[STT] audio pump error: %s", e)
                        finally:
                            audio_done.set()
                            try:
                                socket.send_close_stream()
                            except Exception:
                                pass

                    pump_thread = threading.Thread(
                        target=_audio_pump, daemon=True, name="dg-audio-pump",
                    )
                    pump_thread.start()
                    # IMPORTANT: do NOT call socket.start_listening() —
                    # in Deepgram SDK 6.x it's a BLOCKING event-loop
                    # that drains every message via internal callbacks,
                    # leaving nothing for recv() to consume. Calling
                    # both makes recv() block forever (no transcripts
                    # ever arrive). Use recv() polling exclusively.

                    # Drain events until the socket closes or we get a
                    # final + audio_done.
                    #
                    # Deepgram emits two related flags on Results events:
                    #   • is_final     — transcript is fully refined
                    #                    (~600-1000ms after speech ends)
                    #   • speech_final — VAD detected end-of-utterance
                    #                    (~150-250ms earlier than is_final)
                    # We treat speech_final as good-enough-to-commit so we
                    # don't pay the extra refinement window. The transcript
                    # at speech_final is essentially identical to the
                    # eventual is_final on short utterances; on longer
                    # ones the tiny refinement (punctuation tweaks) isn't
                    # worth 200ms of latency.
                    while True:
                        try:
                            event = socket.recv()
                        except Exception as e:
                            logger.debug("[STT] recv ended: %s", e)
                            break
                        if event is None:
                            break
                        # Results events have channel.alternatives[0].transcript
                        channel = getattr(event, "channel", None)
                        if channel is not None:
                            try:
                                transcript = channel.alternatives[0].transcript or ""
                            except Exception:
                                transcript = ""
                            is_final = bool(getattr(event, "is_final", False))
                            speech_final = bool(getattr(event, "speech_final", False))
                            # Either flag means "this is committable text"
                            commit = is_final or speech_final
                            if transcript:
                                asyncio.run_coroutine_threadsafe(
                                    result_q.put((transcript, commit, False)),
                                    loop,
                                ).result()
                            # speech_final fires the moment Deepgram's VAD
                            # decides the user stopped — break immediately
                            # without waiting for the slower is_final
                            # refinement pass.
                            if commit and audio_done.is_set():
                                break

                    pump_thread.join(timeout=0.5)
            except Exception as e:
                # Signal abort to the async side. The error sentinel
                # MUST land before the close sentinel — otherwise the
                # consumer breaks on done=True without ever seeing the
                # "__error__" marker, and the REST fallback never fires.
                # We use .result() to enforce ordering: this call won't
                # return until the put completes on the loop thread.
                logger.warning("[STT] WS streaming error: %s", e)
                try:
                    asyncio.run_coroutine_threadsafe(
                        result_q.put(("__error__", False, False)), loop,
                    ).result()
                except Exception:
                    pass
            finally:
                # Single close sentinel, regardless of success/error path.
                try:
                    asyncio.run_coroutine_threadsafe(
                        result_q.put(("", False, True)), loop,
                    )
                except Exception:
                    pass

        import threading
        runner = threading.Thread(target=_runner, daemon=True, name="dg-stt-runner")
        t0 = time.monotonic()
        runner.start()

        had_error = False
        while True:
            text, is_final, done = await result_q.get()
            if text == "__error__":
                had_error = True
                continue
            if done:
                break
            yield (text, is_final)

        dur = time.monotonic() - t0
        logger.info("[STT] WS stream complete in %.3fs", dur)
        if had_error:
            raise RuntimeError("Deepgram WebSocket stream failed")

    async def _deepgram_rest(
        self, chunks: list
    ) -> AsyncIterator[tuple[str, bool]]:
        """Batch REST transcribe — the reliable fallback path.

        Used when the WebSocket connect fails or yields no final.
        Chunks are pre-buffered by the caller (the WS attempt tees
        audio into both paths so failover doesn't lose the utterance).
        """
        if not chunks:
            return

        audio = np.concatenate(chunks, axis=0)
        if audio.dtype in (np.float32, np.float64):
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm16 = audio.astype(np.int16)

        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())
        wav_bytes = buf.getvalue()

        from deepgram import AsyncDeepgramClient
        client = AsyncDeepgramClient(api_key=self._dg_api_key)

        t0 = time.monotonic()
        try:
            response = await client.listen.v1.media.transcribe_file(
                request=wav_bytes,
                model="nova-3",
                smart_format="true",
                punctuate="true",
            )
            dur = time.monotonic() - t0
            transcript = (
                response.results.channels[0].alternatives[0].transcript
                if response.results and response.results.channels
                else ""
            )
            logger.info("[STT] REST fallback transcribed in %.3fs: %r", dur, transcript)
            if transcript:
                yield (transcript, True)
        except Exception as e:
            logger.warning("[STT] REST fallback error: %s", e)
            raise

    async def _whisper_stream(
        self, audio_queue: asyncio.Queue
    ) -> AsyncIterator[tuple[str, bool]]:
        """Fallback: collect audio from queue, transcribe with Whisper.

        Buffers all audio until a None sentinel arrives, then transcribes.
        """
        logger.info("[STT] Falling back to Faster-Whisper")
        chunks: list[np.ndarray] = []

        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            if isinstance(chunk, np.ndarray):
                chunks.append(chunk.astype(np.float32))
            elif isinstance(chunk, (bytes, bytearray)):
                pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                chunks.append(pcm)

        if not chunks:
            return

        audio = np.concatenate(chunks)
        t0 = time.monotonic()
        text = await asyncio.to_thread(
            self._whisper.transcribe_pcm, audio, self._dg_cfg.sample_rate, "en"
        )
        latency_ms = (time.monotonic() - t0) * 1000
        logger.info("[STT] Whisper transcribed in %.0fms: %s", latency_ms, text[:80])

        if text.strip():
            yield (text.strip(), True)

    async def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Convenience: transcribe a complete audio array.

        BATCH path. Skips the WebSocket entirely and goes straight to
        Deepgram REST (or Whisper if Deepgram is unavailable). The
        WebSocket is only useful when audio is being streamed in real
        time; for a buffered audio array there's no streaming benefit
        and the WS handshake adds 200-500ms over plain REST. Voice
        route uses this as the fallback when its live WS attempt
        didn't deliver a final transcript.
        """
        if self._deepgram_available and self._dg_api_key:
            try:
                final_text = ""
                async for text, is_final in self._deepgram_rest([audio]):
                    if is_final and text:
                        final_text = text
                if final_text:
                    return final_text
            except Exception as e:
                logger.warning("[STT] REST batch failed: %s — using Whisper", e)

        # Fallback: Whisper local
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(audio)
        await queue.put(None)
        final_text = ""
        async for text, is_final in self._whisper_stream(queue):
            if is_final:
                final_text = text
        return final_text
