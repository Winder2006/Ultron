"""Test voice recording → transcription → response latency end-to-end.

Simulates a real recording flow: sends audio in 256ms chunks while "speaking",
then sends stop and measures when the first audio reply comes back.
"""
import asyncio
import json
import time
import websockets
import numpy as np
import wave
from pathlib import Path


def generate_test_audio(duration_s: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a silent audio buffer (Deepgram will return empty transcript).
    To test real recognition you'd need to feed it actual speech audio."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def load_wav_as_float32(path: Path) -> np.ndarray:
    """Load a WAV file as float32 mono at 16kHz."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        nch = wf.getnchannels()
        pcm = np.frombuffer(frames, dtype=np.int16)
        if nch > 1:
            pcm = pcm.reshape(-1, nch).mean(axis=1).astype(np.int16)
        audio = pcm.astype(np.float32) / 32767.0
        if sr != 16000:
            # Simple resample — good enough for a timing test
            from math import ceil
            ratio = 16000 / sr
            new_len = int(ceil(len(audio) * ratio))
            indices = (np.arange(new_len) / ratio).astype(np.int64)
            indices = np.clip(indices, 0, len(audio) - 1)
            audio = audio[indices]
        return audio


async def main():
    # Use an existing WAV file if available, else generate silence
    test_wav = Path("scripts/test.wav")
    if test_wav.exists():
        audio = load_wav_as_float32(test_wav)
        print(f"[Setup] Loaded {len(audio)/16000:.1f}s audio from {test_wav}")
    else:
        audio = generate_test_audio(2.0)
        print(f"[Setup] Using {len(audio)/16000:.1f}s of silence (no test.wav found)")

    uri = "ws://localhost:8300/ws/voice"
    async with websockets.connect(uri) as ws:
        # Start recording
        t_start = time.monotonic()
        await ws.send(json.dumps({"action": "start"}))
        print(f"[{time.monotonic() - t_start:.3f}s] start sent")

        # Read recording_started
        reply = json.loads(await ws.recv())
        print(f"[{time.monotonic() - t_start:.3f}s] {reply.get('event')}")

        # Stream audio in 256ms chunks (like the real mic does)
        chunk_size = 4096  # 256ms at 16kHz
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(chunk_size / 16000)  # simulate real-time capture

        # Stop recording
        t_stop = time.monotonic() - t_start
        await ws.send(json.dumps({"action": "stop"}))
        print(f"[{t_stop:.3f}s] stop sent")

        # Collect events
        first_stt_t = None
        first_token_t = None
        first_tts_t = None
        async for raw in ws:
            ev = json.loads(raw)
            elapsed = time.monotonic() - t_start
            since_stop = elapsed - t_stop
            evt = ev.get("event")
            if evt == "stt":
                if first_stt_t is None:
                    first_stt_t = since_stop
                    print(f"[{elapsed:.3f}s] [+{since_stop:.3f}s after stop] STT: {ev.get('text')!r}")
            elif evt == "llm_token":
                if first_token_t is None:
                    first_token_t = since_stop
                    print(f"[{elapsed:.3f}s] [+{since_stop:.3f}s after stop] FIRST LLM TOKEN: {ev.get('token')!r}")
            elif evt == "tts_ready":
                if first_tts_t is None:
                    first_tts_t = since_stop
                    print(f"[{elapsed:.3f}s] [+{since_stop:.3f}s after stop] FIRST AUDIO READY (legacy WAV)")
            elif evt == "tts_chunk":
                if first_tts_t is None:
                    first_tts_t = since_stop
                    print(f"[{elapsed:.3f}s] [+{since_stop:.3f}s after stop] FIRST PCM CHUNK (streaming)")
            elif evt == "tts_start":
                pass  # per-sentence marker
            elif evt == "tts_end":
                pass  # per-sentence marker
            elif evt == "llm_done":
                print(f"[{elapsed:.3f}s] [+{since_stop:.3f}s after stop] DONE: {ev.get('full_text', '')[:80]!r}")
                break
            elif evt == "error":
                print(f"[{elapsed:.3f}s] ERROR: {ev.get('message')}")
                break

        print()
        print(f"Time from stop→STT:       {first_stt_t:.3f}s" if first_stt_t else "No STT")
        print(f"Time from stop→LLM token: {first_token_t:.3f}s" if first_token_t else "No LLM")
        print(f"Time from stop→audio:     {first_tts_t:.3f}s" if first_tts_t else "No audio")


asyncio.run(main())
