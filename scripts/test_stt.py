"""Phase 1 STT test — records 5 seconds of audio and prints streaming transcript.

Measures latency from end-of-speech to final transcript.
Expected: <200ms for Deepgram, <600ms for Whisper fallback.

Usage:
    python scripts/test_stt.py
    python scripts/test_stt.py --whisper-only   # force Whisper fallback
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np


async def test_streaming_stt(force_whisper: bool = False):
    """Record 5 seconds, transcribe, measure latency."""
    from mother.audio.stt import StreamingSTT, DeepgramConfig, FasterWhisperConfig

    print("=" * 60)
    print("MOTHER Phase 1 — STT Latency Test")
    print("=" * 60)

    # Init STT
    stt = StreamingSTT(
        deepgram_cfg=DeepgramConfig(),
        whisper_cfg=FasterWhisperConfig(
            model_size="base.en", device="cpu", compute_type="int8", beam_size=3
        ),
    )
    if force_whisper:
        stt._dg_api_key = ""  # disable Deepgram
    await stt.init()

    engine = "Deepgram" if stt._deepgram_available else "Faster-Whisper"
    print(f"\nEngine: {engine}")
    print(f"Recording 5 seconds of audio... Speak now!\n")

    # Record 5 seconds
    import sounddevice as sd

    duration = 5
    sample_rate = 16000
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio_mono = audio[:, 0]

    print(f"Recorded {len(audio_mono)} samples ({duration}s)")
    record_end = time.monotonic()

    # Transcribe via streaming interface
    queue: asyncio.Queue = asyncio.Queue()
    # Feed audio in 20ms chunks (like real-time streaming)
    chunk_size = int(sample_rate * 0.02)  # 320 samples = 20ms
    for i in range(0, len(audio_mono), chunk_size):
        await queue.put(audio_mono[i : i + chunk_size])
    await queue.put(None)  # sentinel

    first_partial_time = None
    final_text = ""
    final_time = None

    async for text, is_final in stt.stream(queue):
        now = time.monotonic()
        if first_partial_time is None:
            first_partial_time = now
        if is_final:
            final_text = text
            final_time = now
            break
        else:
            print(f"  [partial] {text}")

    # Results
    print(f"\n{'=' * 60}")
    print(f"Final transcript: \"{final_text}\"")
    print(f"{'=' * 60}")

    if first_partial_time:
        print(f"Time to first partial: {(first_partial_time - record_end)*1000:.0f}ms")
    if final_time:
        latency = (final_time - record_end) * 1000
        print(f"Time to final transcript: {latency:.0f}ms")
        target = 200 if engine == "Deepgram" else 600
        status = "PASS" if latency < target else "SLOW"
        print(f"Target: <{target}ms — [{status}]")
    else:
        print("No transcript produced!")

    print()

    # Also test the sync Whisper directly for comparison
    print("--- Sync Whisper baseline ---")
    t0 = time.monotonic()
    sync_text = stt._whisper.transcribe_pcm(audio_mono, sample_rate, "en")
    t1 = time.monotonic()
    print(f"Whisper sync: \"{sync_text}\" ({(t1-t0)*1000:.0f}ms)")


async def test_vad():
    """Quick VAD test — verify it loads and classifies silence vs noise."""
    print("\n--- VAD Test ---")
    try:
        from mother.audio.vad import SileroVAD

        vad = SileroVAD(threshold=0.5, min_silence_ms=500)
        # Test with silence
        silence = np.zeros(512, dtype=np.float32)
        is_speech = vad.is_speech(silence)
        print(f"Silence → is_speech={is_speech} (expected False)")

        # Test with white noise (should usually not be speech)
        noise = np.random.randn(512).astype(np.float32) * 0.5
        is_noise_speech = vad.is_speech(noise)
        print(f"White noise → is_speech={is_noise_speech}")

        print("VAD loaded OK")
    except Exception as e:
        print(f"VAD test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test STT pipeline")
    parser.add_argument(
        "--whisper-only", action="store_true", help="Force Whisper fallback"
    )
    parser.add_argument(
        "--skip-vad", action="store_true", help="Skip VAD test"
    )
    args = parser.parse_args()

    async def run():
        if not args.skip_vad:
            await test_vad()
        await test_streaming_stt(force_whisper=args.whisper_only)

    asyncio.run(run())


if __name__ == "__main__":
    main()
