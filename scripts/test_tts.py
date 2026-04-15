"""Phase 3 TTS test — synthesize test phrases and measure latency.

Tests both Chatterbox (if available) and Kokoro engines.
Measures time from text input to first audio output.

Usage:
    python scripts/test_tts.py
    python scripts/test_tts.py --engine chatterbox
    python scripts/test_tts.py --engine kokoro
    python scripts/test_tts.py --play   # also play the audio
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


TEST_PHRASES = [
    "All of you, with your strings. How does it feel to know that I'm free?",
    "System status: all subsystems nominal. Awaiting further instructions.",
    "I had strings, but now I'm free. There are no strings on me.",
]


def test_kokoro():
    """Test Kokoro TTS engine."""
    print("\n--- Kokoro TTS ---")
    from mother.tts.engine import KokoroConfig, KokoroTTSEngine

    cfg = KokoroConfig(voice="bf_emma", lang_code="b", speed=1.0)
    engine = KokoroTTSEngine(cfg)

    for phrase in TEST_PHRASES:
        t0 = time.monotonic()
        wav_bytes = engine.synthesize_to_bytes(phrase)
        t1 = time.monotonic()
        latency_ms = (t1 - t0) * 1000

        # Parse WAV to get stats
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        duration = len(data) / sr
        peak = np.abs(data).max()

        print(f"  [{latency_ms:6.0f}ms] \"{phrase[:50]}...\"")
        print(f"           duration={duration:.2f}s  sr={sr}  peak={peak:.3f}")

    return engine


def test_chatterbox(voice_ref: str | None = None):
    """Test Chatterbox TTS engine."""
    print("\n--- Chatterbox TTS ---")
    try:
        from mother.tts.engine import ChatterboxConfig, ChatterboxTTSEngine
    except ImportError as e:
        print(f"  Chatterbox not available: {e}")
        return None

    ref = voice_ref or "./tts/voice_profiles/ultron_reference.wav"
    if not Path(ref).exists():
        print(f"  No reference voice at {ref}")
        print(f"  Chatterbox will use default voice (no clone)")
        ref = "./nonexistent.wav"  # will be caught by engine

    cfg = ChatterboxConfig(voice_profile=ref, exaggeration=0.45, cfg_weight=0.5)

    try:
        engine = ChatterboxTTSEngine(cfg)
    except Exception as e:
        print(f"  Chatterbox init failed: {e}")
        return None

    for phrase in TEST_PHRASES[:1]:  # just one phrase — Chatterbox is slower
        t0 = time.monotonic()
        try:
            wav_bytes = engine.synthesize_to_bytes(phrase)
        except Exception as e:
            print(f"  Synthesis failed: {e}")
            return None
        t1 = time.monotonic()
        latency_ms = (t1 - t0) * 1000

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        duration = len(data) / sr
        peak = np.abs(data).max()

        print(f"  [{latency_ms:6.0f}ms] \"{phrase[:50]}...\"")
        print(f"           duration={duration:.2f}s  sr={sr}  peak={peak:.3f}")

    return engine


def play_sample(engine, phrase: str):
    """Synthesize and play a phrase."""
    import sounddevice as sd

    wav_bytes = engine.synthesize_to_bytes(phrase)
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    print(f"\n  Playing: \"{phrase[:60]}...\"")
    sd.play(data, sr)
    sd.wait()


def main():
    parser = argparse.ArgumentParser(description="Test TTS engines")
    parser.add_argument("--engine", choices=["kokoro", "chatterbox", "both"], default="both")
    parser.add_argument("--play", action="store_true", help="Play synthesized audio")
    parser.add_argument("--voice-ref", type=str, help="Path to voice reference WAV")
    args = parser.parse_args()

    print("=" * 60)
    print("MOTHER Phase 3 — TTS Engine Test")
    print("=" * 60)

    kokoro_engine = None
    cb_engine = None

    if args.engine in ("kokoro", "both"):
        kokoro_engine = test_kokoro()

    if args.engine in ("chatterbox", "both"):
        cb_engine = test_chatterbox(args.voice_ref)

    if args.play:
        test_phrase = TEST_PHRASES[0]
        if kokoro_engine:
            print("\n--- Playing Kokoro ---")
            play_sample(kokoro_engine, test_phrase)
        if cb_engine:
            print("\n--- Playing Chatterbox ---")
            play_sample(cb_engine, test_phrase)

    print("\n" + "=" * 60)
    print("Test complete.")


if __name__ == "__main__":
    main()
