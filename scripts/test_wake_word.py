"""Phase 2 wake word test — shows detection scores in real time.

Listens on mic and prints openWakeWord scores every ~80ms.
Say the wake word and watch the score spike above the threshold.

Usage:
    python scripts/test_wake_word.py
    python scripts/test_wake_word.py --sensitivity 0.3
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Test wake word detection")
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--duration", type=int, default=30, help="Seconds to listen")
    args = parser.parse_args()

    from openwakeword.model import Model
    import sounddevice as sd

    # Try custom model first, then fallback
    from pathlib import Path
    custom = Path("./models/mother_wakeword.onnx")
    if custom.exists():
        print(f"Using custom model: {custom}")
        model = Model(wakeword_models=[str(custom)], inference_framework="onnx")
    else:
        print("Custom model not found — using 'hey_jarvis_v0.1' fallback")
        model = Model(wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx")

    keywords = list(model.models.keys())
    print(f"Models loaded: {keywords}")
    print(f"Sensitivity threshold: {args.sensitivity}")
    print(f"Listening for {args.duration}s — say the wake word!\n")
    print(f"{'Time':>6}  {'Keyword':<25}  {'Score':>6}  {'Status'}")
    print("-" * 55)

    sample_rate = 16000
    chunk_size = 1280  # 80ms

    detections = 0
    start = time.monotonic()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=chunk_size) as stream:
        while time.monotonic() - start < args.duration:
            audio, overflowed = stream.read(chunk_size)
            if overflowed:
                continue
            chunk = audio.flatten()
            prediction = model.predict(chunk)

            elapsed = time.monotonic() - start
            for kw, score in prediction.items():
                if score > 0.01:  # only print non-zero scores
                    status = " <<< DETECTED!" if score >= args.sensitivity else ""
                    print(f"{elapsed:6.1f}s  {kw:<25}  {score:6.3f}  {status}")
                    if score >= args.sensitivity:
                        detections += 1
                        model.reset()
                        time.sleep(0.5)  # cooldown

    print(f"\n{'=' * 55}")
    print(f"Total detections: {detections}")
    print(f"Duration: {args.duration}s")


if __name__ == "__main__":
    main()
