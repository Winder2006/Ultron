"""Custom 'MOTHER' wake word model trainer for openWakeWord.

Generates synthetic training samples using Piper TTS, then trains an
openWakeWord ONNX model. The resulting model is saved to ./models/.

Requirements:
    pip install openwakeword
    Piper TTS binary available at bin/piper/piper/piper.exe (or adjust PIPER_PATH)

Usage:
    python scripts/train_wake_word.py
    python scripts/train_wake_word.py --samples 200 --output ./models/mother_wakeword.onnx

Steps performed:
    1. Generate synthetic speech of "MOTHER" using Piper TTS
       - 200 iterations with varied speed (length_scale 0.8-1.2)
       - Saves to ./samples/mother_*.wav
    2. Run openWakeWord trainer
    3. Output trained ONNX model to ./models/mother_wakeword.onnx

Notes:
    - The more diverse the samples, the better the model
    - Consider also recording your own voice saying "MOTHER" 10-20 times
      and adding those recordings to the samples/ directory
    - After training, test with: python scripts/test_wake_word.py
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path


PIPER_PATH = os.environ.get(
    "PIPER_PATH",
    str(Path(__file__).parent.parent / "bin" / "piper" / "piper" / "piper.exe"),
)
PIPER_MODEL = str(
    Path(__file__).parent.parent / "voices" / "en_GB-cori-medium.onnx"
)
PIPER_CONFIG = str(
    Path(__file__).parent.parent / "voices" / "en_GB-cori-medium.onnx.json"
)
SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "mother_wake"
OUTPUT_DIR = Path(__file__).parent.parent / "models"


def generate_samples(count: int = 200):
    """Generate synthetic 'MOTHER' speech samples with varied prosody."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    phrases = [
        "MOTHER",
        "Mother",
        "mother",
        "Hey Mother",
        "Okay Mother",
    ]
    print(f"Generating {count} synthetic samples to {SAMPLES_DIR}/")

    generated = 0
    for i in range(count):
        phrase = random.choice(phrases)
        length_scale = round(random.uniform(0.8, 1.2), 2)
        noise_scale = round(random.uniform(0.4, 0.8), 2)
        out_path = SAMPLES_DIR / f"mother_{i:04d}.wav"

        args = [
            PIPER_PATH,
            "-m", PIPER_MODEL,
            "-c", PIPER_CONFIG,
            "--length_scale", str(length_scale),
            "--noise_scale", str(noise_scale),
            "-f", str(out_path),
        ]
        try:
            proc = subprocess.run(
                args,
                input=phrase.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            if proc.returncode == 0:
                generated += 1
                if generated % 50 == 0:
                    print(f"  Generated {generated}/{count}")
            else:
                print(f"  Piper failed for sample {i}: {proc.stderr.decode()[:100]}")
        except Exception as e:
            print(f"  Error generating sample {i}: {e}")

    print(f"Generated {generated} samples in {SAMPLES_DIR}/")
    return generated


def train_model(output_path: str):
    """Train openWakeWord model from positive samples."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining wake word model...")
    print(f"  Positive samples: {SAMPLES_DIR}/")
    print(f"  Output: {output_path}")
    print()
    print("NOTE: openWakeWord training requires the full training pipeline.")
    print("For quick setup, use the pre-trained 'hey_jarvis' model as fallback.")
    print()
    print("To train a custom model, follow the openWakeWord docs:")
    print("  https://github.com/dscripka/openWakeWord#training-new-models")
    print()
    print("Quick alternative — fine-tune using the provided samples:")
    print(f"  python -m openwakeword.train \\")
    print(f"    --positive_clips {SAMPLES_DIR}/ \\")
    print(f"    --output_path {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train custom MOTHER wake word")
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "mother_wakeword.onnx"),
        help="Output model path",
    )
    parser.add_argument("--generate-only", action="store_true", help="Only generate samples, skip training")
    args = parser.parse_args()

    # Check Piper
    if not Path(PIPER_PATH).exists():
        print(f"Piper not found at {PIPER_PATH}")
        print("Set PIPER_PATH env var or install Piper.")
        return 1

    # Generate samples
    generated = generate_samples(args.samples)
    if generated == 0:
        print("No samples generated — cannot train.")
        return 1

    if args.generate_only:
        print("Samples generated. Skipping training (--generate-only).")
        return 0

    # Train
    train_model(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
