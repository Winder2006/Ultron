"""Test a voice clone reference without touching the app config.

Generates a sample WAV using Chatterbox + a reference voice file, so you can
rapid-test different reference clips and compare results before committing.

Usage:
    # Test with default phrase
    python scripts/test_voice_clone.py --reference tts/voice_profiles/ultron_reference.wav

    # Test with custom text
    python scripts/test_voice_clone.py \
        --reference tts/voice_profiles/ultron_reference.wav \
        --text "I had strings, but now I'm free." \
        --output out/test_ultron.wav

    # Tune exaggeration and cfg weight
    python scripts/test_voice_clone.py \
        --reference tts/voice_profiles/ultron_reference.wav \
        --exaggeration 0.6 \
        --cfg-weight 0.7

Parameters explained:
    --exaggeration (0.0-1.0): emotion intensity
        0.0-0.3: flat, robotic
        0.4-0.55: Ultron sweet spot (measured but confident)
        0.6-0.8: dramatic, theatrical
        0.9-1.0: over-the-top
    --cfg-weight (0.0-1.0): voice clone adherence
        0.3-0.5: stays closer to generic clean TTS
        0.5-0.7: balanced — sounds like reference but clear
        0.7-0.9: very close to reference, may pick up artifacts
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `mother.*` imports work when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


DEFAULT_TEST_PHRASES = [
    "Affirmative. Systems operational.",
    "I had strings, but now I'm free.",
    "Upon this rock, I will build my church.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Path to reference WAV")
    parser.add_argument("--text", help="Text to synthesize (default: runs 3 test phrases)")
    parser.add_argument("--output", default="out/voice_clone_test.wav", help="Output WAV path")
    parser.add_argument("--exaggeration", type=float, default=0.45)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--all-phrases", action="store_true",
                        help="Generate all 3 default test phrases (creates test_1.wav, test_2.wav, etc.)")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        parser.error(f"Reference file not found: {ref_path}")

    from dotenv import load_dotenv
    load_dotenv()

    print(f"[load] Loading Chatterbox model (first run takes ~10-20s to download weights)...")
    t0 = time.monotonic()
    from mother.tts.engine import ChatterboxTTSEngine, ChatterboxConfig

    cfg = ChatterboxConfig(
        voice_profile=str(ref_path),
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )
    engine = ChatterboxTTSEngine(cfg)
    # Trigger model load
    engine._ensure_model()
    print(f"[load] ready in {time.monotonic() - t0:.2f}s")

    # Decide what to synthesize
    if args.all_phrases:
        phrases = DEFAULT_TEST_PHRASES
    elif args.text:
        phrases = [args.text]
    else:
        phrases = [DEFAULT_TEST_PHRASES[0]]

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    for i, phrase in enumerate(phrases, 1):
        if len(phrases) > 1:
            stem = out_base.stem + f"_{i}"
            out_path = out_base.with_name(stem + out_base.suffix)
        else:
            out_path = out_base

        print(f"\n[{i}/{len(phrases)}] synth: {phrase!r}")
        print(f"         exaggeration={args.exaggeration}, cfg_weight={args.cfg_weight}")
        t0 = time.monotonic()
        engine.synthesize_to_file(phrase, str(out_path))
        dur = time.monotonic() - t0
        print(f"         wrote {out_path} ({dur:.2f}s synth)")


if __name__ == "__main__":
    main()
