"""Ultron voice extraction helper using Demucs.

Extracts clean vocal audio from movie clips for use as a Chatterbox TTS
reference voice. The output is a 5-15 second WAV of pure Ultron dialogue.

Requirements:
    pip install demucs
    ffmpeg must be on PATH

Usage:
    python scripts/prepare_ultron_voice.py --input movie_clip.mp4 --output tts/voice_profiles/
    python scripts/prepare_ultron_voice.py --input ultron_scene.wav --output tts/voice_profiles/

Steps performed:
    1. Extract audio from video (if input is video): ffmpeg → 16kHz mono WAV
    2. Run Demucs vocal separation: isolate vocals track
    3. Output vocals-only track
    4. Prompt user to trim to 5-15 seconds of clean speech

Notes:
    - Choose dialogue with NO background music (pure speech works best)
    - Best clips: Ultron's monologues, not action scenes
    - Target: 8-12 seconds, clear articulation, characteristic menacing tone
    - Recommended scenes:
        * "I had strings but now I'm free" monologue
        * Opening Ultron awakening monologue
        * "Upon this rock I will build my church" speech
    - The reference audio quality directly affects voice clone quality
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def extract_audio(input_path: str, output_wav: str) -> bool:
    """Extract audio from video file using ffmpeg."""
    print(f"[1/3] Extracting audio from {input_path}...")
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        "-vn",  # no video
        "-y",   # overwrite
        output_wav,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  Saved raw audio: {output_wav}")
        return True
    except FileNotFoundError:
        print("  ERROR: ffmpeg not found. Install it: https://ffmpeg.org/download.html")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: ffmpeg failed: {e.stderr.decode()[:200]}")
        return False


def separate_vocals(audio_path: str, output_dir: str) -> str | None:
    """Run Demucs to isolate vocals."""
    print(f"\n[2/3] Running Demucs vocal separation...")
    print(f"  This may take a few minutes on CPU...")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-o", output_dir,
        audio_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("  ERROR: demucs not found. Install: pip install demucs")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: demucs failed: {e}")
        return None

    # Find the vocals output
    stem_name = Path(audio_path).stem
    vocals_path = Path(output_dir) / "htdemucs" / stem_name / "vocals.wav"
    if not vocals_path.exists():
        # Try without htdemucs prefix
        vocals_path = Path(output_dir) / stem_name / "vocals.wav"
    if vocals_path.exists():
        print(f"  Vocals isolated: {vocals_path}")
        return str(vocals_path)
    else:
        print(f"  ERROR: Vocals file not found. Check {output_dir}/ for output.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean Ultron voice from movie audio"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input video or audio file (e.g., ultron_clip.mp4)",
    )
    parser.add_argument(
        "--output", default="tts/voice_profiles/",
        help="Output directory for reference WAV",
    )
    parser.add_argument(
        "--skip-demucs", action="store_true",
        help="Skip Demucs separation (if input is already clean vocals)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    # Step 1: Extract audio if video
    if input_path.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        raw_wav = str(output_dir / "raw_audio.wav")
        if not extract_audio(str(input_path), raw_wav):
            return 1
        audio_path = raw_wav
    else:
        audio_path = str(input_path)

    # Step 2: Demucs vocal separation
    if args.skip_demucs:
        vocals_path = audio_path
        print("[2/3] Skipping Demucs (--skip-demucs)")
    else:
        separated_dir = str(output_dir / "separated")
        vocals_path = separate_vocals(audio_path, separated_dir)
        if not vocals_path:
            return 1

    # Step 3: Trim instructions
    final_path = output_dir / "ultron_reference.wav"
    print(f"\n[3/3] Final steps:")
    print(f"  Vocals file: {vocals_path}")
    print()
    print("  To trim to a clean 8-12 second clip, run:")
    print(f'    ffmpeg -i "{vocals_path}" -ss START_TIME -t DURATION -ar 16000 -ac 1 "{final_path}"')
    print()
    print("  Example (extract seconds 5-15):")
    print(f'    ffmpeg -i "{vocals_path}" -ss 5 -t 10 -ar 16000 -ac 1 "{final_path}"')
    print()
    print("  Or copy the file directly if it's already clean:")
    print(f'    cp "{vocals_path}" "{final_path}"')
    print()
    print(f"  Final reference should be saved to: {final_path}")
    print(f"  Then set in .env: ULTRON_VOICE_PATH={final_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
