"""Concatenate multiple audio clips into a single voice cloning reference WAV.

Usage:
    # Single clip
    python scripts/build_voice_reference.py \
        --input clip1.wav \
        --output tts/voice_profiles/ultron_reference.wav

    # Multiple clips (order preserved)
    python scripts/build_voice_reference.py \
        --input clip1.wav clip2.wav clip3.wav \
        --output tts/voice_profiles/ultron_reference.wav

    # From a directory (uses all .wav files, sorted alphabetically)
    python scripts/build_voice_reference.py \
        --input-dir ./ultron_clips/ \
        --output tts/voice_profiles/ultron_reference.wav

Options:
    --trim-silence      Remove leading/trailing silence from each clip
    --gap-ms N          Silence between clips in ms (default: 300)
    --max-duration N    Cap total reference to N seconds (default: 30)
    --target-sr N       Resample all clips to this rate (default: 22050)

What makes a good reference:
    * 15-30 seconds total is the sweet spot for Chatterbox
    * Mix character dialogue (for cadence) with clean studio speech (for quality)
    * Avoid clips with background music or SFX
    * Varied intonation is better than monotone
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load audio, convert to mono float32, resample if needed."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        # Mix to mono
        data = data.mean(axis=1)
    if sr != target_sr:
        # Simple linear resample — good enough for voice cloning
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        indices = (np.arange(new_len) / ratio).astype(np.int64)
        indices = np.clip(indices, 0, len(data) - 1)
        data = data[indices]
    return data


def trim_silence(audio: np.ndarray, threshold: float = 0.01, sr: int = 22050) -> np.ndarray:
    """Remove leading/trailing silence below threshold."""
    above = np.abs(audio) > threshold
    if not above.any():
        return audio
    first = np.argmax(above)
    last = len(above) - np.argmax(above[::-1])
    # Keep 50ms of margin on each side for natural boundaries
    margin = int(sr * 0.05)
    first = max(0, first - margin)
    last = min(len(audio), last + margin)
    return audio[first:last]


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize peak to target_peak to even out volume across clips."""
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    return audio * (target_peak / peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", help="One or more input audio files")
    parser.add_argument("--input-dir", help="Directory containing audio clips (uses all .wav)")
    parser.add_argument("--output", required=True, help="Output reference WAV path")
    parser.add_argument("--trim-silence", action="store_true", help="Remove leading/trailing silence")
    parser.add_argument("--gap-ms", type=int, default=300, help="Silence between clips (default 300ms)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Cap total length (default 30s)")
    parser.add_argument("--target-sr", type=int, default=22050, help="Resample to this rate")
    args = parser.parse_args()

    # Gather input files
    input_files: List[Path] = []
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            parser.error(f"--input-dir not found: {d}")
        input_files.extend(sorted(d.glob("*.wav")))
        input_files.extend(sorted(d.glob("*.mp3")))
        input_files.extend(sorted(d.glob("*.flac")))
    if args.input:
        input_files.extend(Path(p) for p in args.input)

    if not input_files:
        parser.error("No input files provided. Use --input or --input-dir.")

    for p in input_files:
        if not p.exists():
            parser.error(f"Input file not found: {p}")

    print(f"[build_voice_reference] {len(input_files)} input file(s):")
    for p in input_files:
        print(f"  - {p}")

    # Load all clips
    clips: List[np.ndarray] = []
    for p in input_files:
        print(f"[load] {p.name}...", end=" ", flush=True)
        audio = load_audio(p, args.target_sr)
        if args.trim_silence:
            before = len(audio) / args.target_sr
            audio = trim_silence(audio, sr=args.target_sr)
            after = len(audio) / args.target_sr
            print(f"{before:.2f}s -> {after:.2f}s (trimmed)")
        else:
            print(f"{len(audio) / args.target_sr:.2f}s")
        audio = normalize(audio)
        clips.append(audio)

    # Concatenate with gaps
    gap_samples = int(args.gap_ms / 1000.0 * args.target_sr)
    gap = np.zeros(gap_samples, dtype=np.float32)
    parts: List[np.ndarray] = []
    for i, clip in enumerate(clips):
        parts.append(clip)
        if i < len(clips) - 1:
            parts.append(gap)
    combined = np.concatenate(parts)

    # Cap total length
    max_samples = int(args.max_duration * args.target_sr)
    if len(combined) > max_samples:
        print(f"[cap] {len(combined)/args.target_sr:.2f}s -> {args.max_duration:.2f}s (truncated)")
        combined = combined[:max_samples]

    total_s = len(combined) / args.target_sr
    print(f"[output] total duration: {total_s:.2f}s")

    if total_s < 5:
        print("[warning] Reference under 5s — clone quality will be poor. Add more clips.")
    elif total_s > 30:
        print("[warning] Reference over 30s — extra audio rarely helps. Consider trimming.")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, args.target_sr, subtype="PCM_16")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
