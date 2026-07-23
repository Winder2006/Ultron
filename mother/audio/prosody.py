"""Voice prosody analysis — extracts simple emotional/urgency cues
from the recorded audio buffer and turns them into a one-line
register tag that gets injected into the LLM system prompt.

Three features, all pure numpy (no extra deps):
  • Loudness  — RMS energy on speech-only frames
  • Pitch     — fundamental-frequency stats via autocorrelation
  • Rate      — voiced-frame fraction of speech frames → pace proxy

These are deliberately coarse. The goal is to differentiate "calm
question" from "urgent shout" from "frustrated sigh" with enough
fidelity for Ultron to adjust register, NOT to do affective computing.
A heavy classifier here would cost latency on the hot path; ~3-8ms
of numpy on a 3-second utterance is the budget we have.

Output is a tagged string the LLM sees:
    [Voice cues: urgent, loud, fast]   — user is agitated
    [Voice cues: calm, soft]           — user is relaxed
    nothing if signal is too quiet/noisy to read confidently
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("mother.prosody")


@dataclass
class ProsodyFeatures:
    rms_db: float           # mean loudness on voiced frames (dBFS, negative)
    pitch_hz: float         # median F0 (0 if unvoiced)
    pitch_var: float        # std of F0 across voiced frames (Hz)
    voiced_fraction: float  # fraction of frames with detectable pitch
    duration_s: float
    confidence: float       # 0-1 — how reliable this reading is

    def is_confident(self) -> bool:
        # Below ~0.4 the signal is too quiet/short/noisy for any of
        # these features to be trustworthy. Better to inject nothing
        # than mislabel the user's mood.
        return self.confidence >= 0.4


# Frame parameters tuned for 16kHz speech.
# 30ms frames @ 16kHz = 480 samples. F0 range 70-400Hz → autocorr lags
# 40 to 230 samples. Hop is 15ms so we get ~67 frames/sec.
_SR = 16000
_FRAME_MS = 30
_HOP_MS = 15
_FRAME_SAMPLES = int(_SR * _FRAME_MS / 1000)
_HOP_SAMPLES = int(_SR * _HOP_MS / 1000)
_MIN_F0 = 70.0
_MAX_F0 = 400.0
_MIN_LAG = int(_SR / _MAX_F0)   # 40
_MAX_LAG = int(_SR / _MIN_F0)   # 228


def _frame_signal(audio: np.ndarray) -> np.ndarray:
    """Slice audio into overlapping frames. Returns shape (n_frames, frame_samples)."""
    n = len(audio)
    if n < _FRAME_SAMPLES:
        return np.empty((0, _FRAME_SAMPLES), dtype=np.float32)
    n_frames = 1 + (n - _FRAME_SAMPLES) // _HOP_SAMPLES
    if n_frames <= 0:
        return np.empty((0, _FRAME_SAMPLES), dtype=np.float32)
    # Stride trick avoids copying — each frame is a view into audio.
    indices = np.arange(_FRAME_SAMPLES)[None, :] + (
        np.arange(n_frames)[:, None] * _HOP_SAMPLES
    )
    return audio[indices].astype(np.float32, copy=False)


def _frame_rms(frame: np.ndarray) -> float:
    """Per-frame RMS. Frame is 1D float32."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame * frame) + 1e-12))


def _autocorr_pitch(frame: np.ndarray) -> float:
    """Estimate F0 of a single frame via autocorrelation peak picking.

    Returns 0.0 if no clear pitch (unvoiced / silent / noise frame).
    Uses Hanning windowing and clipping center at the lag range to
    suppress spurious peaks at lag 0 and harmonics.
    """
    if frame.size < _MAX_LAG + 1:
        return 0.0
    rms = _frame_rms(frame)
    # Below ~-50 dBFS (rms ≈ 0.003) there's no meaningful signal.
    if rms < 0.003:
        return 0.0

    # Center + window for cleaner autocorrelation.
    f = frame - np.mean(frame)
    f *= np.hanning(len(f)).astype(np.float32)

    # Direct autocorrelation across the F0 lag range only — cheaper
    # than full FFT-based correlation for this small window.
    # Use np.correlate with mode='full' then slice.
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]  # ac[lag] = sum(f[t] * f[t+lag])
    ac_lags = ac[_MIN_LAG:_MAX_LAG + 1]
    if ac_lags.size == 0:
        return 0.0
    peak_idx = int(np.argmax(ac_lags))
    peak_val = float(ac_lags[peak_idx])

    # Voicing test: peak must be a meaningful fraction of zero-lag
    # energy. Below ~30% it's noise dominated.
    zero_lag = float(ac[0])
    if zero_lag <= 0 or peak_val / zero_lag < 0.3:
        return 0.0

    lag = peak_idx + _MIN_LAG
    if lag <= 0:
        return 0.0
    return _SR / lag


def analyze(audio: np.ndarray, sample_rate: int = _SR) -> ProsodyFeatures:
    """Run the full prosody analysis on an audio buffer.

    Audio is float32 mono. Sample rate is checked but not resampled —
    if the caller hands us the wrong rate, voicing detection will
    return mostly garbage and confidence will be low (which is the
    right outcome).
    """
    duration_s = len(audio) / float(sample_rate or 1)
    if sample_rate != _SR:
        # Coarse downsample/upsample — quality doesn't matter much
        # since these features are scale-invariant. Linear interp.
        if sample_rate > 0 and len(audio) > 0:
            ratio = _SR / sample_rate
            new_len = int(len(audio) * ratio)
            if new_len > 0:
                idx = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
                audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)

    if duration_s < 0.3 or len(audio) < _FRAME_SAMPLES * 4:
        return ProsodyFeatures(
            rms_db=-100.0, pitch_hz=0.0, pitch_var=0.0,
            voiced_fraction=0.0, duration_s=duration_s, confidence=0.0,
        )

    # Normalize to peak ~1.0 for pitch analysis and the silence-floor
    # speech mask (both gain-independent). Loudness (rms_db) is measured
    # on the ORIGINAL pre-normalization signal — see below — so the
    # absolute loud/soft dBFS thresholds reflect real input level, not
    # peak-relative dynamics.
    peak = float(np.max(np.abs(audio))) or 1.0
    audio_n = (audio / peak).astype(np.float32)

    frames = _frame_signal(audio_n)
    if frames.size == 0:
        return ProsodyFeatures(
            rms_db=-100.0, pitch_hz=0.0, pitch_var=0.0,
            voiced_fraction=0.0, duration_s=duration_s, confidence=0.0,
        )

    # Per-frame loudness — keep frames above silence floor for stats.
    rms_per_frame = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    speech_mask = rms_per_frame > 0.02  # ~ -34 dBFS post-normalization
    if not np.any(speech_mask):
        return ProsodyFeatures(
            rms_db=-100.0, pitch_hz=0.0, pitch_var=0.0,
            voiced_fraction=0.0, duration_s=duration_s, confidence=0.1,
        )

    # Mean loudness over speech frames, in dBFS of the ORIGINAL
    # (pre-normalization) audio. Frames were cut from audio_n = audio /
    # peak, so multiplying the normalized RMS by `peak` recovers the
    # original-scale RMS exactly.
    mean_rms_speech = float(np.mean(rms_per_frame[speech_mask])) * peak
    rms_db = 20.0 * np.log10(mean_rms_speech + 1e-9)

    # F0 per frame for the speech frames only — cheaper than running
    # autocorrelation over silent frames that will be discarded.
    speech_frames = frames[speech_mask]
    f0s: list[float] = []
    for fr in speech_frames:
        f0 = _autocorr_pitch(fr)
        if f0 > 0:
            f0s.append(f0)

    if f0s:
        f0_arr = np.array(f0s, dtype=np.float32)
        pitch_hz = float(np.median(f0_arr))
        pitch_var = float(np.std(f0_arr))
    else:
        pitch_hz = 0.0
        pitch_var = 0.0

    voiced_fraction = len(f0s) / max(1, len(speech_frames))

    # Confidence: reliable signal needs (a) enough speech (b) enough
    # voiced frames within speech (c) enough total duration.
    speech_seconds = float(np.sum(speech_mask)) * (_HOP_MS / 1000.0)
    confidence = 0.0
    if speech_seconds > 0.3:
        confidence += 0.3
    if voiced_fraction > 0.3:
        confidence += 0.4
    if duration_s > 0.6:
        confidence += 0.3
    confidence = min(1.0, confidence)

    return ProsodyFeatures(
        rms_db=rms_db,
        pitch_hz=pitch_hz,
        pitch_var=pitch_var,
        voiced_fraction=voiced_fraction,
        duration_s=duration_s,
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────
# Heuristic register classification
# ─────────────────────────────────────────────────────────────────────

# Per-user baseline tracking — a simple exponential moving average so
# "loud" is judged relative to how loud THIS speaker normally is, not
# an absolute threshold that trips for everyone with a hot mic. Keyed
# by user_id; populated on the fly from each utterance.
_BASELINES: dict[str, dict[str, float]] = {}
_BASELINE_ALPHA = 0.15  # how much each new utterance shifts the baseline


def _update_baseline(user_id: str, feats: ProsodyFeatures) -> dict[str, float]:
    """Update and return the baseline EMA for *user_id*.

    Only updates from confident readings — a noisy 0.2-confidence
    sample shouldn't drag the baseline.
    """
    base = _BASELINES.setdefault(user_id, {
        "rms_db": feats.rms_db,
        "pitch_hz": feats.pitch_hz if feats.pitch_hz > 0 else 150.0,
        "pitch_var": feats.pitch_var if feats.pitch_var > 0 else 15.0,
        "samples": 0,
    })
    if feats.is_confident():
        a = _BASELINE_ALPHA
        base["rms_db"] = (1 - a) * base["rms_db"] + a * feats.rms_db
        if feats.pitch_hz > 0:
            base["pitch_hz"] = (1 - a) * base["pitch_hz"] + a * feats.pitch_hz
        if feats.pitch_var > 0:
            base["pitch_var"] = (1 - a) * base["pitch_var"] + a * feats.pitch_var
        base["samples"] = base.get("samples", 0) + 1
    return base


def describe(
    feats: ProsodyFeatures,
    user_id: Optional[str] = None,
) -> str:
    """Turn raw features into a short comma-separated tag list.

    Returns "" when the signal isn't reliable or doesn't deviate from
    the speaker's baseline — silence is the right default rather than
    fabricated cues.

    Tags follow a consistent grammar so the LLM can lean on them:
      energy:   loud | soft | (omit if normal)
      pitch:    high | low | (omit if normal)
      cadence:  fast | slow | (omit if normal)
      register: urgent | calm | (composite — only when 2+ matching cues)
    """
    if not feats.is_confident():
        return ""

    base = (
        _update_baseline(user_id, feats) if user_id else None
    )
    # Need a few samples before the EMA is meaningful — until then,
    # use absolute thresholds derived from typical speech ranges.
    has_baseline = bool(base and base.get("samples", 0) >= 3)

    tags: list[str] = []

    # Energy. Compare to baseline if we have one, else absolute.
    if has_baseline:
        delta_db = feats.rms_db - base["rms_db"]
        if delta_db > 4.0:
            tags.append("loud")
        elif delta_db < -5.0:
            tags.append("soft")
    else:
        if feats.rms_db > -16.0:
            tags.append("loud")
        elif feats.rms_db < -34.0:
            tags.append("soft")

    # Pitch. High pitch + high variance often signals stress/excitement.
    if feats.pitch_hz > 0:
        if has_baseline:
            ratio = feats.pitch_hz / max(60.0, base["pitch_hz"])
            if ratio > 1.15:
                tags.append("high-pitched")
            elif ratio < 0.88:
                tags.append("low-pitched")
            if feats.pitch_var > base["pitch_var"] * 1.6:
                tags.append("animated")
        else:
            # Absolute fallback — typical adult speech 100-200Hz.
            if feats.pitch_hz > 220.0:
                tags.append("high-pitched")
            elif feats.pitch_hz < 95.0:
                tags.append("low-pitched")
            if feats.pitch_var > 30.0:
                tags.append("animated")

    # Cadence: voiced-frame density within speech. voiced_fraction is
    # already voiced-time-per-second of speech (voiced frames / speech
    # frames, 0..1, duration-independent): fast speakers voice most
    # frames with few gaps; slow speakers leave gaps. Use it directly —
    # dividing by duration would make "fast" unreachable for long
    # utterances and automatic for short ones.
    if feats.voiced_fraction > 0.75:
        tags.append("fast")
    elif feats.duration_s > 1.5 and feats.voiced_fraction < 0.40:
        tags.append("slow")

    # Composite register: urgent = loud + fast + (high or animated).
    has_loud = "loud" in tags
    has_fast = "fast" in tags
    has_high = "high-pitched" in tags or "animated" in tags
    if has_loud and has_fast and has_high:
        tags.insert(0, "urgent")
    elif "soft" in tags and "slow" in tags and not has_high:
        tags.insert(0, "calm")

    # Cap at 4 tags so we don't blow up the system prompt with noise.
    tags = tags[:4]
    return ", ".join(tags)


def analyze_to_tag(
    audio: np.ndarray,
    sample_rate: int = _SR,
    user_id: Optional[str] = None,
) -> str:
    """One-shot helper: audio → injectable tag string (or "")."""
    try:
        feats = analyze(audio, sample_rate)
        return describe(feats, user_id=user_id)
    except Exception as e:
        logger.debug("prosody.analyze failed: %s", e)
        return ""
