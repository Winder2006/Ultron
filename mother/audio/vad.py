"""Silero VAD for voice activity detection.

Lightweight VAD — <1ms per 30ms chunk on CPU.
Used for offline endpoint detection when Deepgram is unavailable,
and for reducing false positives in wake word detection.
"""
from __future__ import annotations

import numpy as np


class SileroVAD:
    """Silero VAD wrapper using torch.hub."""

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 500):
        import torch
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self._silence_frames = 0
        self._sample_rate = 16000
        self._frame_size = 512  # 32ms at 16kHz
        self._torch = torch

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Returns True if the chunk contains speech.

        Args:
            audio_chunk: float32 numpy array, 512 samples (32ms at 16kHz).
        """
        tensor = self._torch.from_numpy(audio_chunk).float()
        if tensor.dim() == 1:
            # Ensure correct length for Silero (512 samples at 16kHz)
            if len(tensor) < self._frame_size:
                tensor = self._torch.nn.functional.pad(
                    tensor, (0, self._frame_size - len(tensor))
                )
        confidence = self.model(tensor, self._sample_rate).item()
        return confidence >= self.threshold

    def is_end_of_utterance(self, audio_chunk: np.ndarray) -> bool:
        """Returns True when silence duration exceeds min_silence_ms.

        Call this with each audio frame. Returns True once enough
        consecutive silent frames have passed.
        """
        if not self.is_speech(audio_chunk):
            self._silence_frames += 1
        else:
            self._silence_frames = 0
        frame_duration_ms = (self._frame_size / self._sample_rate) * 1000
        frames_needed = self.min_silence_ms / frame_duration_ms
        return self._silence_frames >= frames_needed

    def reset(self):
        """Reset silence counter for a new utterance."""
        self._silence_frames = 0
        self.model.reset_states()
