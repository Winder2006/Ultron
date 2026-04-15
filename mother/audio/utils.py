"""Audio utility functions for MOTHER.

Provides common audio operations used across the application.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from mother.core.logging_config import get_logger

logger = get_logger("audio_utils")

__all__ = [
    "play_audio_file",
    "play_audio_bytes",
    "apply_fade",
    "synthesize_and_play",
]


def apply_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_seconds: float = 0.005
) -> np.ndarray:
    """Apply fade in/out to audio to reduce clicks.
    
    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate
        fade_seconds: Fade duration in seconds
        
    Returns:
        Audio with fades applied
    """
    n = max(1, int(sample_rate * fade_seconds))
    
    if audio.ndim == 1:
        n = min(n, max(1, audio.shape[0] // 4))
        ramp = np.linspace(0.0, 1.0, n, dtype=audio.dtype)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    else:
        n = min(n, max(1, audio.shape[0] // 4))
        ramp = np.linspace(0.0, 1.0, n, dtype=audio.dtype)[:, None]
        audio[:n, :] *= ramp
        audio[-n:, :] *= ramp[::-1]
    
    # Clamp to valid range
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio


def play_audio_file(
    path: str,
    apply_fades: bool = True,
    blocking: bool = True
) -> bool:
    """Play an audio file.
    
    Args:
        path: Path to audio file
        apply_fades: Whether to apply fade in/out
        blocking: Whether to wait for playback to complete
        
    Returns:
        True if playback started successfully
    """
    try:
        import soundfile as sf
        import sounddevice as sd
        
        data, sr = sf.read(path, dtype="float32")
        
        if apply_fades:
            data = apply_fade(data, sr)
        
        sd.play(data, sr)
        if blocking:
            sd.wait()
        
        return True
    except ImportError:
        logger.error("soundfile/sounddevice required for audio playback")
        return False
    except Exception as e:
        logger.error(f"Audio playback error: {e}")
        return False


def play_audio_bytes(
    audio_data: bytes,
    sample_rate: int = 22050,
    apply_fades: bool = True,
    blocking: bool = True
) -> bool:
    """Play audio from bytes.
    
    Args:
        audio_data: Raw audio bytes
        sample_rate: Sample rate
        apply_fades: Whether to apply fade in/out
        blocking: Whether to wait for playback to complete
        
    Returns:
        True if playback started successfully
    """
    try:
        import sounddevice as sd
        
        audio = np.frombuffer(audio_data, dtype=np.float32)
        
        if apply_fades:
            audio = apply_fade(audio, sample_rate)
        
        sd.play(audio, sample_rate)
        if blocking:
            sd.wait()
        
        return True
    except ImportError:
        logger.error("sounddevice required for audio playback")
        return False
    except Exception as e:
        logger.error(f"Audio playback error: {e}")
        return False


def synthesize_and_play(
    text: str,
    tts_engine,
    apply_fades: bool = True,
    blocking: bool = True
) -> Tuple[bool, Optional[str]]:
    """Synthesize text and play it.
    
    Args:
        text: Text to speak
        tts_engine: TTS engine instance with synthesize_to_file method
        apply_fades: Whether to apply fade in/out
        blocking: Whether to wait for playback to complete
        
    Returns:
        (success, error_message)
    """
    if not text or not text.strip():
        return True, None
    
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    
    try:
        # Synthesize
        tts_engine.synthesize_to_file(text.strip(), tmp_path)
        
        # Play
        success = play_audio_file(tmp_path, apply_fades=apply_fades, blocking=blocking)
        
        if not success:
            return False, "Playback failed"
        
        return True, None
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False, str(e)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

