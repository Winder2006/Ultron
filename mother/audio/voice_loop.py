"""Voice loop utilities for MOTHER CLI."""
from __future__ import annotations

import sys
import time
import wave
import tempfile
from pathlib import Path
from typing import Generator, Tuple, Optional, Callable, Any
import numpy as np

from mother.core.logging_config import get_logger

logger = get_logger("voice_loop")

__all__ = [
    "drain_keys",
    "key_pressed",
    "prompt_and_continue",
    "save_audio_to_wav",
    "audio_energy",
    "detect_silence",
    "AudioRecorder",
    "audio_chunks",
    "RATE",
    "CHANNELS",
    "DTYPE",
    "FRAME_MS",
    "FRAME_SAMPLES",
]

# Audio constants
RATE = 16000
CHANNELS = 1
DTYPE = np.int16
FRAME_MS = 30
FRAME_SAMPLES = int(RATE * FRAME_MS / 1000)


def drain_keys():
    """Drain any queued keyboard input (Windows/Unix compatible)."""
    if sys.platform == "win32":
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    else:
        import select
        import termios
        import tty
        try:
            old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        except Exception:
            pass


def key_pressed() -> bool:
    """Check if a key was pressed (non-blocking)."""
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.kbhit()
    else:
        import select
        return bool(select.select([sys.stdin], [], [], 0)[0])


def prompt_and_continue():
    """Show listening prompt and drain keys."""
    drain_keys()
    print("Listening... (press Enter to stop)")


def save_audio_to_wav(audio: np.ndarray, rate: int = RATE) -> Path:
    """Save audio to a temporary WAV file.
    
    Args:
        audio: Audio samples as numpy array
        rate: Sample rate
        
    Returns:
        Path to the temporary WAV file
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(rate)
            wf.writeframes(audio.tobytes())
    finally:
        import os
        os.close(fd)
    return Path(path)


def audio_energy(audio: np.ndarray) -> float:
    """Calculate RMS energy of audio.
    
    Args:
        audio: Audio samples
        
    Returns:
        RMS energy value
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def detect_silence(
    audio: np.ndarray,
    threshold: float = 500.0,
    min_silence_frames: int = 10
) -> bool:
    """Detect if audio is mostly silence.
    
    Args:
        audio: Audio samples
        threshold: Energy threshold for silence
        min_silence_frames: Minimum frames to consider silence
        
    Returns:
        True if audio is silence
    """
    if len(audio) < FRAME_SAMPLES * min_silence_frames:
        return True
    
    # Check last portion of audio
    check_samples = FRAME_SAMPLES * min_silence_frames
    recent = audio[-check_samples:]
    energy = audio_energy(recent)
    
    return energy < threshold


class AudioRecorder:
    """Simple audio recorder with silence detection."""
    
    def __init__(
        self,
        rate: int = RATE,
        channels: int = CHANNELS,
        frame_ms: int = FRAME_MS,
        silence_threshold: float = 500.0,
        max_silence_seconds: float = 1.5,
        min_speech_seconds: float = 0.5,
        max_duration_seconds: float = 30.0,
    ):
        self.rate = rate
        self.channels = channels
        self.frame_samples = int(rate * frame_ms / 1000)
        self.silence_threshold = silence_threshold
        self.max_silence_frames = int(max_silence_seconds * 1000 / frame_ms)
        self.min_speech_samples = int(min_speech_seconds * rate)
        self.max_samples = int(max_duration_seconds * rate)
        
        self._stream = None
        self._recording = False
    
    def record_until_silence(
        self,
        stop_check: Callable[[], bool] = lambda: False,
        vad=None
    ) -> Tuple[np.ndarray, float]:
        """Record audio until silence is detected or stop_check returns True.
        
        Args:
            stop_check: Callable that returns True to stop recording
            vad: Optional VAD instance for voice activity detection
            
        Returns:
            (audio_array, duration_seconds)
        """
        import sounddevice as sd
        
        frames: list[np.ndarray] = []
        silence_count = 0
        speech_detected = False
        
        logger.debug("Starting audio recording")
        
        try:
            with sd.InputStream(
                samplerate=self.rate,
                channels=self.channels,
                dtype=DTYPE,
                blocksize=self.frame_samples,
            ) as stream:
                while True:
                    if stop_check():
                        logger.debug("Recording stopped by stop_check")
                        break
                    
                    frame, overflowed = stream.read(self.frame_samples)
                    if overflowed:
                        logger.warning("Audio buffer overflow")
                    
                    frame_1d = frame.flatten()
                    frames.append(frame_1d)
                    
                    # Check for speech/silence
                    energy = audio_energy(frame_1d)
                    is_speech = energy > self.silence_threshold
                    
                    if vad is not None:
                        try:
                            is_speech = vad.is_speech(frame_1d.tobytes(), self.rate)
                        except Exception:
                            pass
                    
                    if is_speech:
                        speech_detected = True
                        silence_count = 0
                    else:
                        silence_count += 1
                    
                    # Stop on extended silence after speech
                    if speech_detected and silence_count >= self.max_silence_frames:
                        logger.debug("Silence detected, stopping recording")
                        break
                    
                    # Stop on max duration
                    total_samples = sum(len(f) for f in frames)
                    if total_samples >= self.max_samples:
                        logger.debug("Max duration reached")
                        break
        
        except Exception as e:
            logger.error(f"Recording error: {e}")
        
        if not frames:
            return np.array([], dtype=DTYPE), 0.0
        
        audio = np.concatenate(frames)
        duration = len(audio) / self.rate
        
        logger.debug(f"Recorded {duration:.2f}s of audio")
        return audio, duration


def audio_chunks(
    rate: int = RATE,
    frame_ms: int = FRAME_MS,
    stop_check: Callable[[], bool] = lambda: False,
) -> Generator[np.ndarray, None, None]:
    """Generate audio chunks for streaming.
    
    Args:
        rate: Sample rate
        frame_ms: Frame duration in milliseconds
        stop_check: Callable that returns True to stop
        
    Yields:
        Audio frames as numpy arrays
    """
    import sounddevice as sd
    
    frame_samples = int(rate * frame_ms / 1000)
    
    try:
        with sd.InputStream(
            samplerate=rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=frame_samples,
        ) as stream:
            while not stop_check():
                frame, _ = stream.read(frame_samples)
                yield frame.flatten()
    except Exception as e:
        logger.error(f"Audio stream error: {e}")

