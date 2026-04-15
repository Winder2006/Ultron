"""Shim: re-exports from mother.audio.stt for backward compatibility."""
from mother.audio.stt import *  # noqa: F401,F403
from mother.audio.stt import (  # explicit for IDE
    STTEngine, FasterWhisperConfig, FasterWhisperSTT,
    DeepgramConfig, StreamingSTT,
)
