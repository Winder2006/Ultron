"""Always-on wake word detection for MOTHER.

Uses openWakeWord with a custom 'MOTHER' model (ONNX).
Falls back to built-in 'hey_jarvis' if custom model not found.
Runs entirely on CPU in a background thread.

Events emitted via callback or asyncio.Queue:
    {"event": "wake_word_detected", "score": float, "keyword": str}
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("mother.wake_word")

CUSTOM_MODEL_PATH = os.environ.get(
    "WAKE_WORD_MODEL_PATH", "./models/mother_wakeword.onnx"
)
FALLBACK_MODEL = "hey_jarvis_v0.1"
CHUNK_SIZE = 1280  # 80ms at 16kHz — openWakeWord requirement


class WakeWordDetector:
    """Continuously listens on mic for the wake word.

    Parameters:
        event_queue: asyncio.Queue to push detection events into.
        sensitivity: Detection threshold (0.0-1.0). Lower = more sensitive.
        on_detected: Optional sync callback fired on detection (for non-async use).
    """

    def __init__(
        self,
        event_queue: Optional[asyncio.Queue] = None,
        sensitivity: float = 0.5,
        on_detected: Optional[Callable[[str, float], None]] = None,
    ):
        self._event_queue = event_queue
        self._sensitivity = sensitivity
        self._on_detected = on_detected
        self._model = None
        self._model_name = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _load_model(self):
        """Load custom MOTHER model if exists, else use fallback."""
        from openwakeword.model import Model

        custom = Path(CUSTOM_MODEL_PATH)
        if custom.exists():
            logger.info("[WakeWord] Loading custom model: %s", custom)
            self._model = Model(
                wakeword_models=[str(custom)],
                inference_framework="onnx",
            )
            self._model_name = custom.stem
        else:
            logger.warning(
                "[WakeWord] Custom model not found at %s — using fallback '%s'",
                custom,
                FALLBACK_MODEL,
            )
            self._model = Model(
                wakeword_models=[FALLBACK_MODEL],
                inference_framework="onnx",
            )
            self._model_name = FALLBACK_MODEL

        logger.info(
            "[WakeWord] Loaded model: %s (sensitivity=%.2f)",
            self._model_name,
            self._sensitivity,
        )

    def start(self):
        """Begin continuous listening in a background thread."""
        if self._running:
            return
        self._load_model()
        self._running = True
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("[WakeWord] Detector started (background thread)")

    def stop(self):
        """Stop the detector."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("[WakeWord] Detector stopped")

    def _listen_loop(self):
        """Main listening loop — runs in background thread."""
        import sounddevice as sd

        sample_rate = 16000
        # Use a blocking InputStream so we don't need a callback
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE,
            ) as stream:
                logger.info("[WakeWord] Mic stream open — listening for '%s'", self._model_name)
                while self._running:
                    audio, overflowed = stream.read(CHUNK_SIZE)
                    if overflowed:
                        continue
                    # openWakeWord expects int16 numpy array
                    chunk = audio.flatten()
                    prediction = self._model.predict(chunk)

                    for keyword, score in prediction.items():
                        if score >= self._sensitivity:
                            self._fire_detection(keyword, score)
                            # Reset model state to avoid rapid re-fires
                            self._model.reset()
                            # Brief cooldown to prevent double-triggers
                            time.sleep(0.5)
                            break
        except Exception as e:
            logger.error("[WakeWord] Listener error: %s", e)
            self._running = False

    def _fire_detection(self, keyword: str, score: float):
        """Handle a wake word detection."""
        logger.info("[WakeWord] Detected '%s' (score: %.2f)", keyword, score)

        # Sync callback
        if self._on_detected:
            try:
                self._on_detected(keyword, score)
            except Exception as e:
                logger.error("[WakeWord] Callback error: %s", e)

        # Async queue
        if self._event_queue and self._loop:
            event = {
                "event": "wake_word_detected",
                "score": score,
                "keyword": keyword,
            }
            self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def model_name(self) -> str:
        return self._model_name
