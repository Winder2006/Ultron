"""Push-to-talk input trigger for MOTHER.

Wraps the Enter-key recording trigger in an async interface compatible
with the new StreamingSTT pipeline. Also provides a sync interface
for backward compatibility with the existing cli.py loop.

Press Enter to start recording, press Enter again to stop.
Audio is queued as float32 numpy chunks at 16kHz.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from typing import Optional

import numpy as np


class PushToTalkTrigger:
    """Async push-to-talk recording triggered by Enter key.

    Usage:
        ptt = PushToTalkTrigger()
        audio_queue = asyncio.Queue()
        await ptt.record_to_queue(audio_queue)
        # audio_queue now has chunks + None sentinel
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._recording = False

    async def wait_for_keypress(self) -> None:
        """Block until Enter is pressed. Works on both Windows and Unix."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._blocking_wait_enter)

    def _blocking_wait_enter(self) -> None:
        """Blocking Enter key wait — runs in thread executor."""
        if sys.platform == "win32":
            try:
                import msvcrt
                while True:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ("\r", "\n"):
                            return
                    else:
                        import time
                        time.sleep(0.02)
            except ImportError:
                input()
        else:
            input()

    async def record_to_queue(
        self,
        audio_queue: asyncio.Queue,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Record audio from mic until Enter is pressed again.

        Pushes float32 numpy chunks into audio_queue, then a None sentinel.
        """
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        recording_chunks: list[np.ndarray] = []
        stream_ref: list = [None]

        def audio_callback(indata, frames, time_info, status):
            recording_chunks.append(indata.copy())
            # Also push to queue for real-time streaming
            try:
                audio_queue.put_nowait(indata[:, 0].copy())
            except asyncio.QueueFull:
                pass

        # Start recording
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        stream.start()
        stream_ref[0] = stream
        self._recording = True

        # Wait for Enter or stop_event
        if stop_event:
            enter_done = asyncio.Event()

            async def _wait_enter():
                await self.wait_for_keypress()
                enter_done.set()

            enter_task = asyncio.create_task(_wait_enter())
            done, pending = await asyncio.wait(
                [enter_task, asyncio.create_task(stop_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
        else:
            await self.wait_for_keypress()

        # Stop recording
        stream.stop()
        stream.close()
        self._recording = False

        # Send None sentinel to signal end of audio
        await audio_queue.put(None)

    async def listen(self) -> np.ndarray:
        """Record a complete utterance and return as a single numpy array.

        Returns raw float32 PCM at self.sample_rate.
        """
        queue: asyncio.Queue = asyncio.Queue()
        await self.record_to_queue(queue)

        chunks = []
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            chunks.append(chunk)

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
