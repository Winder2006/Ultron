"""Standalone test of Deepgram streaming STT in the current SDK."""
import asyncio
import time
from dotenv import load_dotenv
load_dotenv()

import numpy as np

async def main():
    from mother.audio.stt import StreamingSTT

    stt = StreamingSTT()
    await stt.init()
    print(f"Deepgram available: {stt._deepgram_available}")

    # Generate 3 seconds of silence as a trivial test — Deepgram will say empty
    audio = np.zeros(16000 * 2, dtype=np.float32)

    t0 = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()

    # Feed audio in chunks like real mic would
    async def feeder():
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            await queue.put(audio[i:i+chunk_size])
            await asyncio.sleep(chunk_size / 16000)  # real-time
        await queue.put(None)

    feed_task = asyncio.create_task(feeder())

    got_any = False
    async for text, is_final in stt.stream(queue):
        elapsed = time.monotonic() - t0
        print(f"[{elapsed:.3f}s] final={is_final} text={text!r}")
        got_any = True
        if is_final:
            break

    await feed_task
    if not got_any:
        print("No transcripts received (empty silent input is expected)")

asyncio.run(main())
