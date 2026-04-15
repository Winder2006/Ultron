"""Save all TTS chunks from one response into a WAV file so we can hear exactly what the browser receives."""
import asyncio
import base64
import json
import time
import websockets
import numpy as np
import wave


async def main():
    uri = "ws://localhost:8300/ws/voice"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"action": "prompt", "text": "Tell me a short three-sentence story about a dragon."}))

        pcm_bytes = bytearray()
        sentence_markers = []  # sample positions where new sentences begin
        current_sample = 0

        async for raw in ws:
            ev = json.loads(raw)
            if ev.get("event") == "tts_start":
                sentence_markers.append(current_sample)
                print(f"  sentence_start at sample {current_sample}: {ev.get('text')[:40]!r}")
            elif ev.get("event") == "tts_chunk":
                b64 = ev.get("pcm_b64", "")
                data = base64.b64decode(b64)
                pcm_bytes.extend(data)
                current_sample += len(data) // 2
            elif ev.get("event") == "llm_done":
                break
            elif ev.get("event") == "error":
                print(f"ERROR: {ev.get('message')}")
                break

        # Save to WAV
        with wave.open("out/debug_tts_chunks.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(bytes(pcm_bytes))

        # Analyze amplitude at each sentence boundary
        samples = np.frombuffer(bytes(pcm_bytes), dtype=np.int16)
        print(f"\nTotal samples: {len(samples)}")
        print(f"Sentence markers: {sentence_markers}")
        print("\nAmplitude at sentence boundaries (should all be ~0):")
        for m in sentence_markers:
            if m == 0:
                continue
            # Check 30 samples before and 30 after the boundary
            before = samples[max(0, m-30):m]
            after = samples[m:m+30]
            print(f"  boundary @ {m}:")
            print(f"    before (last 30): {list(before)}")
            print(f"    after  (first 30): {list(after)}")
            # Check for pattern of alternating zeros (stereo-as-mono bug)
            after_abs = np.abs(after)
            zeros_at_even = sum(1 for i in range(0, len(after), 2) if after_abs[i] < 5)
            zeros_at_odd  = sum(1 for i in range(1, len(after), 2) if after_abs[i] < 5)
            print(f"    zeros @ even idx: {zeros_at_even}/{len(after)//2+1}, @ odd idx: {zeros_at_odd}/{len(after)//2}")

        print(f"\nWrote out/debug_tts_chunks.wav — play this to hear exactly what the browser gets")


asyncio.run(main())
