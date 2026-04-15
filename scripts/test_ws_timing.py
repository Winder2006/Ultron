"""Test WebSocket prompt latency end-to-end."""
import asyncio
import json
import time
import websockets


async def main():
    uri = "ws://localhost:8300/ws/voice"
    async with websockets.connect(uri) as ws:
        t0 = time.monotonic()
        await ws.send(json.dumps({"action": "prompt", "text": "Hello, can you hear me?"}))
        print(f"[{time.monotonic() - t0:.3f}s] prompt sent")

        first_token_t = None
        first_tts_t = None
        last_tts_t = None
        tts_count = 0
        token_count = 0

        async for raw in ws:
            ev = json.loads(raw)
            elapsed = time.monotonic() - t0
            evt = ev.get("event")
            if evt == "intent":
                print(f"[{elapsed:.3f}s] intent={ev.get('intent')} route={ev.get('route')} tier={ev.get('tier')}")
            elif evt == "llm_token":
                if first_token_t is None:
                    first_token_t = elapsed
                    print(f"[{elapsed:.3f}s] FIRST TOKEN: {ev.get('token')!r}")
                token_count += 1
            elif evt == "tts_ready":
                if first_tts_t is None:
                    first_tts_t = elapsed
                    print(f"[{elapsed:.3f}s] FIRST TTS READY ({len(ev.get('audio_b64', ''))} chars b64)")
                last_tts_t = elapsed
                tts_count += 1
            elif evt == "llm_done":
                text = ev.get("full_text", "")
                print(f"[{elapsed:.3f}s] LLM_DONE ({len(text)} chars): {text[:80]!r}")
                print(f"[{elapsed:.3f}s] Total tokens: {token_count}")
                print(f"[{elapsed:.3f}s] TTS chunks: {tts_count}")
                break
            else:
                print(f"[{elapsed:.3f}s] {evt}")

        if first_token_t:
            print(f"\nTTFT: {first_token_t:.3f}s")
        if first_tts_t:
            print(f"First audio ready: {first_tts_t:.3f}s")
        if last_tts_t:
            print(f"Last audio ready: {last_tts_t:.3f}s")


asyncio.run(main())
