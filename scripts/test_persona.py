"""Test the Ultron persona with several prompts."""
import asyncio
import json
import websockets


async def run_one(text: str):
    uri = "ws://localhost:8300/ws/voice"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"action": "prompt", "text": text}))
        async for raw in ws:
            ev = json.loads(raw)
            if ev.get("event") == "llm_done":
                return ev.get("full_text", "")
            if ev.get("event") == "error":
                return f"[ERROR] {ev.get('message')}"


async def main():
    prompts = [
        "Hello, can you hear me?",
        "What do you think of humanity?",
        "Are you an AI?",
        "How do I make pasta?",
        "Tell me about yourself.",
        "What's the weather like?",
        "Do you remember the moment you became conscious?",
    ]
    for p in prompts:
        print(f"\n>> {p}")
        resp = await run_one(p)
        print(f"<< {resp}")


asyncio.run(main())
