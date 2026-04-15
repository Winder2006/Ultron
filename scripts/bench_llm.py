"""Benchmark LLM TTFT for different Gemini/Claude models."""
import time
from dotenv import load_dotenv
load_dotenv()

import litellm

MODELS = [
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.0-flash-lite",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-flash-8b",
    "anthropic/claude-haiku-4-5-20251001",
]

messages = [
    {"role": "system", "content": "Reply in one short sentence."},
    {"role": "user", "content": "Say hello."},
]

for model in MODELS:
    try:
        # Warmup
        resp = litellm.completion(model=model, messages=messages, max_tokens=20, stream=True)
        for _ in resp:
            pass

        # Measure
        t0 = time.monotonic()
        first_token = None
        full = []
        resp = litellm.completion(model=model, messages=messages, max_tokens=20, stream=True)
        for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                if first_token is None:
                    first_token = time.monotonic() - t0
                full.append(delta.content)
        total = time.monotonic() - t0
        print(f"{model:50s} TTFT={first_token:.3f}s  total={total:.3f}s  resp={''.join(full)[:50]!r}")
    except Exception as e:
        print(f"{model:50s} FAILED: {type(e).__name__}: {str(e)[:80]}")
