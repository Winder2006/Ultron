"""Benchmark all Cerebras production + preview models."""
import time
from dotenv import load_dotenv
load_dotenv()

import litellm

MODELS = [
    "cerebras/gpt-oss-120b",
    "cerebras/llama3.1-8b",
    "cerebras/llama-3.1-8b",
    "cerebras/qwen-3-235b-a22b-instruct-2507",
    "cerebras/zai-glm-4.7",
    "cerebras/zai-glm-4-7",
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
        print(f"{model:55s} TTFT={first_token:.3f}s  total={total:.3f}s  resp={''.join(full)[:40]!r}")
    except Exception as e:
        name = type(e).__name__
        msg = str(e)[:80]
        print(f"{model:55s} FAILED: {name}: {msg}")
