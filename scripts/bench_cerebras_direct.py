"""Benchmark Cerebras models via direct OpenAI-compatible HTTP API."""
import os
import time
import json
import httpx
from dotenv import load_dotenv
load_dotenv()

KEY = os.environ["CEREBRAS_API_KEY"]
URL = "https://api.cerebras.ai/v1/chat/completions"

MODELS = [
    "llama3.1-8b",
    "qwen-3-235b-a22b-instruct-2507",
    "gpt-oss-120b",
    "zai-glm-4.7",
]

messages = [
    {"role": "system", "content": "Reply in one short sentence."},
    {"role": "user", "content": "Say hello."},
]

for model in MODELS:
    body = {"model": model, "messages": messages, "max_tokens": 20, "stream": True}
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

    # Warmup
    try:
        with httpx.stream("POST", URL, json=body, headers=headers, timeout=15) as r:
            for _ in r.iter_lines():
                pass
    except Exception:
        pass

    # Measure
    try:
        t0 = time.monotonic()
        first = None
        full = []
        with httpx.stream("POST", URL, json=body, headers=headers, timeout=15) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    if first is None:
                        first = time.monotonic() - t0
                    full.append(content)
        total = time.monotonic() - t0
        print(f"{model:40s} TTFT={first:.3f}s  total={total:.3f}s  resp={''.join(full)[:50]!r}")
    except Exception as e:
        print(f"{model:40s} FAILED: {type(e).__name__}: {str(e)[:100]}")
