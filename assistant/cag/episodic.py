from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from ..nlp.embeddings import embed_texts


PROFILE_PATH = Path("assistant/memory/profile.json")


def _load_profile() -> dict:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {"episodic": []}


def _save_profile(data: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def top_nuggets(query: str, n: int = 3) -> List[str]:
    prof = _load_profile()
    nuggets = prof.get("episodic", [])
    if not nuggets:
        return []
    texts = [x.get("text", "") for x in nuggets]
    X = embed_texts(texts)
    q = embed_texts([query])
    if X.size == 0 or q.size == 0:
        return texts[:n]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    scores = (X @ q.T).ravel()
    order = np.argsort(-scores)
    return [texts[i] for i in order[:n]]


def maybe_write_nugget(text: str, tags: List[str], confidence: float) -> bool:
    """Store if looks durable/personal.

    Heuristics: text length 5..200, confidence >= 0.7.
    """
    text = (text or "").strip()
    if not (5 <= len(text) <= 200):
        return False
    if confidence < 0.7:
        return False
    prof = _load_profile()
    prof.setdefault("episodic", []).append({"text": text, "tags": tags or [], "confidence": float(confidence)})
    _save_profile(prof)
    return True


