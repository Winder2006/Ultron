"""Long-conversation semantic memory.

Every user/assistant exchange gets embedded and stored in a per-user
FAISS index. On each new turn, the top-K most similar past exchanges
are retrieved and injected into the system prompt — so when you ask
"what did we decide about X last week", the relevant exchange surfaces
even though the rolling-summary path long since collapsed it.

Composes with:
  • Conversation memory (sliding 4-turn window)        — verbatim recent
  • Conversation memory rolling summary                — gist of older
  • Episodic memory (UserMemory.search_episodic)       — extracted facts
  • THIS                                               — searchable raw history

Shape:
  Storage:  assistant/memory/users/<uid>/long_memory.index   (FAISS IndexFlatIP)
            assistant/memory/users/<uid>/long_memory_meta.jsonl
  Each line of the meta file describes one indexed exchange:
    {"id": int, "ts": iso8601, "user": str, "assistant": str}

  Index dimension: 384 (MiniLM ONNX), vectors L2-normalized so inner
  product equals cosine similarity.

Skip rules:
  • Trivial greetings / acks aren't worth indexing.
  • Empty assistant responses (canceled turns, errors) are skipped.
  • Anything <12 chars combined is skipped.

Concurrency:
  One file lock per user keeps the writer (voice route ending a turn)
  and reader (voice route starting next turn) from racing.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mother.memory.manager import USERS_DIR

logger = logging.getLogger("mother.memory.long")

# Retrieval defaults — kept conservative because each retrieved exchange
# can be ~200-400 tokens once formatted. Three is enough to surface real
# context without crowding out the system prompt.
DEFAULT_TOP_K = 3

# Index hygiene
_EMB_DIM = 384
_MIN_TEXT_CHARS = 12
_TRIVIAL_INPUTS = {
    "yes", "no", "ok", "okay", "sure", "thanks", "thank you",
    "hi", "hello", "hey", "good morning", "good evening",
    "good night", "goodnight", "bye", "goodbye",
    "yeah", "yep", "nope", "cool", "got it", "right",
}

# Per-user singleton + lock map. Building a FAISS index + loading the
# encoder is non-trivial; we keep one LongMemory per user_id alive for
# the process.
_INSTANCES: Dict[str, "LongMemory"] = {}
_INSTANCES_LOCK = threading.Lock()


def get_long_memory(user_id: str) -> "LongMemory":
    """Return the (cached) LongMemory for *user_id*, creating it if needed."""
    with _INSTANCES_LOCK:
        m = _INSTANCES.get(user_id)
        if m is None:
            m = LongMemory(user_id)
            _INSTANCES[user_id] = m
        return m


class LongMemory:
    """Per-user FAISS-backed conversation history."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dir = USERS_DIR / user_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "long_memory.index"
        self.meta_path = self.dir / "long_memory_meta.jsonl"

        self._lock = threading.Lock()
        self._index = None  # faiss.IndexFlatIP — created lazily
        self._meta: List[dict] = []
        self._loaded = False

    # ─────────────────────── load / persist ──────────────────────

    def _ensure_loaded(self) -> None:
        """Load FAISS index + meta from disk on first use."""
        if self._loaded:
            return
        try:
            import faiss
        except ImportError:
            logger.warning("[long_memory] faiss not installed — disabling")
            self._loaded = True
            return

        # Meta first — index without meta is unusable.
        if self.meta_path.exists():
            try:
                with self.meta_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            self._meta.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning("[long_memory] meta load failed: %s", e)

        # Index
        if self.index_path.exists() and self._meta:
            try:
                self._index = faiss.read_index(str(self.index_path))
                # Sanity: vectors and meta should be the same count.
                # The usual cause of drift is a crash between
                # _append_meta and _save_index, leaving meta one entry
                # ahead of the index. Repair IN PLACE: trim the meta
                # tail down to what the index actually holds and
                # rewrite the file. The old behavior (wipe in-memory
                # state, leave stale files on disk) recreated the
                # mismatch on the very next add — so the ENTIRE history
                # was silently discarded again on every restart.
                n_idx, n_meta = self._index.ntotal, len(self._meta)
                if n_idx != n_meta:
                    if 0 < n_idx < n_meta:
                        logger.warning(
                            "[long_memory] %s: meta ahead of index (%d vs %d)"
                            " — trimming meta tail",
                            self.user_id, n_meta, n_idx,
                        )
                        self._meta = self._meta[:n_idx]
                        self._rewrite_meta()
                    else:
                        # Index has vectors with no meta (or is empty)
                        # — unusable. Reset BOTH disk files so memory
                        # and disk agree again.
                        logger.warning(
                            "[long_memory] %s: index/meta mismatch (%d vs %d)"
                            " — resetting store",
                            self.user_id, n_idx, n_meta,
                        )
                        self._index = None
                        self._meta = []
                        self._reset_disk()
            except Exception as e:
                logger.warning("[long_memory] index load failed: %s — resetting", e)
                self._index = None
                self._meta = []
                self._reset_disk()

        if self._index is None:
            self._index = faiss.IndexFlatIP(_EMB_DIM)

        self._loaded = True
        logger.info(
            "[long_memory] %s loaded — %d exchanges indexed",
            self.user_id, len(self._meta),
        )

    def _rewrite_meta(self) -> None:
        """Rewrite the JSONL meta file from the in-memory list."""
        try:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            with self.meta_path.open("w", encoding="utf-8") as f:
                for rec in self._meta:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[long_memory] meta rewrite failed: %s", e)

    def _reset_disk(self) -> None:
        """Remove both store files so disk matches the (empty) memory
        state. Leaving stale files behind guarantees the next append
        recreates the index/meta mismatch."""
        for p in (self.index_path, self.meta_path):
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("[long_memory] reset unlink %s failed: %s", p, e)

    def _save_index(self) -> None:
        """Persist the FAISS index. Called after each add."""
        try:
            import faiss
            faiss.write_index(self._index, str(self.index_path))
        except Exception as e:
            logger.warning("[long_memory] save_index failed: %s", e)

    def _append_meta(self, record: dict) -> None:
        """Append one meta record to disk (atomic on POSIX, best-effort on Win)."""
        try:
            with self.meta_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[long_memory] meta append failed: %s", e)

    # ─────────────────────── add / search ────────────────────────

    @staticmethod
    def _is_trivial(user_text: str) -> bool:
        t = (user_text or "").strip().lower().rstrip("!.?,")
        return t in _TRIVIAL_INPUTS

    @staticmethod
    def _embed_one(text: str) -> Optional[np.ndarray]:
        """Embed a single string and L2-normalize. Returns shape (1, 384) or None."""
        try:
            from assistant.nlp.embeddings import embed_texts
            v = embed_texts([text])
            if v.size == 0:
                return None
            n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            return (v / n).astype(np.float32, copy=False)
        except Exception as e:
            logger.debug("[long_memory] embed failed: %s", e)
            return None

    def add_exchange(
        self,
        user_text: str,
        assistant_text: str,
    ) -> bool:
        """Embed and index one user/assistant exchange. Returns True if added."""
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text or not assistant_text:
            return False
        if len(user_text) + len(assistant_text) < _MIN_TEXT_CHARS:
            return False
        if self._is_trivial(user_text):
            return False

        with self._lock:
            self._ensure_loaded()
            if self._index is None:
                return False

            # Embed the JOINT exchange so user-question and assistant-answer
            # share one vector — recall returns them together as one unit
            # of context, which reads more naturally than retrieving them
            # separately and stitching.
            joint = f"User: {user_text}\nAssistant: {assistant_text}"
            vec = self._embed_one(joint)
            if vec is None:
                return False

            new_id = len(self._meta)
            record = {
                "id": new_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "user": user_text,
                "assistant": assistant_text,
            }
            self._index.add(vec)
            self._meta.append(record)
            self._append_meta(record)
            # Save index on every add — cheap (FAISS Flat is just numpy
            # serialisation) and makes the next-process-boot pickup work.
            self._save_index()
            logger.debug(
                "[long_memory] %s added id=%d (%d total)",
                self.user_id, new_id, len(self._meta),
            )
            return True

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        min_score: float = 0.45,
    ) -> List[Tuple[float, dict]]:
        """Return [(similarity, record), ...] sorted high→low.

        `min_score` is on cosine similarity (vectors are L2-normed). 0.45
        is conservative — true matches typically score 0.6+, noisy matches
        sit around 0.3. Tune empirically.
        """
        if not query or not query.strip():
            return []
        with self._lock:
            self._ensure_loaded()
            if self._index is None or len(self._meta) == 0:
                return []
            qvec = self._embed_one(query)
            if qvec is None:
                return []

            k_actual = min(k, len(self._meta))
            scores, idxs = self._index.search(qvec, k_actual)
            results: List[Tuple[float, dict]] = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                if score < min_score:
                    continue
                results.append((float(score), self._meta[idx]))
            return results

    def stats(self) -> Dict:
        with self._lock:
            self._ensure_loaded()
            return {
                "user_id": self.user_id,
                "indexed": len(self._meta),
                "first_ts": self._meta[0]["ts"] if self._meta else None,
                "last_ts": self._meta[-1]["ts"] if self._meta else None,
            }


def format_for_prompt(hits: List[Tuple[float, dict]]) -> str:
    """Render search hits as a system-prompt snippet.

    Format is deliberately bare so it slots into a longer prompt without
    looking like a tool result. Timestamps are abbreviated.
    """
    if not hits:
        return ""
    lines: List[str] = []
    for score, rec in hits:
        ts = rec.get("ts", "")
        # Pretty-up the timestamp — "5 days ago", etc., would be nicer
        # but parsing iso to that is more code than the gain warrants.
        # Keep ISO-date-only for clarity.
        date_only = ts[:10] if ts else "?"
        u = (rec.get("user") or "").strip()
        a = (rec.get("assistant") or "").strip()
        # Clip individual sides so a long historical answer can't blow
        # the prompt budget.
        if len(u) > 240:
            u = u[:237] + "..."
        if len(a) > 360:
            a = a[:357] + "..."
        lines.append(f"[{date_only}] User: {u}\n           You: {a}")
    body = "\n".join(lines)
    return (
        "Earlier conversations with this user that may be relevant "
        "(retrieved by similarity, may or may not apply):\n" + body
    )
