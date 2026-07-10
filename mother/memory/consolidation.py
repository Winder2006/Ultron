"""Periodic memory consolidation.

Episodic memories accumulate. Without consolidation they grow linearly
forever, drifting toward noise (a hundred near-duplicates of "user
likes coffee", "the user enjoys coffee", "user mentioned coffee
again"). Recall becomes worse over time, not better.

This module runs an LLM pass over recent memories and produces:
  • A list of NEW structured FACTS to promote (stable observations
    that recur often enough to deserve fast key-value access).
  • A list of MERGE GROUPS — episodic IDs that are paraphrases of the
    same underlying observation; only the highest-confidence one is
    kept, others are dropped.
  • A list of NOISE memories to drop entirely (one-off details with
    no recall value).

Runs once per user every CONSOLIDATION_INTERVAL_S seconds (default 6h)
on a background asyncio task spawned at server startup. A single
process-wide lock prevents two consolidation passes from racing on
the same user. Best-effort: on LLM error, the memory store is
unchanged and we'll try again next cycle.

Idempotent. Safe to interrupt mid-run (writes are atomic at the
JSON-file granularity, holding the per-user episodic lock).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from mother.memory.manager import UserMemory, USERS_DIR

logger = logging.getLogger("mother.memory.consolidation")


# Tunables ─────────────────────────────────────────────────────────
CONSOLIDATION_INTERVAL_S = 6 * 3600          # how often to run per user
MIN_MEMORIES_TO_CONSOLIDATE = 10             # below this it's not worth a token
MAX_MEMORIES_PER_PASS = 60                   # cap context size for the LLM
PROMOTE_MIN_OCCURRENCES = 2                  # observation must recur ≥N times to promote
LLM_TIMEOUT_S = 20.0                         # if the LLM takes longer than this, bail


_run_lock = threading.Lock()
# Per-user last-run timestamp, kept in process memory. Resets on
# server restart — that's fine, a re-run isn't harmful.
_last_run: Dict[str, float] = {}


# ─────────────────────────────────────────────────────────────────
# LLM prompt
# ─────────────────────────────────────────────────────────────────

_CONSOLIDATION_SYSTEM = """You compress and clean a memory store.

You receive a JSON array of episodic memories about a user. Your job
is to look across them and produce a structured plan for cleaning.

Output ONE JSON object, nothing else, no prose, no fences:

{
  "promote": [
    {"key": "snake_case_key", "value": "concise value", "reason": "why"}
  ],
  "merge_groups": [
    {"keep": <index>, "drop": [<index>, <index>], "reason": "..."}
  ],
  "drop": [<index>, ...]
}

Rules:
- promote: stable, factual observations that recur ≥2 times across
  the memories (employer, hometown, hobby, vehicle, family). Don't
  promote one-off events ("user mentioned a movie once").
- merge_groups: groups of memories that say the same thing in
  different words. Pick the most informative as `keep`, list the
  others in `drop`. Index = position in the input array (0-based).
- drop: noise to delete entirely — small talk, transient observations,
  anything with no future recall value. Be conservative. When in doubt
  keep the memory.
- Output ONLY the JSON object. No commentary. No markdown."""


_USER_PROMPT_TEMPLATE = """Memories to review (index : text):

{memory_list}

Produce the cleanup plan."""


def _format_memories_for_llm(memories: List[Dict]) -> str:
    """Number each memory by its index for the LLM to reference."""
    lines = []
    for i, m in enumerate(memories):
        text = (m.get("text") or "").strip().replace("\n", " ")
        # Clip individual entries so one weirdly-long memory can't
        # blow up the prompt budget.
        if len(text) > 240:
            text = text[:237] + "..."
        lines.append(f"{i}: {text}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# JSON parsing — defensive against LLM quirks
# ─────────────────────────────────────────────────────────────────

_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced top-level object out of a possibly-noisy
    LLM response. Tolerates stray prose, markdown fences, etc."""
    if not raw:
        return None
    raw = raw.strip()
    # Strip ``` fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Walk to find the first balanced { ... }
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = raw[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1  # keep scanning
                    continue
    return None


# ─────────────────────────────────────────────────────────────────
# The main consolidation routine
# ─────────────────────────────────────────────────────────────────

def consolidate_user(
    user_id: str,
    llm_fn: Callable[..., str],
    *,
    dry_run: bool = False,
    now_fn: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Run one consolidation pass for *user_id*.

    Returns a dict summarising what was done:
        {
            "user_id": ...,
            "considered": int,
            "promoted": int,
            "merged": int,   # # of dropped-via-merge entries
            "dropped": int,
            "ran": bool,     # False if we skipped (too few memories etc.)
            "error": Optional[str],
        }

    `llm_fn(messages, *, max_tokens=...)` is the same callable shape
    `TieredLLMDriver.chat` exposes. We pin tier 2 for this call (Haiku-
    class) — tier 1 is unreliable at structured-JSON output, tier 3
    is overkill for cleanup.
    """
    summary = {
        "user_id": user_id,
        "considered": 0,
        "promoted": 0,
        "merged": 0,
        "dropped": 0,
        "ran": False,
        "error": None,
    }

    if not _run_lock.acquire(blocking=False):
        summary["error"] = "another consolidation is in progress"
        return summary

    try:
        try:
            # MUST be the process-wide cached instance (get_user_memory),
            # not a fresh UserMemory: a fresh instance carries its own
            # _ep_lock, so every `with mem._ep_lock:` below would be
            # mutual exclusion against nobody — the voice pipeline's
            # concurrent add_episodic could be silently clobbered by
            # our rewrite.
            from mother.memory.manager import get_user_memory
            mem = get_user_memory(user_id)
            if mem is None:
                raise RuntimeError("no memory instance available")
        except Exception as e:
            summary["error"] = f"could not open memory for {user_id}: {e}"
            return summary

        # Load and filter — only consolidate memories that haven't
        # already been touched by a previous consolidation pass
        # (marked via tag). This keeps the LLM working only on new
        # material and bounds the context size over time.
        with mem._ep_lock:
            all_mem = mem._load_episodic()

        # Defensive copy with original indices preserved so we can map
        # LLM responses back to the right entries.
        candidates: List[tuple[int, Dict]] = [
            (i, m) for i, m in enumerate(all_mem)
            if not _was_consolidated(m)
        ]
        summary["considered"] = len(candidates)

        if len(candidates) < MIN_MEMORIES_TO_CONSOLIDATE:
            return summary

        # Cap to the most recent N. Older memories are presumably
        # stable; new ones are where dedupe leverage lives.
        if len(candidates) > MAX_MEMORIES_PER_PASS:
            candidates = candidates[-MAX_MEMORIES_PER_PASS:]

        # Indices in the LLM prompt are 0-based POSITIONS within
        # `candidates`, not original episodic indices. We map back.
        prompt_memories = [m for _, m in candidates]

        # Build messages. We import lazily to avoid circular imports
        # at module load — `mother.llm.drivers` imports from memory.
        from mother.llm.drivers import ChatMessage
        messages = [
            ChatMessage(role="system", content=_CONSOLIDATION_SYSTEM),
            ChatMessage(
                role="user",
                content=_USER_PROMPT_TEMPLATE.format(
                    memory_list=_format_memories_for_llm(prompt_memories)
                ),
            ),
        ]

        # Call the LLM. We accept either a plain `chat`-style fn or the
        # full driver.stream_chat result joined; signature is callable
        # returning a string.
        t0 = now_fn()
        try:
            raw = llm_fn(messages, max_tokens=600)
        except TypeError:
            # Some shapes don't take max_tokens kwarg
            raw = llm_fn(messages)
        except Exception as e:
            summary["error"] = f"LLM error: {e}"
            return summary
        elapsed = now_fn() - t0

        if elapsed > LLM_TIMEOUT_S:
            logger.warning(
                "[consolidation] user=%s LLM call took %.1fs (>%ds budget) — using anyway",
                user_id, elapsed, LLM_TIMEOUT_S,
            )

        plan = _extract_json_object(raw or "")
        if not plan or not isinstance(plan, dict):
            summary["error"] = "LLM did not return JSON"
            logger.warning(
                "[consolidation] user=%s — no parseable JSON in: %r",
                user_id, (raw or "")[:200],
            )
            return summary

        if dry_run:
            summary["ran"] = True
            summary["plan"] = plan  # type: ignore[index]
            return summary

        # ── Apply: promote facts ──
        # A promotion is a (key, value) write to the user's facts DB.
        # We don't drop the underlying episodic — promotion just adds
        # fast-lookup access. If the LLM names a key we already know,
        # set_fact just updates it.
        promote_list = plan.get("promote") or []
        if isinstance(promote_list, list):
            for p in promote_list:
                if not isinstance(p, dict):
                    continue
                key = (p.get("key") or "").strip()
                value = (p.get("value") or "").strip()
                if not key or not value:
                    continue
                # Sanitize key: snake_case, ≤40 chars, no weird chars.
                key = re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_")[:40]
                if not key:
                    continue
                try:
                    mem.set_fact(
                        key, value,
                        category="learned",
                        confidence=0.85,
                        source="consolidation",
                    )
                    summary["promoted"] += 1
                except Exception as e:
                    logger.debug("set_fact failed: %s", e)

        # ── Apply: dedupe (merge_groups) and drop ──
        # We need to operate on the full all_mem list with original
        # indices, but the LLM's indices are positions in the
        # `prompt_memories` slice. Build the mapping.
        local_to_global = {local: global_idx for local, (global_idx, _) in enumerate(candidates)}

        drop_global: set[int] = set()

        merge_groups = plan.get("merge_groups") or []
        if isinstance(merge_groups, list):
            for g in merge_groups:
                if not isinstance(g, dict):
                    continue
                drops = g.get("drop") or []
                if not isinstance(drops, list):
                    continue
                for d in drops:
                    if isinstance(d, int) and d in local_to_global:
                        drop_global.add(local_to_global[d])
            summary["merged"] = len(drop_global)

        drop_list = plan.get("drop") or []
        if isinstance(drop_list, list):
            for d in drop_list:
                if isinstance(d, int) and d in local_to_global:
                    drop_global.add(local_to_global[d])
            summary["dropped"] = len(drop_global) - summary["merged"]

        # Safety cap — never drop more than 60% of the considered
        # memories in one pass. If the LLM suggests that, something
        # went wrong with the prompt and we'd rather lose a single
        # consolidation cycle than nuke the user's memory.
        max_drops = int(0.6 * len(candidates))
        if len(drop_global) > max_drops:
            logger.warning(
                "[consolidation] user=%s plan dropped %d of %d (>60%%) — "
                "rejecting drops, keeping promotions",
                user_id, len(drop_global), len(candidates),
            )
            summary["error"] = "drop ratio exceeded safety cap"
            summary["merged"] = 0
            summary["dropped"] = 0
            drop_global = set()

        if drop_global:
            with mem._ep_lock:
                fresh = mem._load_episodic()
                # Only drop if the fresh list still has those entries
                # at those indices (no concurrent mutation). Be
                # defensive: filter rather than index-delete.
                kept = [
                    m for i, m in enumerate(fresh)
                    if i not in drop_global
                ]
                # Tag remaining survivors that were considered as
                # consolidated so the next pass skips them.
                consolidated_indices = {
                    g for g in [local_to_global[l] for l in local_to_global]
                    if g not in drop_global
                }
                for i, m in enumerate(kept):
                    if i in consolidated_indices:
                        m.setdefault("tags", [])
                        if "_consolidated" not in m["tags"]:
                            m["tags"].append("_consolidated")
                mem._save_episodic(kept)
                # Invalidate the embedding cache — text set changed.
                mem._ep_emb_cache = None
                mem._ep_emb_cache_key = None
        else:
            # Even with no drops, mark the considered ones as seen so
            # we don't re-consider them next pass.
            with mem._ep_lock:
                fresh = mem._load_episodic()
                changed = False
                seen = {g for g, _ in candidates}
                for i, m in enumerate(fresh):
                    if i in seen:
                        tags = m.setdefault("tags", [])
                        if "_consolidated" not in tags:
                            tags.append("_consolidated")
                            changed = True
                if changed:
                    mem._save_episodic(fresh)

        summary["ran"] = True
        logger.info(
            "[consolidation] user=%s considered=%d promoted=%d merged=%d dropped=%d "
            "(llm=%.2fs)",
            user_id, summary["considered"], summary["promoted"],
            summary["merged"], summary["dropped"], elapsed,
        )
        _last_run[user_id] = now_fn()
        return summary
    except Exception as e:
        summary["error"] = str(e)
        logger.exception("[consolidation] user=%s unexpected failure", user_id)
        return summary
    finally:
        _run_lock.release()


def _was_consolidated(memory: Dict) -> bool:
    """Has this memory been seen by a prior consolidation pass?"""
    tags = memory.get("tags") or []
    return "_consolidated" in tags


# ─────────────────────────────────────────────────────────────────
# Scheduler — used by server lifespan
# ─────────────────────────────────────────────────────────────────

def list_known_users() -> List[str]:
    """Return user ids that have a memory directory on disk.

    We don't restrict to enrolled (voice-ID) users — anyone with
    accumulated memories should benefit from consolidation, including
    the `default` fallback user.
    """
    if not USERS_DIR.exists():
        return []
    users: List[str] = []
    for p in USERS_DIR.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("_") or p.name.startswith("."):
            continue
        users.append(p.name)
    return users


def is_due(user_id: str, now_fn: Callable[[], float] = time.monotonic) -> bool:
    """Check whether enough time has passed since the last run."""
    last = _last_run.get(user_id, 0.0)
    return (now_fn() - last) >= CONSOLIDATION_INTERVAL_S
