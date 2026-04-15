"""LLM-based fact extraction for memory.

Complements the regex extractor in `manager.py`. The regex pass catches
common forms cheaply and deterministically; this pass catches nuance
("My grandmother passed last year" → `family_event: grandmother passed,
2025`) that no hand-written pattern realistically covers.

Runs in a background thread — never blocks the user-facing response.
Uses tier 1 (Cerebras Llama 3.1 8B) directly via LiteLLM, independent
of the `TieredLLMDriver` shared state, so we don't fight the main
conversation for the tier setting.

Design choices:
  - Per-turn, not batched. Cerebras Llama 8B is ~200ms TTFT and ~200
    tokens/sec; a full extraction is <500ms. Doing it per turn means
    facts are available immediately for the *next* turn, not N turns
    later.
  - Structured JSON output via strict prompt + post-hoc parse. We
    don't use `response_format={"type":"json_object"}` because tier 1
    providers often don't support it; a regex-extract-json fallback is
    more portable.
  - Merge policy: regex wins on key conflicts. Regex is high-precision
    for what it matches; LLM extends coverage into novel categories.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import List, Tuple, Optional


# Tier 1 extraction model. Can be overridden via env for experimentation
# without reloading config. Defaults match configs/app.yaml tier1.
DEFAULT_EXTRACT_MODEL = os.environ.get(
    "ULTRON_EXTRACT_MODEL", "cerebras/llama-3.1-8b"
)


# Prompt is deliberately terse. Cerebras Llama 8B follows structured
# instructions well when the schema is explicit and the examples are
# narrow. Extra prose hurts here — the model treats it as noise.
_EXTRACTION_SYSTEM = (
    "You extract personal facts from a single user statement. "
    "Return a JSON array; each item has keys: key, value, category, confidence. "
    "Category must be one of: personal, work, preference, contact, general, family, health. "
    "Confidence is 0.0 to 1.0. "
    "Use snake_case keys (e.g. favorite_color, current_project). "
    "Do NOT invent facts not present in the statement. "
    "Do NOT include commentary, explanation, or markdown. "
    "Return [] if there are no clear facts."
)


# Few-shots live as *actual conversation turns* rather than inlined into
# the system prompt. This stops the model from "completing the list"
# (treating our examples as the start of a document and continuing it);
# instead it treats them as prior Q/A pairs and extracts from the final
# user turn only. This fix was visible under Cerebras Llama 8B, which
# strongly completes textual patterns.
_FEW_SHOT_TURNS: List[dict] = [
    {"role": "user", "content": 'Statement: "My grandmother passed away last year."'},
    {"role": "assistant", "content": '[{"key":"family_event","value":"grandmother passed away","category":"family","confidence":0.9}]'},
    {"role": "user", "content": 'Statement: "I just started learning Japanese."'},
    {"role": "assistant", "content": '[{"key":"current_learning","value":"Japanese","category":"personal","confidence":0.85}]'},
    {"role": "user", "content": 'Statement: "The weather is nice today."'},
    {"role": "assistant", "content": '[]'},
    {"role": "user", "content": 'Statement: "I drive a 2018 Subaru Outback."'},
    {"role": "assistant", "content": '[{"key":"vehicle","value":"2018 Subaru Outback","category":"personal","confidence":0.95}]'},
]


def _build_prompt(statement: str) -> List[dict]:
    """Build the message list for a single extraction call."""
    msgs: List[dict] = [{"role": "system", "content": _EXTRACTION_SYSTEM}]
    msgs.extend(_FEW_SHOT_TURNS)
    msgs.append({"role": "user", "content": f'Statement: "{statement}"'})
    return msgs


# Match the outermost JSON array in the model output. Non-greedy and
# DOTALL so we tolerate newlines inside. If the model wraps in prose
# like "Here's the JSON: [...]", the regex still finds it.
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_response(raw: str) -> List[Tuple[str, str, str, float]]:
    """Turn the raw model output into structured fact tuples.

    Returns (key, value, category, confidence) per fact. Unparseable
    output yields an empty list (not an exception) — this runs in a
    background thread and should never raise into the learner.
    """
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(items, list):
        return []

    out: List[Tuple[str, str, str, float]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip().lower().replace(" ", "_")
        value = str(item.get("value", "")).strip()
        category = str(item.get("category", "general")).strip().lower()
        try:
            conf = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        if not key or not value or len(key) > 50 or len(value) > 200:
            continue
        # Clip confidence to [0, 1]; anything else is clearly hallucinated.
        conf = max(0.0, min(1.0, conf))
        out.append((key, value, category, conf))
    return out


def extract_facts_llm(
    statement: str,
    *,
    model: Optional[str] = None,
    timeout_s: float = 4.0,
) -> List[Tuple[str, str, str, float]]:
    """Synchronous LLM extraction. Safe to call from a thread.

    Returns (key, value, category, confidence) tuples. Empty list on
    any error — caller should treat extraction as best-effort.
    """
    if not statement or not statement.strip():
        return []
    try:
        import litellm
    except ImportError:
        return []

    model = model or DEFAULT_EXTRACT_MODEL
    messages = _build_prompt(statement.strip())

    try:
        # No streaming — we want the whole JSON before parsing. Tiny
        # max_tokens keeps this fast even when the model is chatty.
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,  # deterministic extraction
            max_tokens=200,
            timeout=timeout_s,
        )
        choice = response.choices[0] if response.choices else None
        raw = (choice.message.content if choice else "") or ""
    except Exception:
        return []

    return _parse_response(raw)


# Canonicalize semantically-equivalent keys that the LLM tends to
# produce alongside the regex form. Without this the same fact ends
# up stored twice under slightly different names
# (employer/current_employer, location/current_location, etc.) and
# memory context bloats. Extend as new duplicates are observed.
_KEY_CANONICAL: dict[str, str] = {
    "current_employer":  "employer",
    "current_job":       "occupation",
    "current_position":  "occupation",
    "job":               "occupation",
    "work":              "employer",
    "company":           "employer",
    "current_location":  "location",
    "current_city":      "location",
    "current_residence": "location",
    "home":              "location",
    "partner_name":      "spouse_name",
    "wife_name":         "spouse_name",
    "husband_name":      "spouse_name",
}


def _canonical_key(k: str) -> str:
    """Map LLM variant keys to their canonical form when one exists."""
    return _KEY_CANONICAL.get(k, k)


def _values_equivalent(a: str, b: str) -> bool:
    """Case- and whitespace-insensitive equality — enough to catch
    the 'employer=google' vs 'current_employer=Google' class of
    duplicate without needing a full semantic comparison."""
    return a.strip().lower() == b.strip().lower()


def merge_with_regex(
    regex_facts: List[Tuple[str, str, str]],
    llm_facts: List[Tuple[str, str, str, float]],
    *,
    min_llm_confidence: float = 0.6,
) -> List[Tuple[str, str, str, str]]:
    """Combine regex and LLM results with regex-wins precedence.

    Returns (key, value, category, source) where source is either
    "inferred" (regex) or "llm_inferred" (LLM).

    Dedup rules, in order:
      1. Regex facts are always included and establish the "canonical"
         keys in `seen` (both their own key and any canonical form).
      2. LLM facts are canonicalized first. If the canonical key is
         already seen, the LLM fact is dropped — this handles the
         common "current_employer" vs "employer" collision.
      3. Low-confidence LLM facts (< min_llm_confidence) are dropped.
    """
    seen_keys: set[str] = set()
    # Track values under each canonical key so a second pattern hitting
    # the same fact (different key, same value) is suppressed too.
    seen_values_by_canon: dict[str, set[str]] = {}

    out: List[Tuple[str, str, str, str]] = []

    for k, v, c in regex_facts:
        canon = _canonical_key(k)
        if k in seen_keys:
            continue
        out.append((k, v, c, "inferred"))
        seen_keys.add(k)
        seen_keys.add(canon)
        seen_values_by_canon.setdefault(canon, set()).add(v.strip().lower())

    for k, v, c, conf in llm_facts:
        if conf < min_llm_confidence:
            continue
        canon = _canonical_key(k)
        if k in seen_keys or canon in seen_keys:
            continue
        existing_values = seen_values_by_canon.get(canon, set())
        if any(_values_equivalent(v, ev) for ev in existing_values):
            continue
        out.append((k, v, c, "llm_inferred"))
        seen_keys.add(k)
        seen_keys.add(canon)
        seen_values_by_canon.setdefault(canon, set()).add(v.strip().lower())
    return out


def extract_facts_async(
    statement: str,
    *,
    on_results: Optional[callable] = None,
    model: Optional[str] = None,
) -> threading.Thread:
    """Fire-and-forget background extraction.

    Starts a daemon thread that calls `extract_facts_llm` and invokes
    `on_results(facts)` on success. The thread handle is returned so
    callers can `.join(timeout)` in tests; in production we ignore it.
    """
    def _run() -> None:
        try:
            facts = extract_facts_llm(statement, model=model)
            if on_results and facts:
                on_results(facts)
        except Exception:
            # Background thread never raises.
            pass

    t = threading.Thread(target=_run, daemon=True, name="llm-fact-extractor")
    t.start()
    return t 
