"""Conversation memory for multi-turn context awareness."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from collections import deque

from mother.llm.drivers import ChatMessage

# Anchor to repo root so persistence works regardless of process cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HISTORY_BASE = _REPO_ROOT / "assistant" / "memory" / "users"


# When history grows past `max_turns`, the oldest `_SUMMARIZE_DROP_TURNS`
# turns are summarized into `summary` and then dropped from the deque.
# Keeping this small (2 turns per compaction) avoids thrashing â€” we
# summarize just enough to make room for the new turn without losing
# recent context abruptly.
_SUMMARIZE_DROP_TURNS = 2

# Character budget for the running summary. Kept small because it lives
# inside the system prompt every turn.
_SUMMARY_CHAR_BUDGET = 500


@dataclass
class ConversationMemory:
    """Maintains a sliding window of conversation history with a
    compacted summary of older turns.

    Active window: last `max_turns` exchanges kept verbatim.
    Older turns: collapsed into a rolling textual `summary` that is
    injected into the system prompt every turn. Ultron remembers
    yesterday without eating the context window."""

    # 20 exchanges (40 messages). The driver now places a MOVING cache
    # breakpoint on the last message (drivers.py), so the conversation
    # prefix genuinely hits Anthropic's prompt cache across turns —
    # widened history costs one cache-read (0.1x price) instead of full
    # re-processing. Twenty exchanges ≈ the last 15-20 minutes verbatim.
    max_turns: int = 20
    _history: deque = field(default_factory=lambda: deque(maxlen=40))
    summary: str = ""  # Compacted history beyond the active window.
    _summarize_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def add_user(self, content: str) -> None:
        """Add a user message to history. Triggers compaction if needed."""
        with self._summarize_lock:
            self._maybe_compact()
            self._history.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Add an assistant response to history."""
        with self._summarize_lock:
            self._history.append(ChatMessage(role="assistant", content=content))

    def get_messages(self, system_prompt: str) -> List[ChatMessage]:
        """Get full message list including system prompt and history.

        If a rolling summary exists, it's appended to the system prompt
        as 'Earlier conversation:' so the model sees long-term context
        without the verbatim transcript eating tokens.
        """
        if self.summary:
            system_prompt = (
                f"{system_prompt}\n\nEarlier conversation (summary): {self.summary}"
            )
        messages = [ChatMessage(role="system", content=system_prompt)]
        messages.extend(self._history)
        return messages

    def get_context_messages(self) -> List[ChatMessage]:
        """Get just the conversation history (no system prompt)."""
        return list(self._history)

    def clear(self) -> None:
        """Clear conversation history and summary."""
        self._history.clear()
        self.summary = ""

    # â”€â”€ Rolling summary compaction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _maybe_compact(self) -> None:
        """If history is at capacity, summarize the oldest turns inline.

        Called before appending a new user turn so we always leave room
        for the new turn + its response. Summarization uses a cheap
        tier 1 call; on failure we silently keep the old summary and
        still drop the oldest turns (privileging recent context over
        perfect memory).
        """
        # Deque is a fixed-maxlen buffer; when at capacity, adding one
        # more message would silently drop the oldest. Compact first.
        if len(self._history) < self._history.maxlen:
            return
        # Pull oldest messages to summarize. Grab pairs (user+assistant)
        # where possible so the summary reads coherently.
        drop_n = min(_SUMMARIZE_DROP_TURNS * 2, len(self._history))
        old = [self._history.popleft() for _ in range(drop_n)]
        # Summarize synchronously-but-bounded. We're already on a
        # background thread in production (voice pipeline calls this
        # via `maybe_learn_from_statement`'s thread context), so a
        # 1-2s summary call is acceptable. If it fails we still keep
        # the existing summary plus drop the old turns.
        try:
            addition = _summarize_turns(old, prior=self.summary)
            if addition:
                self.summary = _merge_summary(self.summary, addition)
        except Exception:
            pass  # drop-turns still happens; summary just doesn't grow

    # â”€â”€ Persistence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def save(self, user_id: str) -> None:
        """Persist the conversation window (and rolling summary) to disk.

        Saved as ``assistant/memory/users/<user_id>/conv_history.json``.
        File format is backwards-compatible: the old format was a plain
        list of messages; the new format is a dict with ``summary`` and
        ``messages`` keys. `load` reads both.
        """
        path = _HISTORY_BASE / user_id / "conv_history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary,
            "messages": [{"role": m.role, "content": m.content} for m in self._history],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, user_id: str) -> None:
        """Restore conversation history for *user_id* from disk (if it exists).

        Reads both the legacy list-of-messages format and the new
        dict-with-summary format.
        """
        path = _HISTORY_BASE / user_id / "conv_history.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._history.clear()
            self.summary = ""
            if isinstance(payload, dict):
                self.summary = str(payload.get("summary", "") or "")
                items = payload.get("messages", [])
            else:
                items = payload  # legacy format
            for item in items:
                role = item.get("role", "user")
                content = item.get("content", "")
                if role and content:
                    self._history.append(ChatMessage(role=role, content=content))
        except Exception:
            pass  # Corrupt history â€” start fresh
    
    def summarize_for_context(self, max_chars: int = 500) -> str:
        """Get a brief summary of recent conversation for context injection.
        
        Useful for including conversation context in RAG queries.
        """
        if not self._history:
            return ""
        
        lines = []
        char_count = 0
        # Work backwards from most recent
        for msg in reversed(self._history):
            role = "User" if msg.role == "user" else "Ultron"
            line = f"{role}: {msg.content[:100]}..."
            if char_count + len(line) > max_chars:
                break
            lines.insert(0, line)
            char_count += len(line)
        
        return "\n".join(lines)
    
    @property
    def last_user_input(self) -> Optional[str]:
        """Get the most recent user input."""
        for msg in reversed(self._history):
            if msg.role == "user":
                return msg.content
        return None
    
    @property
    def last_response(self) -> Optional[str]:
        """Get the most recent assistant response."""
        for msg in reversed(self._history):
            if msg.role == "assistant":
                return msg.content
        return None
    
    @property
    def turn_count(self) -> int:
        """Number of complete turns (user + assistant pairs)."""
        users = sum(1 for m in self._history if m.role == "user")
        assistants = sum(1 for m in self._history if m.role == "assistant")
        return min(users, assistants)


# Global conversation memory instance
_global_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Get or create the global conversation memory."""
    global _global_memory
    if _global_memory is None:
        _global_memory = ConversationMemory()
    return _global_memory


def reset_memory() -> None:
    """Reset the global conversation memory."""
    global _global_memory
    _global_memory = ConversationMemory()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Summary helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Kept at module level so `ConversationMemory` stays small and easy to
# test. Both helpers are safe to call with no LLM configured: they fall
# back to a keyword-extractive summary so the feature degrades
# gracefully rather than blocking the pipeline.

def _summarize_turns(turns: List[ChatMessage], *, prior: str = "") -> str:
    """Condense a list of (user, assistant) messages into 1-2 sentences.

    Uses Cerebras tier 1 for speed (~200-500ms). Falls back to a trivial
    head-of-text extract if the model is unavailable.
    """
    if not turns:
        return ""

    # Render the turns in a compact transcript the model can compress.
    lines = []
    for m in turns:
        who = "User" if m.role == "user" else "Ultron"
        content = (m.content or "").strip().replace("\n", " ")
        if len(content) > 300:
            content = content[:300] + "â€¦"
        lines.append(f"{who}: {content}")
    transcript = "\n".join(lines)

    system = (
        "You are a dry, factual log compressor. Summarize a dialog in ONE "
        "third-person sentence, maximum 25 words. Record only what was "
        "literally said: the user's topic or question, and the factual "
        "answer given. "
        "DO NOT editorialize. DO NOT describe the assistant's personality, "
        "superiority, emotions, or self-assessment. DO NOT dramatize. "
        "DO NOT invent or embellish. If the conversation was banal, the "
        "summary should be banal. Output the sentence only, no prefix."
    )
    user_msg = (
        f"Existing summary (if any): {prior or '(none)'}\n\n"
        f"New turns to add:\n{transcript}\n\n"
        "Concise summary update:"
    )
    try:
        import litellm
        import os
        model = os.environ.get("ULTRON_SUMMARY_MODEL", "cerebras/gemma-4-31b")
        r = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=80,
            timeout=6.0,
        )
        out = (r.choices[0].message.content or "").strip()
        # Strip leading labels like "Summary: " if the model adds them
        for prefix in ("summary:", "summary update:", "update:"):
            if out.lower().startswith(prefix):
                out = out[len(prefix):].strip()
        return out
    except Exception:
        # Degrade to an extractive summary: first sentence of each turn
        snippets = []
        for m in turns:
            content = (m.content or "").strip()
            if not content:
                continue
            first = content.split(".")[0][:80]
            snippets.append(first)
        return "; ".join(snippets)[:_SUMMARY_CHAR_BUDGET]


def _recompress_summary(combined: str) -> str:
    """Ask the LLM to re-summarize a bloated summary into one sentence.

    Called from _merge_summary when prior + addition overflows the
    budget. Falls back to a lossy tail-keep truncation if the LLM
    is unavailable so we never block compaction."""
    system = (
        "You compress a rambling dialog summary into a single clean "
        "third-person sentence, no more than 40 words, preserving the "
        "most important facts and decisions. Drop small talk. Drop "
        "repetition. Output only the sentence."
    )
    try:
        import litellm
        import os
        model = os.environ.get("ULTRON_SUMMARY_MODEL", "cerebras/gemma-4-31b")
        r = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": combined},
            ],
            temperature=0.1,
            max_tokens=100,
            timeout=6.0,
        )
        out = (r.choices[0].message.content or "").strip()
        for prefix in ("summary:", "condensed summary:", "compressed:"):
            if out.lower().startswith(prefix):
                out = out[len(prefix):].strip()
        if out:
            return out[:_SUMMARY_CHAR_BUDGET]
    except Exception:
        pass
    # Fallback: keep the newest tail. Better than keeping the oldest.
    overflow = len(combined) - _SUMMARY_CHAR_BUDGET
    return "â€¦" + combined[overflow + 1:]


def _merge_summary(prior: str, addition: str) -> str:
    """Combine the old summary with a newly-compacted chunk.

    When combined length is under the budget, just concat. When it
    overflows, re-summarize the whole thing via the LLM so the result
    reads like a coherent paragraph instead of a repetitive
    concatenation of chunks. The re-summarize call itself is bounded
    (timeout + graceful degrade path inside `_recompress_summary`),
    so compaction can't wedge the conversation even if the network
    drops."""
    if not prior:
        return addition[:_SUMMARY_CHAR_BUDGET]
    if not addition:
        return prior
    combined = f"{prior} {addition}".strip()
    if len(combined) <= _SUMMARY_CHAR_BUDGET:
        return combined
    return _recompress_summary(combined)

