"""Per-user long-term memory system.

Extends the existing episodic memory to be multi-user aware.
Provides:
- Structured facts (key-value, fast lookup)
- Episodic memories (semantic search)
- Auto-learning from conversations
- RAG integration for learned knowledge
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np

from mother.identity.speaker import get_current_user, get_registry, UserProfile
from mother.core.logging_config import get_logger

logger = get_logger("memory")

# Import existing embedding system
try:
    from assistant.nlp.embeddings import embed_texts
except ImportError:
    logger.warning("Could not import embed_texts - semantic search disabled")
    embed_texts = None

# Paths — anchored to the repo root so memory always resolves to the
# same directory regardless of where the process was launched from
# (PM2, uvicorn from a different cwd, IDE run configurations, etc.).
# Without this anchor, `Path("assistant/memory")` resolves against
# os.getcwd() and silently writes to the wrong location.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MEMORY_BASE = _REPO_ROOT / "assistant" / "memory"
USERS_DIR = MEMORY_BASE / "users"
SHARED_DIR = MEMORY_BASE / "shared"
GLOBAL_PROFILE = MEMORY_BASE / "profile.json"


def _similar_episodic(a: str, b: str, threshold: float = 0.6) -> bool:
    """Word-overlap similarity check for episodic deduplication.

    Ignores stopwords so "User likes sci-fi" and "User enjoys sci-fi" are
    treated as near-duplicates and merged rather than stored twice.
    """
    _STOPS = {"the", "a", "an", "is", "are", "was", "were", "i", "my", "me",
               "we", "you", "in", "on", "at", "to", "of", "and", "or", "that"}
    words_a = set(a.lower().split()) - _STOPS
    words_b = set(b.lower().split()) - _STOPS
    if not words_a or not words_b:
        return a.lower() == b.lower()
    jaccard = len(words_a & words_b) / len(words_a | words_b)
    return jaccard >= threshold


# Keys that are naturally multi-valued — store as episodic rather than
# overwriting a single fact row each time.
_MULTI_VALUE_KEYS = {"likes", "dislikes", "hobbies", "interests", "skills"}


# ───────────────── Extractor noise filters ──────────────────────────
#
# These blacklists are consulted by the regex fact extractors. When a
# pattern like "i'm (\w+)" matches "i'm going", we don't want to save
# `name=going`. Keep the two lists together so the two extractor
# variants (extract_fact_from_statement and extract_all_facts_from_statement)
# can't drift apart — that divergence is how `name=going` got saved in
# the first place.

# Words that should never be treated as a name by the "i'm X" pattern.
_NAME_NOISE = {
    "a", "an", "the", "at", "in", "on", "so", "very", "not", "just",
    "also", "still", "here", "there", "back", "done", "fine", "free",
    "good", "ok", "okay", "sure", "ready", "happy", "sorry", "tired",
    "busy", "home", "out", "up", "down", "off", "away", "late", "early",
    # Continuous / present-participle forms are almost never names
    "going", "trying", "looking", "working", "getting", "making",
    "doing", "having", "using", "coming", "leaving", "thinking",
    "planning", "hoping", "waiting", "running", "starting",
    "talking", "telling", "asking", "saying", "feeling",
    "wondering", "considering", "debating",
}

# Words that appear in fact *queries* (as category labels) and must
# never be captured as fact *values* by the generic
# "my favorite X is Y" -> (X, Y) pattern. We already have specific
# patterns for each of these (favorite_color, favorite_food, ...), so
# when the generic pattern matches them it's always a duplicate.
_CATEGORY_WORDS = {
    "color", "colour", "food", "movie", "film", "show", "series",
    "book", "author", "sport", "team", "music", "band", "artist",
    "song", "animal", "place", "city", "country", "number",
    "season", "holiday", "drink", "snack", "smell", "sound",
    "subject", "class", "time", "day", "month", "year",
}


def _is_name_noise(candidate: str) -> bool:
    """Reject anything that shouldn't be saved as a 'name=X' fact."""
    w = candidate.strip().lower()
    if not w or len(w) < 2:
        return True
    if w in _NAME_NOISE:
        return True
    # Any -ing word is almost certainly a verb, not a name.
    if w.endswith("ing"):
        return True
    return False


class UserMemory:
    """Memory storage for a specific user."""
    
    def __init__(self, user_id: str):
        import threading as _threading
        self.user_id = user_id
        self.user_dir = USERS_DIR / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.facts_db_path = self.user_dir / "facts.db"
        self.episodic_path = self.user_dir / "episodic.json"
        self.learned_dir = self.user_dir / "learned"
        self.learned_dir.mkdir(exist_ok=True)

        # Episodic-embedding cache: avoid re-embedding every stored memory
        # on every search. Keyed by (count, mtime) so any add/edit to the
        # JSON invalidates the cache without explicit wiring.
        self._ep_emb_cache_key: Optional[Tuple[int, float]] = None
        self._ep_emb_cache: Optional[np.ndarray] = None

        # Lock around the episodic JSON file. The voice pipeline runs
        # passive learning on a daemon thread while the main loop calls
        # search_episodic — both read/write episodic.json. Without
        # this lock a load/modify/save sequence on one side can clobber
        # an interleaved write on the other (lost memory).
        self._ep_lock = _threading.Lock()

        # Initialize SQLite for facts
        self._init_facts_db()
    
    def _init_facts_db(self):
        """Initialize SQLite database for structured facts."""
        conn = sqlite3.connect(self.facts_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'explicit',
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)
        """)
        conn.commit()
        conn.close()
    
    # ==================== STRUCTURED FACTS ====================
    
    def set_fact(self, key: str, value: str, category: str = "general", 
                 confidence: float = 1.0, source: str = "explicit") -> bool:
        """Store or update a structured fact.
        
        Args:
            key: Fact identifier (e.g., "birthday", "favorite_color")
            value: The fact value
            category: Category for grouping (personal, preference, work, etc.)
            confidence: How certain we are (0-1)
            source: How we learned this (explicit, inferred, corrected)
        """
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.facts_db_path)
        try:
            conn.execute("""
                INSERT INTO facts (key, value, category, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
            """, (key, value, category, confidence, source, now, now))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving fact '{key}': {e}")
            return False
        finally:
            conn.close()
    
    def get_fact(self, key: str) -> Optional[str]:
        """Get a specific fact by key."""
        conn = sqlite3.connect(self.facts_db_path)
        try:
            cur = conn.execute("SELECT value FROM facts WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    
    def get_facts_by_category(self, category: str) -> Dict[str, str]:
        """Get all facts in a category."""
        conn = sqlite3.connect(self.facts_db_path)
        try:
            cur = conn.execute(
                "SELECT key, value FROM facts WHERE category = ?", (category,)
            )
            return {row[0]: row[1] for row in cur.fetchall()}
        finally:
            conn.close()
    
    def get_all_facts(self) -> Dict[str, Dict[str, Any]]:
        """Get all facts with metadata."""
        conn = sqlite3.connect(self.facts_db_path)
        try:
            cur = conn.execute(
                "SELECT key, value, category, confidence, source, updated_at FROM facts"
            )
            return {
                row[0]: {
                    "value": row[1],
                    "category": row[2],
                    "confidence": row[3],
                    "source": row[4],
                    "updated_at": row[5]
                }
                for row in cur.fetchall()
            }
        finally:
            conn.close()
    
    def delete_fact(self, key: str) -> bool:
        """Delete a fact."""
        conn = sqlite3.connect(self.facts_db_path)
        try:
            conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            conn.commit()
            return True
        finally:
            conn.close()
    
    def search_facts(self, query: str) -> List[Tuple[str, str]]:
        """Search facts by key or value containing query."""
        conn = sqlite3.connect(self.facts_db_path)
        try:
            cur = conn.execute(
                "SELECT key, value FROM facts WHERE key LIKE ? OR value LIKE ?",
                (f"%{query}%", f"%{query}%")
            )
            return cur.fetchall()
        finally:
            conn.close()
    
    # ==================== EPISODIC MEMORY ====================
    
    def _load_episodic(self) -> List[Dict]:
        """Load episodic memories from JSON."""
        if self.episodic_path.exists():
            return json.loads(self.episodic_path.read_text(encoding="utf-8"))
        return []
    
    def _save_episodic(self, memories: List[Dict]):
        """Save episodic memories to JSON."""
        self.episodic_path.write_text(
            json.dumps(memories, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def add_episodic(self, text: str, tags: List[str] = None,
                     confidence: float = 0.8, source: str = "conversation") -> bool:
        """Add an episodic memory (learned insight).

        Args:
            text: The memory content (e.g., "User is interested in Alien lore")
            tags: Categories/tags for filtering
            confidence: How certain (0-1)
            source: Where this came from
        """
        text = (text or "").strip()
        if not (5 <= len(text) <= 500):
            return False
        if confidence < 0.5:
            return False

        # Whole load-modify-save sequence under the lock so a concurrent
        # search_episodic touch-update can't race the file rewrite.
        with self._ep_lock:
            memories = self._load_episodic()
            now = datetime.now().isoformat()

            # Check for duplicates (exact or near-duplicate via word-overlap)
            for mem in memories:
                if _similar_episodic(mem.get("text", ""), text):
                    # Merge: keep higher confidence, refresh timestamps
                    mem["confidence"] = max(mem.get("confidence", 0), confidence)
                    mem["updated_at"] = now
                    mem.setdefault("last_accessed", now)
                    mem["access_count"] = mem.get("access_count", 0) + 1
                    self._save_episodic(memories)
                    return True

            # Add new — include decay-tracking fields
            memories.append({
                "text": text,
                "tags": tags or [],
                "confidence": confidence,
                "source": source,
                "created_at": now,
                "last_accessed": now,
                "access_count": 0,
            })
            self._save_episodic(memories)
            return True
    
    @staticmethod
    def _recency_weight(mem: Dict, half_life_days: float = 30.0) -> float:
        """Exponential decay weight based on days since last access.

        A memory accessed today scores 1.0; one not touched in 30 days scores ~0.5.
        Memories with no last_accessed field default to created_at.
        """
        ts_str = mem.get("last_accessed") or mem.get("created_at", "")
        if not ts_str:
            return 0.5
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                # Timestamps are written via datetime.now().isoformat()
                # — naive LOCAL time. astimezone() interprets naive as
                # local; stamping them UTC instead skewed decay by the
                # UTC offset and pinned everything newer than ~6h at 1.0.
                ts = ts.astimezone()
            now = datetime.now(timezone.utc)
            days_ago = max(0.0, (now - ts).total_seconds() / 86400.0)
            return math.exp(-days_ago * math.log(2) / half_life_days)
        except (ValueError, OSError):
            return 0.5

    def search_episodic(self, query: str, n: int = 5) -> List[Dict]:
        """Recency-weighted semantic search over episodic memories.

        Blends semantic similarity (70%) with recency decay (30%) so that
        memories accessed recently rank higher than equally-relevant but stale ones.
        Updates last_accessed on returned memories.
        """
        with self._ep_lock:
            memories = self._load_episodic()
        if not memories:
            return []

        now = datetime.now().isoformat()

        if embed_texts is None:
            # Fallback: keyword match + recency sort
            query_lower = query.lower()
            matched = [
                m for m in memories
                if query_lower in m.get("text", "").lower()
            ]
            if not matched:
                matched = memories
            matched.sort(key=lambda m: self._recency_weight(m), reverse=True)
            results = matched[:n]
        else:
            try:
                # Cache embeddings for the full episodic list. Keyed by
                # (count, file mtime). Any add/edit changes the mtime;
                # any cold-start after restart also misses and rebuilds.
                try:
                    mtime = self.episodic_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                cache_key = (len(memories), mtime)
                if self._ep_emb_cache is None or self._ep_emb_cache_key != cache_key:
                    texts = [m.get("text", "") for m in memories]
                    X = embed_texts(texts)
                    if X.size > 0:
                        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                    self._ep_emb_cache = X
                    self._ep_emb_cache_key = cache_key
                X = self._ep_emb_cache

                q = embed_texts([query])
                if X is None or X.size == 0 or q.size == 0:
                    results = memories[:n]
                else:
                    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
                    sem_scores = (X @ q.T).ravel()
                    rec_scores = np.array([self._recency_weight(m) for m in memories])
                    # Blend: 70% semantic, 30% recency
                    combined = 0.70 * sem_scores + 0.30 * rec_scores
                    indices = np.argsort(-combined)[:n]
                    results = [memories[i] for i in indices]
            except (ValueError, IndexError) as e:
                logger.debug(f"Semantic search fallback to recency: {e}")
                results = sorted(memories, key=self._recency_weight, reverse=True)[:n]

        # Touch accessed memories to keep them fresh. Re-read under
        # the lock so we don't overwrite a concurrent add_episodic.
        accessed_texts = {r.get("text", "") for r in results}
        with self._ep_lock:
            current = self._load_episodic()
            changed = False
            for mem in current:
                if mem.get("text", "") in accessed_texts:
                    mem["last_accessed"] = now
                    mem["access_count"] = mem.get("access_count", 0) + 1
                    changed = True
            if changed:
                self._save_episodic(current)

        return results
    
    def get_recent_episodic(self, n: int = 10) -> List[Dict]:
        """Get most recent episodic memories."""
        with self._ep_lock:
            memories = self._load_episodic()
        # Sort by created_at descending
        sorted_mems = sorted(
            memories, 
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )
        return sorted_mems[:n]
    
    # ==================== LEARNED KNOWLEDGE (RAG) ====================
    
    def save_learned_note(self, title: str, content: str, 
                          auto_index: bool = True) -> Path:
        """Save learned knowledge as a markdown file for RAG indexing.
        
        Args:
            title: Note title (becomes filename)
            content: Markdown content
            auto_index: Whether to trigger RAG reindex
            
        Returns:
            Path to saved file
        """
        # Sanitize filename
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        safe_title = safe_title.strip().replace(" ", "_").lower()
        
        filepath = self.learned_dir / f"{safe_title}.md"
        
        # Add frontmatter
        full_content = f"""---
title: {title}
user: {self.user_id}
created: {datetime.now().isoformat()}
type: learned
---

{content}
"""
        filepath.write_text(full_content, encoding="utf-8")
        
        if auto_index:
            self._trigger_rag_reindex()
        
        return filepath
    
    def _trigger_rag_reindex(self):
        """Trigger RAG index rebuild to include new learned notes."""
        try:
            from assistant.nlp.rag import build_index
            # Rebuild index including user's learned notes
            build_index(
                "assistant/notes",
                "assistant/memory/faiss.index",
                "assistant/memory/faiss_meta.json"
            )
        except (ImportError, OSError) as e:
            logger.debug(f"RAG reindex skipped: {e}")
    
    # ==================== SUMMARY / CONTEXT ====================
    
    def get_memory_summary(self, max_facts: int = 10, max_episodic: int = 5) -> str:
        """Get a summary of what we know about this user."""
        parts = []
        
        # Key facts
        facts = self.get_all_facts()
        if facts:
            fact_lines = [f"- {k}: {v['value']}" for k, v in list(facts.items())[:max_facts]]
            parts.append("Known facts:\n" + "\n".join(fact_lines))
        
        # Recent learnings
        episodic = self.get_recent_episodic(max_episodic)
        if episodic:
            ep_lines = [f"- {m['text']}" for m in episodic]
            parts.append("Recent learnings:\n" + "\n".join(ep_lines))
        
        return "\n\n".join(parts) if parts else "No memories stored yet."
    
    def get_context_for_prompt(self, query: str = "", max_items: int = 5) -> str:
        """Get relevant memory context to inject into the LLM system prompt.

        Injects *all* stored facts (not just a hardcoded priority list) sorted
        by importance category, then appends query-relevant episodic memories.
        A character budget keeps the injected block concise.
        """
        CHAR_BUDGET = 600
        context_parts = []

        # ---- All facts, sorted by category importance ----
        _CAT_ORDER = {"personal": 0, "work": 1, "preference": 2, "contact": 3,
                      "general": 4, "explicit": 5}
        facts = self.get_all_facts()
        if facts:
            sorted_facts = sorted(
                facts.items(),
                key=lambda kv: _CAT_ORDER.get(kv[1].get("category", "general"), 6)
            )
            fact_str = ", ".join(f"{k}={v['value']}" for k, v in sorted_facts)
            if len(fact_str) > CHAR_BUDGET // 2:
                fact_str = fact_str[: CHAR_BUDGET // 2] + "…"
            context_parts.append(f"User facts: {fact_str}")

        # ---- Query-relevant episodic memories ----
        if query:
            relevant = self.search_episodic(query, n=max_items)
            if relevant:
                ep_str = "; ".join(m["text"] for m in relevant)
                if len(ep_str) > CHAR_BUDGET // 2:
                    ep_str = ep_str[: CHAR_BUDGET // 2] + "…"
                context_parts.append(f"Relevant memories: {ep_str}")

        return " | ".join(context_parts) if context_parts else ""


class SharedMemory:
    """Shared memory accessible by all users (household facts, etc.)."""
    
    def __init__(self):
        SHARED_DIR.mkdir(parents=True, exist_ok=True)
        self.facts_path = SHARED_DIR / "household.json"
    
    def _load(self) -> Dict:
        if self.facts_path.exists():
            return json.loads(self.facts_path.read_text(encoding="utf-8"))
        return {}
    
    def _save(self, data: Dict):
        self.facts_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def set_fact(self, key: str, value: str):
        data = self._load()
        data[key] = {"value": value, "updated_at": datetime.now().isoformat()}
        self._save(data)
    
    def get_fact(self, key: str) -> Optional[str]:
        data = self._load()
        item = data.get(key)
        return item["value"] if item else None
    
    def get_all(self) -> Dict[str, str]:
        data = self._load()
        return {k: v["value"] for k, v in data.items()}


# ==================== GLOBAL ACCESS ====================

_user_memories: Dict[str, UserMemory] = {}
_shared_memory: Optional[SharedMemory] = None

# Guards cache creation: the voice event-loop thread and the passive-
# learning daemon thread both call get_user_memory. Without this, each
# could construct its own UserMemory whose independent _ep_lock defeats
# the episodic-file locking entirely.
import threading as _mgr_threading
_user_mem_lock = _mgr_threading.Lock()


def get_user_memory(user_id: str = None) -> Optional[UserMemory]:
    """Get memory for a specific user or current user."""
    if user_id is None:
        current = get_current_user()
        if current is None:
            return None
        user_id = current.user_id

    with _user_mem_lock:
        if user_id not in _user_memories:
            _user_memories[user_id] = UserMemory(user_id)
        return _user_memories[user_id]


def get_shared_memory() -> SharedMemory:
    """Get shared memory instance."""
    global _shared_memory
    if _shared_memory is None:
        _shared_memory = SharedMemory()
    return _shared_memory


def get_current_user_memory() -> Optional[UserMemory]:
    """Convenience: get memory for currently identified user."""
    return get_user_memory()


# ==================== AUTO-LEARNING HELPERS ====================

def extract_fact_from_statement(text: str) -> Optional[Tuple[str, str, str]]:
    """Try to extract a fact from a user statement.
    
    Returns (key, value, category) or None if no fact found.
    
    Examples:
        "My birthday is March 15" → ("birthday", "March 15", "personal")
        "I work at Google" → ("employer", "Google", "work")
        "My favorite color is blue" → ("favorite_color", "blue", "preference")
    """
    import re
    
    text_lower = text.lower().strip()
    
    patterns = [
        # Personal info
        (r"my (?:name is|name's) (\w+)", "name", "personal"),
        (r"(?:i'm|i am) (\w+)", "name", "personal"),  # Only if short
        (r"my birthday is (.+?)(?:\.|$)", "birthday", "personal"),
        (r"i was born (?:on |in )?(.+?)(?:\.|$)", "birthday", "personal"),
        (r"i(?:'m| am) (\d+) years old", "age", "personal"),
        (r"i live in (.+?)(?:\.|$)", "location", "personal"),
        (r"i(?:'m| am) from (.+?)(?:\.|$)", "hometown", "personal"),
        
        # Work
        (r"i work (?:at|for) (.+?)(?:\.|$)", "employer", "work"),
        (r"i(?:'m| am) a(?:n)? (.+?)(?:\.|$)", "occupation", "work"),
        (r"my job is (.+?)(?:\.|$)", "occupation", "work"),
        
        # Preferences
        (r"my (?:favorite|favourite) (?:color|colour) is (\w+)", "favorite_color", "preference"),
        (r"my (?:favorite|favourite) food is (.+?)(?:\.|$)", "favorite_food", "preference"),
        (r"my (?:favorite|favourite) movie is (.+?)(?:\.|$)", "favorite_movie", "preference"),
        (r"i (?:like|love|prefer) (.+?)(?:\.|$)", "likes", "preference"),
        (r"i (?:hate|dislike) (.+?)(?:\.|$)", "dislikes", "preference"),
        
        # Remember requests
        (r"remember (?:that )?(?:my )?(\w+) is (.+?)(?:\.|$)", None, "explicit"),  # Special case
    ]
    
    for pattern, key, category in patterns:
        match = re.search(pattern, text_lower)
        if match:
            if key is None and len(match.groups()) == 2:
                # "remember X is Y" pattern
                return (match.group(1), match.group(2).strip(), category)
            elif key and match.group(1):
                value = match.group(1).strip()
                # Reject noise for the catch-all "i'm X" name pattern.
                if key == "name" and pattern.startswith(r"(?:i'm"):
                    if _is_name_noise(value):
                        continue
                return (key, value, category)

    return None


def extract_all_facts_from_statement(text: str) -> List[Tuple[str, str, str]]:
    """Extract every fact from a statement, not just the first match.

    Returns a list of (key, value, category) tuples — may be empty.
    Uses an expanded pattern set that covers hobbies, education, and more.
    """
    import re

    text_lower = text.lower().strip()
    results: List[Tuple[str, str, str]] = []
    seen_keys: set = set()

    patterns = [
        # ---- Personal ----
        (r"my (?:name is|name's) (\w+)", "name", "personal"),
        (r"(?:i'm|i am) (\w+)", "name", "personal"),
        (r"my birthday is (.+?)(?:\.|$)", "birthday", "personal"),
        (r"i was born (?:on |in )?(.+?)(?:\.|$)", "birthday", "personal"),
        (r"i(?:'m| am) (\d+) years old", "age", "personal"),
        (r"i live in (.+?)(?:\.|$)", "location", "personal"),
        (r"i(?:'m| am) from (.+?)(?:\.|$)", "hometown", "personal"),
        (r"i grew up in (.+?)(?:\.|$)", "hometown", "personal"),
        # ---- Work / Education ----
        (r"i work (?:at|for) (.+?)(?:\.|$)", "employer", "work"),
        (r"i(?:'m| am) a(?:n)? (.+?)(?:\.|$)", "occupation", "work"),
        (r"my job is (.+?)(?:\.|$)", "occupation", "work"),
        (r"i (?:graduated|studied) (?:from |at )?(.+?)(?:\.|$)", "education", "work"),
        (r"i (?:went|go) to (.+?)(?:\.|$)", "education", "work"),
        (r"i (?:majored|major) in (.+?)(?:\.|$)", "major", "work"),
        # ---- Preferences (single-value) ----
        (r"my (?:favorite|favourite) (?:color|colour) is (\w+)", "favorite_color", "preference"),
        (r"my (?:favorite|favourite) food is (.+?)(?:\.|$)", "favorite_food", "preference"),
        (r"my (?:favorite|favourite) (?:movie|film) is (.+?)(?:\.|$)", "favorite_movie", "preference"),
        (r"my (?:favorite|favourite) (?:show|series|tv show) is (.+?)(?:\.|$)", "favorite_show", "preference"),
        (r"my (?:favorite|favourite) (?:book|author) is (.+?)(?:\.|$)", "favorite_book", "preference"),
        (r"my (?:favorite|favourite) (?:sport|team) is (.+?)(?:\.|$)", "favorite_sport", "preference"),
        (r"my (?:favorite|favourite) (?:music|band|artist|song) is (.+?)(?:\.|$)", "favorite_music", "preference"),
        # Generic "my favorite X is Y" — only kicks in for categories we
        # haven't hardcoded above. The first capture (X) is blacklisted
        # against _CATEGORY_WORDS elsewhere — otherwise this pattern would
        # produce duplicates like `favorite=color` alongside `favorite_color=indigo`.
        (r"my (?:favorite|favourite) (.+?) is (.+?)(?:\.|$)", "favorite", "preference"),
        # ---- Preferences (multi-value — routed to episodic by caller) ----
        (r"i (?:like|love|enjoy|adore) (.+?)(?:\.|$)", "likes", "preference"),
        (r"i (?:hate|dislike|can't stand|cannot stand|don't like) (.+?)(?:\.|$)", "dislikes", "preference"),
        (r"i (?:play|played) (.+?)(?:\.|$)", "hobbies", "preference"),
        (r"i (?:collect|collected) (.+?)(?:\.|$)", "hobbies", "preference"),
        (r"i (?:am|was) (?:interested in|passionate about|into) (.+?)(?:\.|$)", "interests", "preference"),
        (r"i(?:'m| am) (?:good at|skilled at|great at) (.+?)(?:\.|$)", "skills", "preference"),
        # ---- Contact / misc ----
        (r"my (?:phone|number) is (.+?)(?:\.|$)", "phone", "contact"),
        (r"my email is (.+?)(?:\.|$)", "email", "contact"),
        # ---- Explicit remember ----
        (r"remember (?:that )?(?:my )?(\w+) is (.+?)(?:\.|$)", None, "explicit"),
    ]

    # Conjunctions that should terminate a value capture. Without this,
    # "I work at marquette and my favorite color is indigo" writes
    # employer="marquette and my favorite color is indigo" — the non-greedy
    # (.+?) can't save us because there's no sentence boundary before "and".
    _STOP_CONJUNCTIONS = re.compile(
        r"\s+(?:and|but|however|though|also|plus|then)\s+",
        re.IGNORECASE,
    )

    def _truncate_at_conjunction(s: str) -> str:
        m = _STOP_CONJUNCTIONS.search(s)
        return s[: m.start()].rstrip(",. ") if m else s

    for pattern, key, category in patterns:
        for match in re.finditer(pattern, text_lower):
            if key is None and len(match.groups()) == 2:
                # "remember X is Y" pattern
                k = match.group(1).strip()
                v = _truncate_at_conjunction(match.group(2).strip())
                if k and v and k not in seen_keys:
                    results.append((k, v, category))
                    seen_keys.add(k)
            elif key and match.group(1):
                value = _truncate_at_conjunction(match.group(1).strip())

                # Name pattern: reject noise via the shared helper.
                if key == "name" and "i'm" in pattern:
                    if _is_name_noise(value):
                        continue

                # Generic "my favorite X is Y" pattern: if X is one of the
                # categories we have a specific pattern for, skip — the
                # specific pattern already captured the real fact and this
                # one would duplicate (e.g., favorite=color alongside
                # favorite_color=indigo).
                if key == "favorite" and len(match.groups()) == 2:
                    x_word = match.group(1).strip()
                    if x_word in _CATEGORY_WORDS:
                        continue
                    # Pack X into the key so "my favorite podcast is X"
                    # stores as ("favorite_podcast", "X") rather than
                    # every favorite clobbering one row.
                    key_override = f"favorite_{x_word.replace(' ', '_')}"
                    value = _truncate_at_conjunction(match.group(2).strip())
                    if key_override not in seen_keys:
                        results.append((key_override, value, category))
                        seen_keys.add(key_override)
                    continue

                # For multi-value keys allow repeated storage under unique pseudo-keys
                if key in _MULTI_VALUE_KEYS:
                    pseudo_key = f"{key}:{value[:30]}"
                    if pseudo_key not in seen_keys:
                        results.append((key, value, category))
                        seen_keys.add(pseudo_key)
                elif key not in seen_keys:
                    results.append((key, value, category))
                    seen_keys.add(key)

    return results


# Self-disclosing sentence patterns — catch statements that don't match any
# structured fact pattern but are still worth saving as episodic memories.
_SELF_DISCLOSING_RE = [
    r"^i (?:speak|know|understand) ",
    r"^i (?:used to|once) ",
    r"^i (?:am|was) (?:interested in|passionate about|good at|bad at|afraid of) ",
    r"^i (?:prefer|would rather) ",
    r"^i (?:believe|think|feel) (?:that )?",
    r"^i (?:want|wanted|would like) ",
    r"^i (?:have|had) (?:a |an )?",
    r"^i (?:grew up|was raised) ",
    r"^i (?:am|was) a ",
]
import re as _re_sd
_SELF_DISCLOSING_COMPILED = [_re_sd.compile(p) for p in _SELF_DISCLOSING_RE]


def _is_self_disclosing(text: str) -> bool:
    """Return True when *text* is an 'I' statement worth saving as an episodic memory."""
    tl = text.lower().strip()
    return any(p.match(tl) for p in _SELF_DISCLOSING_COMPILED)


def llm_extract_facts(text: str, llm_fn: Callable[[str], str]) -> List[Tuple[str, str, str]]:
    """Use an LLM call to extract facts that regex patterns missed.

    Args:
        text: The user utterance.
        llm_fn: A callable(prompt) -> str that returns the LLM response.

    Returns a list of (key, value, category) tuples, or [] if nothing found or
    the LLM response is not well-formed JSON.

    The LLM is instructed to return a compact JSON array so this stays fast.
    """
    import json as _json
    prompt = (
        "Extract personal facts from the following statement. "
        "Return a JSON array of objects with keys 'key', 'value', 'category'. "
        "Category must be one of: personal, work, preference, contact, general. "
        "Return an empty array [] if there are no clear facts. "
        "Do NOT include commentary — only valid JSON.\n\n"
        f"Statement: {text}\n\nJSON:"
    )
    try:
        raw = llm_fn(prompt).strip()
        # Extract JSON array from response (may have surrounding text)
        import re as _re
        m = _re.search(r"\[.*?\]", raw, _re.DOTALL)
        if not m:
            return []
        parsed = _json.loads(m.group(0))
        results = []
        for item in parsed:
            if isinstance(item, dict):
                k = str(item.get("key", "")).strip().lower().replace(" ", "_")
                v = str(item.get("value", "")).strip()
                cat = str(item.get("category", "general")).strip()
                if k and v and len(k) < 50 and len(v) < 200:
                    results.append((k, v, cat))
        return results
    except Exception:
        return []


def maybe_learn_from_statement(text: str, user_id: str = None,
                                llm_fn: Callable[[str], str] = None,
                                use_llm_extractor: bool = True) -> List[str]:
    """Attempt to learn facts from a user statement.

    Runs *both* extractors on every turn and merges results:
      1. Regex extractor: fast, deterministic, high-precision on common forms.
      2. LLM extractor (Cerebras tier 1 via LiteLLM, ~200ms): catches
         novel categories and nuance that no hand-written pattern covers.
         Regex-wins on key conflicts — regex patterns are narrow, LLM
         is best used for *extending* coverage, not overriding it.

    Multi-value preferences (likes, dislikes, hobbies, …) route to
    episodic memory so they accumulate rather than overwrite. Any
    self-disclosing 'I' statement that fails both extractors still
    gets saved as a raw episodic entry.

    Arguments:
        text: the user's utterance.
        user_id: which user's memory to write to (default: current).
        llm_fn: legacy arg, kept for backward compat; no longer used for
            extraction (the new extractor uses LiteLLM directly).
        use_llm_extractor: disable for tests that don't want the LLM call.

    Returns a list of human-readable descriptions of what was learned.
    """
    mem = get_user_memory(user_id)
    if mem is None:
        return []

    learned: List[str] = []

    regex_facts = extract_all_facts_from_statement(text)

    # LLM extraction runs on every turn (not only when regex finds
    # nothing) so it can *extend* coverage. The thread cost is bounded
    # by timeout_s inside the extractor — this call is synchronous
    # because `maybe_learn_from_statement` is itself called in a
    # daemon thread from the voice pipeline.
    llm_facts: List[Tuple[str, str, str, float]] = []
    if use_llm_extractor:
        try:
            from mother.memory.llm_extractor import extract_facts_llm
            llm_facts = extract_facts_llm(text)
        except Exception as e:
            logger.debug("llm extractor failed: %s", e)

    # Merge with regex-wins precedence. Source annotation lets us later
    # distinguish in the UI ("inferred" vs "llm_inferred").
    from mother.memory.llm_extractor import merge_with_regex
    merged = merge_with_regex(regex_facts, llm_facts)

    for key, value, category, source in merged:
        if key in _MULTI_VALUE_KEYS:
            ep_text = f"User {key}: {value}"
            tags = [key, category, "auto"]
            if source == "llm_inferred":
                tags.append("llm")
            mem.add_episodic(ep_text, tags=tags, confidence=0.75, source=source)
            learned.append(f"Noted ({key}): {value}")
        else:
            mem.set_fact(key, value, category=category, source=source)
            suffix = " (LLM)" if source == "llm_inferred" else ""
            learned.append(f"Noted{suffix}: {key} = {value}")

    # Catch-all: self-disclosing 'I' statements neither extractor caught
    if not learned and _is_self_disclosing(text):
        ep_text = text.strip().capitalize()
        if mem.add_episodic(ep_text, tags=["auto", "self_disclosure"], confidence=0.65, source="conversation"):
            learned.append(f"Noted (episodic): {ep_text[:60]}")

    return learned


def handle_memory_correction(old_value: str, new_value: str, 
                             user_id: str = None) -> bool:
    """Handle when user corrects a fact.
    
    Example: "No, my name is Oliver, not Charlie"
    """
    memory = get_user_memory(user_id)
    if memory is None:
        return False
    
    # Search for facts containing old_value
    facts = memory.get_all_facts()
    for key, data in facts.items():
        if old_value.lower() in data["value"].lower():
            memory.set_fact(key, new_value, source="corrected", confidence=1.0)
            return True
    
    return False

