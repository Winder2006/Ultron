"""PostgreSQL + pgvector memory backend for MOTHER.

Drop-in replacement for the SQLite/JSON UserMemory class.
Uses PostgreSQL for facts + memories, pgvector for semantic search,
and preserves the same public interface so all callers work unchanged.

Requires:
    - PostgreSQL with pgvector extension
    - DATABASE_URL environment variable
    - Schema created via scripts/setup_db.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("mother.memory.pg")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://mother:password@localhost:5432/mother_db"
)


def _get_connection():
    """Get a psycopg2 connection."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def _embed_text(text: str) -> Optional[List[float]]:
    """Generate embedding for text.

    Uses the same embedding function as the legacy system if available,
    otherwise returns None (semantic search degrades to recency-only).
    """
    try:
        from assistant.nlp.embeddings import embed_texts
        vecs = embed_texts([text])
        if vecs and len(vecs) > 0:
            v = vecs[0]
            if hasattr(v, "tolist"):
                return v.tolist()
            return list(v)
    except ImportError:
        pass
    return None


class PgUserMemory:
    """PostgreSQL-backed per-user memory.

    Same public interface as UserMemory in manager.py.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._ensure_user()

    def _ensure_user(self):
        """Make sure this user exists in the users table."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (self.user_id, self.user_id),
            )
            conn.commit()
            cur.close()
        except Exception as e:
            logger.warning("Could not ensure user %s: %s", self.user_id, e)
            conn.rollback()
        finally:
            conn.close()

    # ======================== FACTS ========================

    def set_fact(
        self, key: str, value: str, category: str = "general",
        confidence: float = 1.0, source: str = "auto"
    ):
        """Store or update a structured fact."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO facts (user_id, key, value, category, confidence, source, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id, key) DO UPDATE
                    SET value = EXCLUDED.value,
                        category = EXCLUDED.category,
                        confidence = EXCLUDED.confidence,
                        source = EXCLUDED.source,
                        updated_at = NOW()
            """, (self.user_id, key, value, category, confidence, source))
            conn.commit()
            cur.close()
        except Exception as e:
            logger.error("Failed to set fact %s=%s: %s", key, value, e)
            conn.rollback()
        finally:
            conn.close()

    def get_fact(self, key: str) -> Optional[str]:
        """Get a single fact value by key."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT value FROM facts WHERE user_id = %s AND key = %s",
                (self.user_id, key),
            )
            row = cur.fetchone()
            cur.close()
            return row[0] if row else None
        except Exception as e:
            logger.error("Failed to get fact %s: %s", key, e)
            return None
        finally:
            conn.close()

    def get_facts_by_category(self, category: str) -> Dict[str, str]:
        """Get all facts in a category."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT key, value FROM facts WHERE user_id = %s AND category = %s",
                (self.user_id, category),
            )
            result = {row[0]: row[1] for row in cur.fetchall()}
            cur.close()
            return result
        except Exception as e:
            logger.error("Failed to get facts for category %s: %s", category, e)
            return {}
        finally:
            conn.close()

    def get_all_facts(self) -> Dict[str, Dict[str, Any]]:
        """Get all facts with metadata."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT key, value, category, confidence, source FROM facts WHERE user_id = %s",
                (self.user_id,),
            )
            result = {}
            for row in cur.fetchall():
                result[row[0]] = {
                    "value": row[1], "category": row[2],
                    "confidence": row[3], "source": row[4],
                }
            cur.close()
            return result
        except Exception as e:
            logger.error("Failed to get all facts: %s", e)
            return {}
        finally:
            conn.close()

    def delete_fact(self, key: str) -> bool:
        """Delete a fact."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM facts WHERE user_id = %s AND key = %s",
                (self.user_id, key),
            )
            deleted = cur.rowcount > 0
            conn.commit()
            cur.close()
            return deleted
        except Exception as e:
            logger.error("Failed to delete fact %s: %s", key, e)
            conn.rollback()
            return False
        finally:
            conn.close()

    def search_facts(self, query: str) -> List[Tuple[str, str]]:
        """Search facts by key or value substring."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            pattern = f"%{query.lower()}%"
            cur.execute("""
                SELECT key, value FROM facts
                WHERE user_id = %s AND (LOWER(key) LIKE %s OR LOWER(value) LIKE %s)
            """, (self.user_id, pattern, pattern))
            result = [(row[0], row[1]) for row in cur.fetchall()]
            cur.close()
            return result
        except Exception as e:
            logger.error("Failed to search facts: %s", e)
            return []
        finally:
            conn.close()

    # ======================== EPISODIC ========================

    def add_episodic(
        self, text: str, tags: List[str] = None,
        confidence: float = 0.8, source: str = "auto"
    ) -> bool:
        """Add an episodic memory. Returns True if stored (not a duplicate)."""
        # Check for duplicates
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT content FROM memories WHERE user_id = %s ORDER BY created_at DESC LIMIT 50",
                (self.user_id,),
            )
            existing = [row[0] for row in cur.fetchall()]
            # Simple word-overlap dedup
            for ex in existing:
                if self._similar(text, ex):
                    cur.close()
                    return False

            embedding = _embed_text(text)
            cur.execute("""
                INSERT INTO memories (user_id, content, embedding, tags, confidence, source)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                self.user_id, text,
                embedding if embedding else None,
                tags or [],
                confidence, source,
            ))
            conn.commit()
            cur.close()
            return True
        except Exception as e:
            logger.error("Failed to add episodic memory: %s", e)
            conn.rollback()
            return False
        finally:
            conn.close()

    @staticmethod
    def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
        """Word-overlap similarity for dedup."""
        stops = {"the", "a", "an", "is", "are", "was", "were", "i", "my", "me",
                 "we", "you", "in", "on", "at", "to", "of", "and", "or", "that"}
        wa = set(a.lower().split()) - stops
        wb = set(b.lower().split()) - stops
        if not wa or not wb:
            return a.lower() == b.lower()
        return len(wa & wb) / len(wa | wb) >= threshold

    def search_episodic(self, query: str, n: int = 5) -> List[Dict]:
        """Search episodic memories by semantic similarity + recency decay."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            embedding = _embed_text(query)

            if embedding:
                # Vector similarity search with recency decay
                cur.execute("""
                    SELECT id, content, tags, confidence, access_count,
                           created_at, last_accessed, decay_half_life_days,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM memories
                    WHERE user_id = %s AND embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (embedding, self.user_id, n * 3))
            else:
                # Fallback: text search + recency
                cur.execute("""
                    SELECT id, content, tags, confidence, access_count,
                           created_at, last_accessed, decay_half_life_days,
                           0.5 AS similarity
                    FROM memories
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (self.user_id, n * 3))

            rows = cur.fetchall()
            results = []
            now = datetime.now(timezone.utc)
            for row in rows:
                mem_id, content, tags, conf, access_count, created, last_accessed, half_life, sim = row
                # Recency decay: 30-day half-life
                days_old = (now - created.replace(tzinfo=timezone.utc)).total_seconds() / 86400
                recency = math.exp(-days_old * math.log(2) / (half_life or 30))
                # Combined score: 70% semantic + 30% recency
                score = 0.7 * sim + 0.3 * recency
                results.append({
                    "id": str(mem_id),
                    "text": content,
                    "tags": tags or [],
                    "confidence": conf,
                    "access_count": access_count,
                    "score": score,
                    "similarity": sim,
                    "recency": recency,
                    "created_at": created.isoformat() if created else None,
                })

            # Sort by combined score
            results.sort(key=lambda x: x["score"], reverse=True)

            # Update access counts for returned results
            returned_ids = [r["id"] for r in results[:n]]
            if returned_ids:
                cur.execute("""
                    UPDATE memories SET access_count = access_count + 1, last_accessed = NOW()
                    WHERE id = ANY(%s::uuid[])
                """, (returned_ids,))
                conn.commit()

            cur.close()
            return results[:n]
        except Exception as e:
            logger.error("Failed to search episodic: %s", e)
            return []
        finally:
            conn.close()

    def get_recent_episodic(self, n: int = 10) -> List[Dict]:
        """Get most recent episodic memories."""
        conn = _get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT content, tags, confidence, created_at
                FROM memories WHERE user_id = %s
                ORDER BY created_at DESC LIMIT %s
            """, (self.user_id, n))
            results = []
            for row in cur.fetchall():
                results.append({
                    "text": row[0], "tags": row[1] or [],
                    "confidence": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                })
            cur.close()
            return results
        except Exception as e:
            logger.error("Failed to get recent episodic: %s", e)
            return []
        finally:
            conn.close()

    # ======================== SUMMARY / PROMPT ========================

    def get_memory_summary(self, max_facts: int = 10, max_episodic: int = 5) -> str:
        """Human-readable summary of stored memories."""
        parts = []
        facts = self.get_all_facts()
        if facts:
            # Sort by category
            sorted_facts = sorted(facts.items(), key=lambda x: x[1].get("category", "z"))
            for key, meta in sorted_facts[:max_facts]:
                parts.append(f"{key}: {meta['value']}")
        recent = self.get_recent_episodic(max_episodic)
        if recent:
            parts.append("---")
            for m in recent:
                parts.append(m["text"])
        return "\n".join(parts) if parts else "No memories stored yet."

    def get_context_for_prompt(self, query: str = "", max_items: int = 5) -> str:
        """Build context string for LLM prompt injection."""
        parts = []
        # Inject all facts sorted by category
        facts = self.get_all_facts()
        if facts:
            cat_order = ["personal", "work", "preference", "contact", "general"]
            sorted_facts = sorted(
                facts.items(),
                key=lambda x: (
                    cat_order.index(x[1].get("category", "general"))
                    if x[1].get("category", "general") in cat_order
                    else 99
                ),
            )
            fact_lines = [f"{k}: {v['value']}" for k, v in sorted_facts]
            parts.append("Known facts: " + "; ".join(fact_lines))

        # Top episodic memories relevant to query
        if query:
            eps = self.search_episodic(query, n=max_items)
        else:
            eps = self.get_recent_episodic(max_items)
        if eps:
            ep_lines = [m["text"] for m in eps]
            parts.append("Memories: " + " | ".join(ep_lines))

        return "\n".join(parts)

    def save_learned_note(self, title: str, content: str,
                          source: str = "conversation") -> bool:
        """Save a learned note (stored as high-importance episodic memory)."""
        note_text = f"[{title}] {content}"
        return self.add_episodic(
            note_text, tags=["learned", source],
            confidence=0.9, source=source,
        )


def is_pg_available() -> bool:
    """Check if PostgreSQL is reachable and schema exists."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM facts LIMIT 0")
        cur.close()
        conn.close()
        return True
    except Exception:
        return False
