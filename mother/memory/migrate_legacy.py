"""One-time migration: SQLite/JSON → PostgreSQL.

Migrates all existing MOTHER memory data:
    facts.db (SQLite)              → facts table
    episodic.json / episodic_memory.json → memories table
    conv_history.json              → conversation_history table
    users from assistant/memory/users/ → users table

Safe to re-run: uses INSERT ... ON CONFLICT DO NOTHING.

Usage:
    python -m mother.memory.migrate_legacy
    python -m mother.memory.migrate_legacy --url postgresql://mother:password@localhost:5432/mother_db
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MEMORY_BASE = Path("assistant/memory")
USERS_DIR = MEMORY_BASE / "users"


def migrate_users(conn, cur) -> dict:
    """Migrate user directories to users table. Returns {user_id: uuid} map."""
    user_map = {}
    if not USERS_DIR.exists():
        print("  No users directory found — skipping")
        return user_map

    for user_dir in USERS_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        cur.execute(
            "INSERT INTO users (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (user_id, user_id),
        )
        user_map[user_id] = user_id
        print(f"  User: {user_id}")

    conn.commit()
    print(f"  Migrated {len(user_map)} users")
    return user_map


def migrate_facts(conn, cur, user_map: dict) -> int:
    """Migrate SQLite facts.db to PostgreSQL facts table."""
    count = 0
    for user_id in user_map:
        db_path = USERS_DIR / user_id / "facts.db"
        if not db_path.exists():
            continue

        try:
            sqlite_conn = sqlite3.connect(str(db_path))
            sqlite_cur = sqlite_conn.cursor()
            sqlite_cur.execute("SELECT key, value, category, confidence, source FROM facts")

            for row in sqlite_cur.fetchall():
                key, value, category, confidence, source = row
                cur.execute("""
                    INSERT INTO facts (user_id, key, value, category, confidence, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, key) DO NOTHING
                """, (user_id, key, value, category or "general",
                      confidence or 1.0, source or "migrated"))
                count += 1

            sqlite_conn.close()
        except Exception as e:
            print(f"  Warning: Failed to migrate facts for {user_id}: {e}")

    conn.commit()
    print(f"  Migrated {count} facts")
    return count


def migrate_episodic(conn, cur, user_map: dict) -> int:
    """Migrate episodic JSON memories to PostgreSQL memories table."""
    count = 0
    for user_id in user_map:
        # Try both naming conventions
        for name in ("episodic.json", "episodic_memory.json"):
            ep_path = USERS_DIR / user_id / name
            if not ep_path.exists():
                continue

            try:
                with open(ep_path, "r", encoding="utf-8") as f:
                    memories = json.load(f)

                if isinstance(memories, dict):
                    memories = memories.get("memories", [])

                for mem in memories:
                    text = mem.get("text", mem.get("content", ""))
                    if not text:
                        continue
                    tags = mem.get("tags", [])
                    confidence = mem.get("confidence", 0.8)
                    source = mem.get("source", "migrated")
                    created = mem.get("created_at", mem.get("timestamp"))

                    cur.execute("""
                        INSERT INTO memories (user_id, content, tags, confidence, source, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        user_id, text, tags, confidence, source,
                        created or datetime.now(timezone.utc).isoformat(),
                    ))
                    count += 1
            except Exception as e:
                print(f"  Warning: Failed to migrate episodic for {user_id}/{name}: {e}")

    conn.commit()
    print(f"  Migrated {count} episodic memories")
    return count


def migrate_conversation_history(conn, cur, user_map: dict) -> int:
    """Migrate conversation history JSON to PostgreSQL."""
    count = 0
    for user_id in user_map:
        hist_path = USERS_DIR / user_id / "conv_history.json"
        if not hist_path.exists():
            continue

        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            exchanges = data if isinstance(data, list) else data.get("exchanges", [])
            for entry in exchanges:
                role = entry.get("role", "user")
                content = entry.get("content", "")
                if not content:
                    continue
                cur.execute("""
                    INSERT INTO conversation_history (user_id, role, content)
                    VALUES (%s, %s, %s)
                """, (user_id, role, content))
                count += 1
        except Exception as e:
            print(f"  Warning: Failed to migrate conv history for {user_id}: {e}")

    conn.commit()
    print(f"  Migrated {count} conversation turns")
    return count


def main():
    parser = argparse.ArgumentParser(description="Migrate MOTHER memory to PostgreSQL")
    parser.add_argument(
        "--url",
        default=os.environ.get("DATABASE_URL", "postgresql://mother:password@localhost:5432/mother_db"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MOTHER Phase 4 — Legacy Memory Migration")
    print("=" * 60)
    print(f"Source: {MEMORY_BASE}/")
    print(f"Target: {args.url.split('@')[-1]}")
    print()

    import psycopg2
    try:
        conn = psycopg2.connect(args.url)
        cur = conn.cursor()
    except Exception as e:
        print(f"ERROR: Cannot connect to PostgreSQL: {e}")
        print("Run scripts/setup_db.py first to create the schema.")
        return 1

    print("[1/4] Migrating users...")
    user_map = migrate_users(conn, cur)

    print("[2/4] Migrating facts...")
    migrate_facts(conn, cur, user_map)

    print("[3/4] Migrating episodic memories...")
    migrate_episodic(conn, cur, user_map)

    print("[4/4] Migrating conversation history...")
    migrate_conversation_history(conn, cur, user_map)

    cur.close()
    conn.close()

    print()
    print("Migration complete.")
    print("Set 'memory.backend: postgresql' in config/app.yaml to use the new backend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
