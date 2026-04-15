"""Phase 4 — Idempotent PostgreSQL schema setup for MOTHER.

Creates the database schema including pgvector extension and all tables.
Safe to run multiple times (uses IF NOT EXISTS / ON CONFLICT).

Requirements:
    - PostgreSQL running with pgvector extension available
    - DATABASE_URL set in .env or environment

Usage:
    python scripts/setup_db.py
    python scripts/setup_db.py --url postgresql://mother:password@localhost:5432/mother_db
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

SCHEMA_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    voice_embedding VECTOR(256),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ
);

-- Structured facts (key-value per user)
CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence FLOAT DEFAULT 1.0,
    source TEXT DEFAULT 'auto',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, key)
);

-- Episodic memories with vector embeddings
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    tags TEXT[] DEFAULT '{}',
    importance FLOAT DEFAULT 0.5,
    confidence FLOAT DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    source TEXT DEFAULT 'auto',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    decay_half_life_days INTEGER DEFAULT 30
);

-- Conversation history
CREATE TABLE IF NOT EXISTS conversation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Room context (for vision integration)
CREATE TABLE IF NOT EXISTS room_context (
    room TEXT PRIMARY KEY,
    occupants JSONB DEFAULT '[]',
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast vector similarity search
CREATE INDEX IF NOT EXISTS memories_embedding_idx
ON memories USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index for fast fact lookups
CREATE INDEX IF NOT EXISTS facts_user_key_idx
ON facts(user_id, key);

-- Index for conversation history ordering
CREATE INDEX IF NOT EXISTS conv_history_user_time_idx
ON conversation_history(user_id, created_at DESC);

-- Trigger function to enforce max 8 conversation turns per user (4 exchanges)
CREATE OR REPLACE FUNCTION trim_conversation_history() RETURNS trigger AS $$
BEGIN
    DELETE FROM conversation_history
    WHERE id IN (
        SELECT id FROM conversation_history
        WHERE user_id = NEW.user_id
        ORDER BY created_at DESC
        OFFSET 8  -- 4 exchanges = 8 turns (user + assistant)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop and recreate trigger (idempotent)
DROP TRIGGER IF EXISTS trim_conv_history_trigger ON conversation_history;
CREATE TRIGGER trim_conv_history_trigger
    AFTER INSERT ON conversation_history
    FOR EACH ROW EXECUTE FUNCTION trim_conversation_history();
"""


def setup_database(database_url: str):
    """Create schema in PostgreSQL."""
    import psycopg2

    print(f"Connecting to: {database_url.split('@')[-1]}...")  # hide credentials
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        cur = conn.cursor()

        print("Running schema creation...")
        cur.execute(SCHEMA_SQL)

        # Verify tables
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables created: {tables}")

        # Verify pgvector
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        if row:
            print(f"pgvector version: {row[0]}")
        else:
            print("WARNING: pgvector extension not found!")

        cur.close()
        conn.close()
        print("\nSchema setup complete.")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure PostgreSQL is running and the database exists:")
        print("  createdb mother_db")
        print("  # or: CREATE DATABASE mother_db;")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup MOTHER database schema")
    parser.add_argument(
        "--url",
        default=os.environ.get("DATABASE_URL", "postgresql://mother:password@localhost:5432/mother_db"),
        help="PostgreSQL connection URL",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MOTHER Phase 4 — Database Schema Setup")
    print("=" * 60)

    success = setup_database(args.url)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
