"""Memory subsystem for MOTHER.

Supports two backends:
- sqlite (legacy): SQLite + JSON files, works offline, default
- postgresql: PostgreSQL + pgvector, semantic search, opt-in via config

The backend is selected by reading config at import time.
All public functions work identically regardless of backend.
"""
