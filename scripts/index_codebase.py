"""Rebuild the codebase RAG index.

Walks `mother/`, `dashboard/src/`, `configs/`, and `scripts/` plus the
top-level README/SECURITY, turns each source file into AST-aware chunks
(see `assistant.nlp.code_indexer`), embeds them with the local MiniLM
ONNX model, and writes a FAISS index next to the existing notes index.

Run:
    python scripts/index_codebase.py
    python scripts/index_codebase.py --out assistant/memory
    python scripts/index_codebase.py --dry-run     # just print stats

This is safe to run any time — it fully replaces the code index without
touching the notes index.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the repo root importable when run as `python scripts/index_codebase.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from assistant.nlp.code_indexer import collect_code_records, summarize  # noqa: E402
from assistant.nlp.rag import build_code_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild the codebase RAG index.")
    ap.add_argument(
        "--out",
        default="assistant/memory",
        help="Directory to write code.index + code_meta.json into.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and print stats, but don't embed or write the index.",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    records = collect_code_records(_PROJECT_ROOT)
    t_walk = time.perf_counter() - t0

    stats = summarize(records)
    print(f"[index_codebase] walked in {t_walk*1000:.0f}ms")
    print(f"[index_codebase] chunk stats:")
    for key, val in sorted(stats.items()):
        print(f"   {key:24s}  {val}")

    if args.dry_run:
        # Show a handful of symbols as a sanity check
        print("\n[index_codebase] sample records:")
        for r in records[:8]:
            sym = r.get("meta", {}).get("symbol", "?")
            path = r.get("path", "?")
            print(f"   {sym:60s}  ({path})")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(out_dir / "code.index")
    meta_path = str(out_dir / "code_meta.json")

    t1 = time.perf_counter()
    n = build_code_index(records, index_path, meta_path)
    t_embed = time.perf_counter() - t1

    print(f"\n[index_codebase] embedded {n} chunks in {t_embed:.1f}s")
    print(f"[index_codebase] wrote {index_path}")
    print(f"[index_codebase] wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
