from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .nlp.rag import build_index, search_index
from .finance.yahoo import get_quote, get_news


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="assistant memory CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--k", type=int, default=4)
    p_fin = sub.add_parser("finance")
    p_fin.add_argument("symbol")
    p_news = sub.add_parser("finance-news")
    p_news.add_argument("symbol", nargs="?", default="^GSPC")
    p_news.add_argument("--count", type=int, default=5)
    args = parser.parse_args(argv)

    notes_dir = "assistant/notes"
    idx = "assistant/memory/faiss.index"
    meta = "assistant/memory/faiss_meta.json"

    if args.cmd == "ingest":
        build_index(notes_dir, idx, meta)
        print("Index built:", idx)
        return 0
    if args.cmd == "search":
        hits = search_index(args.query, idx, meta, k=args.k)
        for h in hits:
            print(f"- {h['text']}\n  path: {h['path']} meta: {h['meta']}")
        return 0
    if args.cmd == "finance":
        q = get_quote(args.symbol)
        print(q)
        return 0
    if args.cmd == "finance-news":
        items = get_news(args.symbol, count=args.count)
        for n in items:
            title = n.get("title", "")
            pub = n.get("publisher", "")
            print(f"- {title} ({pub})")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


