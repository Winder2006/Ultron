from __future__ import annotations

from fastapi import FastAPI, Query
from pydantic import BaseModel

from .nlp.rag import search_index, build_index, build_code_index, warm_cache
from .nlp.code_indexer import collect_code_records
from .cag.profile import load_profile_slice
from .cag.episodic import maybe_write_nugget
from .finance.yahoo import get_quote, get_news


# Paths for the dedicated code index. Kept separate from the notes index
# so rebuilding one never clobbers the other.
_CODE_INDEX = "assistant/memory/code.index"
_CODE_META = "assistant/memory/code_meta.json"
_PROJECT_ROOT = "."  # assistant app is launched from the project root


app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}
@app.on_event("startup")
def _startup_warm() -> None:
    # Warm both indexes at startup so the first /search and /code-search
    # don't eat the ONNX load cost.
    try:
        warm_cache("assistant/memory/faiss.index", "assistant/memory/faiss_meta.json")
    except Exception:
        pass
    try:
        warm_cache(_CODE_INDEX, _CODE_META)
    except Exception:
        pass


@app.get("/warmup")
def api_warmup():
    ok = warm_cache("assistant/memory/faiss.index", "assistant/memory/faiss_meta.json")
    return {"warmed": ok}


@app.get("/search")
def api_search(q: str = Query(""), k: int = Query(4), path_contains: str | None = Query(None)):
    hits = search_index(q, "assistant/memory/faiss.index", "assistant/memory/faiss_meta.json", k=k, path_contains=path_contains)
    return hits


@app.get("/memory/profile")
def api_profile():
    return load_profile_slice()


class NuggetIn(BaseModel):
    text: str
    tags: list[str] = []
    confidence: float = 0.9


@app.post("/memory/episodic")
def api_nugget(n: NuggetIn):
    ok = maybe_write_nugget(n.text, n.tags, n.confidence)
    return {"stored": ok}


@app.get("/finance/quote")
def api_finance_quote(symbol: str = Query(...)):
    return get_quote(symbol)


@app.get("/finance/news")
def api_finance_news(symbol: str = Query("^GSPC"), count: int = Query(5)):
    return get_news(symbol, count=count)


@app.post("/reindex")
def api_reindex():
    build_index("assistant/notes", "assistant/memory/faiss.index", "assistant/memory/faiss_meta.json")
    return {"ok": True}


@app.get("/code-search")
def api_code_search(
    q: str = Query(""),
    k: int = Query(3),
    kind: str | None = Query(None, description="Filter by meta.kind: code | config | docs"),
    lang: str | None = Query(None, description="Filter by meta.lang: python | typescript | yaml | ..."),
):
    """Semantic search over the indexed codebase.

    Returns RAG records with `text`, `path`, and `meta.{kind,lang,symbol,lines}`.
    The `kind` and `lang` filters are applied post-retrieval against meta —
    useful for narrowing to e.g. "only Python code" or "only configs".
    """
    hits = search_index(q, _CODE_INDEX, _CODE_META, k=max(k * 2, k), mmr=True)
    if kind or lang:
        filtered = [
            h for h in hits
            if (not kind or h.get("meta", {}).get("kind") == kind)
            and (not lang or h.get("meta", {}).get("lang") == lang)
        ]
        if filtered:
            hits = filtered
    return hits[:k]


@app.post("/reindex-code")
def api_reindex_code():
    """Rebuild the code index in-process. Slower than running the CLI
    (no streaming output) but convenient from the dashboard or a hotkey."""
    records = collect_code_records(_PROJECT_ROOT)
    n = build_code_index(records, _CODE_INDEX, _CODE_META)
    # Re-warm so the first query after reindex doesn't pay the reload cost
    warm_cache(_CODE_INDEX, _CODE_META)
    return {"ok": True, "chunks": n}


