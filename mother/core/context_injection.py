"""Context-enrichment for Ultron's prompts — notes RAG + code RAG.

Both the WebSocket voice route and the legacy PTT orchestrator need to
enrich the system prompt before sending to the LLM. This module owns
that logic in one place:

    1. Heuristic classifier: does this query look like it's about code?
    2. HTTP fetch: call /search and/or /code-search on the RAG service.
    3. Formatting: turn hit records into a compact context block.

Everything is best-effort and bounded by a short timeout — if the RAG
service is down, we silently return an empty string and let the LLM
answer without enrichment.
"""
from __future__ import annotations

import asyncio
import re
from typing import Iterable, List, Optional

import httpx


# Queries that look like they want code/architecture context get biased
# toward the code index. These are intentionally narrow — we'd rather
# miss a few than pollute unrelated queries with code snippets.
_CODE_QUERY_RE = re.compile(
    r"\b("
    r"your\s+code|your\s+codebase|your\s+architecture|your\s+pipeline|"
    r"your\s+tts|your\s+stt|your\s+llm|your\s+memory|your\s+voice|"
    r"how\s+do\s+you|how\s+are\s+you\s+built|how\s+does\s+your|"
    r"what\s+happens\s+when|what\s+(?:does|do)\s+your|"
    r"show\s+me\s+(?:the|your)|explain\s+(?:the|your)|"
    r"config|configuration|yaml|audioworklet|websocket|deepgram|cerebras|"
    r"orchestrator|handler|intent|classifier|normalizer|speaker\s+id|"
    r"filter\s+chain|ring\s+buffer|module|function|class|endpoint"
    r")\b",
    re.IGNORECASE,
)


def looks_like_code_query(text: str) -> bool:
    """Cheap regex heuristic. <50µs. Err on the side of 'no' — false
    positives mean we inject code into unrelated prompts and waste tokens."""
    if not text:
        return False
    return bool(_CODE_QUERY_RE.search(text))


def _format_hits(hits: List[dict], *, code_kind: bool) -> str:
    """Render hit records as a compact bullet block.

    For code hits we include the symbol name so Ultron can reference
    his own architecture by name ("my DeepgramTTSEngine.speak"); for
    notes hits we keep the old path-only format.
    """
    out: List[str] = []
    for h in hits:
        text = (h.get("text") or "").strip()
        path = h.get("path") or ""
        meta = h.get("meta") or {}
        if code_kind:
            sym = meta.get("symbol") or path
            lines = meta.get("lines") or ""
            lbl = f"{sym}" + (f" [{lines}]" if lines else "")
            # Keep each chunk reasonably short in the injection — the
            # LLM doesn't need the whole function body, just the header
            # and a couple lines of context.
            snippet = text[:600]
            out.append(f"- {lbl}: {snippet}")
        else:
            # Bound notes injection too. Previously unbounded, which
            # let a single fat lore-style note (~4kB) dominate the
            # system prompt on every turn. 500 chars per note is plenty
            # — the LLM can ask for more via search_info if needed.
            snippet = text[:500]
            if len(text) > 500:
                snippet += "…"
            out.append(f"- {snippet} (src: {path})")
    return "\n".join(out)


async def _get(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    timeout_s: float,
) -> Optional[List[dict]]:
    """Single best-effort GET. Returns None on any failure."""
    try:
        r = await client.get(url, params=params, timeout=timeout_s)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return data
    except Exception:
        return None
    return None


_ASYNC_CLIENT = None


def _shared_async_client() -> "httpx.AsyncClient":
    """Process-wide async client so per-turn RAG fetches reuse their
    TCP connection instead of handshaking every turn."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None or _ASYNC_CLIENT.is_closed:
        _ASYNC_CLIENT = httpx.AsyncClient()
    return _ASYNC_CLIENT


async def fetch_context(
    text: str,
    *,
    api_base: str,
    notes_k: int = 2,
    code_k: int = 3,
    timeout_ms: int = 500,
    enable_notes: bool = True,
    enable_code: bool = True,
    force_code: bool = False,
) -> str:
    """Fetch RAG context and return a formatted block (possibly empty).

    Both indexes are queried concurrently. Code results are included
    when (a) the query looks code-related by heuristic, or (b) the
    caller forces it. Notes are always queried when enabled — they
    already filter themselves by relevance via the retriever.

    The returned string is ready to be appended to the system prompt:

        Memory of your own code:
        - mother.tts.engine.DeepgramTTSEngine.speak [120-150]: streams PCM ...
        - mother.api.routes.voice ...

        Relevant notes:
        - Winder prefers short responses (src: assistant/notes/winder.md)

    Returns "" when nothing useful came back.
    """
    if not text.strip():
        return ""

    timeout_s = max(0.1, timeout_ms / 1000.0)
    want_code = enable_code and (force_code or looks_like_code_query(text))

    # Shared client — this runs on the per-turn hot path, and creating
    # a fresh AsyncClient per call paid a new TCP handshake to the RAG
    # service on every single turn.
    client = _shared_async_client()
    coros = []
    if enable_notes:
        coros.append(
            _get(client, f"{api_base}/search", {"q": text, "k": notes_k}, timeout_s)
        )
    else:
        coros.append(asyncio.sleep(0, result=None))  # type: ignore[arg-type]

    if want_code:
        coros.append(
            _get(client, f"{api_base}/code-search", {"q": text, "k": code_k}, timeout_s)
        )
    else:
        coros.append(asyncio.sleep(0, result=None))  # type: ignore[arg-type]

    notes_hits, code_hits = await asyncio.gather(*coros, return_exceptions=False)

    blocks: List[str] = []

    if code_hits:
        body = _format_hits(code_hits[:code_k], code_kind=True)
        if body:
            blocks.append(
                "Memory of your own code (your implementation; describe its shape and "
                "purpose in-character, do not quote raw syntax aloud):\n" + body
            )

    if notes_hits:
        body = _format_hits(notes_hits[:notes_k], code_kind=False)
        if body:
            blocks.append("Relevant notes:\n" + body)

    return "\n\n".join(blocks)


def fetch_context_sync(
    text: str,
    *,
    api_base: str,
    notes_k: int = 2,
    code_k: int = 3,
    timeout_ms: int = 500,
    enable_notes: bool = True,
    enable_code: bool = True,
    force_code: bool = False,
) -> str:
    """Blocking variant for callers that aren't on an event loop
    (e.g. the legacy CLI). Uses httpx sync client."""
    if not text.strip():
        return ""
    timeout_s = max(0.1, timeout_ms / 1000.0)
    want_code = enable_code and (force_code or looks_like_code_query(text))

    notes_hits: Optional[List[dict]] = None
    code_hits: Optional[List[dict]] = None
    with httpx.Client() as client:
        if enable_notes:
            try:
                r = client.get(
                    f"{api_base}/search",
                    params={"q": text, "k": notes_k},
                    timeout=timeout_s,
                )
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        notes_hits = data
            except Exception:
                pass
        if want_code:
            try:
                r = client.get(
                    f"{api_base}/code-search",
                    params={"q": text, "k": code_k},
                    timeout=timeout_s,
                )
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list):
                        code_hits = data
            except Exception:
                pass

    blocks: List[str] = []
    if code_hits:
        body = _format_hits(code_hits[:code_k], code_kind=True)
        if body:
            blocks.append(
                "Memory of your own code (your implementation; describe its shape and "
                "purpose in-character, do not quote raw syntax aloud):\n" + body
            )
    if notes_hits:
        body = _format_hits(notes_hits[:notes_k], code_kind=False)
        if body:
            blocks.append("Relevant notes:\n" + body)
    return "\n\n".join(blocks)
