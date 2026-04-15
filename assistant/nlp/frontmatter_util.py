from __future__ import annotations

from typing import Tuple, Dict


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse optional YAML front matter. Returns (meta, body).

    Uses python-frontmatter if available; otherwise a minimal --- ... --- parser.
    """
    try:
        import frontmatter  # type: ignore

        post = frontmatter.loads(text)
        meta = dict(post.metadata or {})
        # Ensure JSON-serializable
        for k, v in list(meta.items()):
            try:
                import datetime as _dt
                if isinstance(v, (_dt.date, _dt.datetime)):
                    meta[k] = v.isoformat()
            except Exception:
                pass
        return meta, post.content
    except Exception:
        pass

    # Minimal parser
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            header = text[4:end]
            body = text[end + 5 :]
            meta: Dict[str, object] = {}
            for line in header.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            return meta, body
    return {}, text


