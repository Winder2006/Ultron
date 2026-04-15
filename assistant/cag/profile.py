from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_profile_slice(path: str = "assistant/memory/profile.json", max_keys: int = 15) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    # Keep order but limit keys
    out = {}
    for k in list(data.keys())[:max_keys]:
        out[k] = data[k]
    return out


