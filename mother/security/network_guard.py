from __future__ import annotations

import socket
import subprocess
from typing import List, Tuple


def _listeners_windows() -> List[str]:
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL)
        offenders: List[str] = []
        for line in out.splitlines():
            if "LISTENING" in line.upper():
                parts = line.split()
                if len(parts) >= 5:
                    addr = parts[1]
                    if not addr.startswith("127.0.0.1:") and not addr.startswith("::1:"):
                        offenders.append(line.strip())
        return offenders
    except Exception:
        return []


def _listeners_posix() -> List[str]:
    try:
        out = subprocess.check_output(["lsof", "-i", "-P", "-n"], text=True, stderr=subprocess.DEVNULL)
        offenders: List[str] = []
        for line in out.splitlines():
            if "LISTEN" in line:
                if "127.0.0.1" not in line and "::1" not in line:
                    offenders.append(line.strip())
        return offenders
    except Exception:
        return []


def assert_local_only() -> Tuple[bool, List[str]]:
    import os
    if os.name == 'nt':
        offenders = _listeners_windows()
    else:
        offenders = _listeners_posix()
    return (len(offenders) == 0, offenders)
