import os
import sys
import time
from datetime import datetime
from typing import Any

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from src.security.command_policy import register_command, run_command
from src.security.network_guard import assert_local_only

BOOT_LINES = [
    "INITIALIZING MOTHER-CORE...",
    "PRIMARY NETWORKS ONLINE",
    "ENVIRONMENTAL SENSORS ACTIVE",
    "BIOHAZARD MONITORING STANDBY",
    "CREW EXPENDABLE PROTOCOL: DISABLED",
    "WELCOME BACK, OFFICER WILLIAMS",
]

ASCII_ART = r"""
███    ███  ██████  ████████ ██   ██ ███████ ██████  
████  ████ ██    ██    ██    ██   ██ ██      ██   ██ 
██ ████ ██ ██    ██    ██    ███████ █████   ██████  
██  ██  ██ ██    ██    ██    ██   ██ ██      ██   ██ 
██      ██  ██████     ██    ██   ██ ███████ ██   ██
"""


def clear_screen() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def print_boot_sequence() -> None:
    clear_screen()
    print("\033[1;32m", end="")
    print(ASCII_ART)
    for line in BOOT_LINES:
        print(f"[MOTHER] {line}")
        time.sleep(0.7)
    print("\033[0m", end="")


def log_event(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[LOG] {ts} :: {message}")


# --- register minimal allowed commands ---

def _system_info() -> dict[str, Any]:
    return {"platform": os.name, "cwd": os.getcwd()}


def _list_files(path: str) -> list[str]:
    from pathlib import Path
    return [p.name for p in Path(os.path.expanduser(path)).expanduser().glob('*')]

register_command('system_info', _system_info)
register_command('list_files', _list_files)


def main() -> int:
    print_boot_sequence()
    ok, offenders = assert_local_only()
    if not ok:
        print("\033[31m[SECURITY] Non-local listeners detected:\033[0m")
        for line in offenders:
            print(line)
        # Auto-continue if MOTHER_AUTO env var is set (for automated launches)
        if os.environ.get("MOTHER_AUTO"):
            print("[SECURITY] Auto-continuing (MOTHER_AUTO mode)...")
        else:
            input("Type CONTINUE to proceed (local only expected): ")
    log_event("SYSTEM.READY")
    try:
        # Example gated call
        r = run_command('system_info')
        log_event(f"system_info -> {r}")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log_event("SHUTDOWN.REQUESTED")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
