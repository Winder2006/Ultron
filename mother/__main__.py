"""Unified entry point for MOTHER.

Usage:
    python -m mother --ptt              # Push-to-talk (default)
    python -m mother --prompt "..."     # Single prompt
    python -m mother --server           # API server mode
    python -m mother --server --port 9000
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="mother",
        description="MU/TH/UR 6000 AI Assistant",
    )
    parser.add_argument("--config", default="configs/app.yaml", help="Path to YAML config")
    parser.add_argument("--ptt", action="store_true", help="Push-to-talk mode (Enter key)")
    parser.add_argument("--prompt", default=None, help="Single text prompt (no mic)")
    parser.add_argument("--server", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8300, help="Server port")
    args = parser.parse_args()

    # Default to PTT if no mode specified
    if not args.server and not args.prompt:
        args.ptt = True

    if args.server:
        # Server mode — delegates to uvicorn
        from mother.core.orchestrator import Orchestrator
        orch = Orchestrator(config_path=args.config)
        asyncio.run(orch.run_server(host=args.host, port=args.port))
    elif args.prompt:
        # Single prompt mode
        from mother.core.orchestrator import Orchestrator
        orch = Orchestrator(config_path=args.config)
        asyncio.run(_run_prompt(orch, args.prompt))
    else:
        # PTT mode
        from mother.core.orchestrator import Orchestrator
        orch = Orchestrator(config_path=args.config)
        asyncio.run(_run_ptt(orch))


async def _run_prompt(orch, prompt: str):
    await orch.init()
    await orch.run_prompt(prompt)


async def _run_ptt(orch):
    await orch.init()
    await orch.run_ptt()


if __name__ == "__main__":
    main()
