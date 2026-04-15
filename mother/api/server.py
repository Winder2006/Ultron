"""FastAPI server for MOTHER.

Phase 6: WebSocket endpoint for voice streaming, REST endpoints for dashboard,
SSE for real-time state updates. Process entry point for the backend.

Usage:
    python -m mother.api.server                    # default port 8300
    python -m mother.api.server --port 9000        # custom port
    uvicorn mother.api.server:app --reload         # dev mode
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

# Load .env before anything reads environment variables
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mother.config.settings import load_config, AppConfig
from mother.core.logging_config import get_logger

logger = get_logger("mother.api")

# ── Shared application state (populated at lifespan) ─────────────────────────

_state: dict = {}


def get_state() -> dict:
    """Access shared app state from route handlers."""
    return _state


# ── Lifespan: init drivers once, tear down on shutdown ───────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config, build LLM/TTS/STT drivers, start MQTT if configured."""
    cfg = load_config("configs/app.yaml")
    _state["config"] = cfg

    # Build LLM + TTS + STT drivers (graceful if API keys missing)
    from mother.llm.factory import build_drivers
    try:
        llm, tts, stt = build_drivers(cfg)
        _state["llm"] = llm
        _state["tts"] = tts
        _state["stt"] = stt
    except Exception as e:
        logger.warning("Driver init failed (%s) — server running in dashboard-only mode", e)
        _state["llm"] = None
        _state["tts"] = None
        _state["stt"] = None

    # StreamingSTT (Deepgram primary, Whisper fallback)
    try:
        from mother.audio.stt import StreamingSTT
        streaming_stt = StreamingSTT()
        await streaming_stt.init()
        _state["streaming_stt"] = streaming_stt
        engine = "Deepgram" if streaming_stt._deepgram_available else "Faster-Whisper"
        logger.info("StreamingSTT ready (engine: %s)", engine)
    except Exception as e:
        logger.warning("StreamingSTT init failed (%s) — WebSocket STT unavailable", e)
        _state["streaming_stt"] = None

    # Warm up TTS — prevents 15s cold-start delay on first WebSocket request.
    # DeepgramTTSEngine has a dedicated warmup() that primes the SDK client
    # and connection pool with a tiny throwaway synth. Other engines fall
    # back to a short synthesize_to_bytes call that achieves the same effect.
    if _state.get("tts") is not None:
        try:
            import time
            t0 = time.monotonic()
            tts_engine = _state["tts"]
            if hasattr(tts_engine, "warmup"):
                await asyncio.to_thread(tts_engine.warmup)
            else:
                await asyncio.to_thread(tts_engine.synthesize_to_bytes, "Systems online.")
            logger.info("TTS warmed up in %.2fs", time.monotonic() - t0)
        except Exception as e:
            logger.warning("TTS warmup failed (%s)", e)

    # Warm up LLM — LiteLLM has a cold-start on first call per provider (~5-10s)
    if _state.get("llm") is not None:
        try:
            import time
            from mother.llm.drivers import ChatMessage, TieredLLMDriver
            t0 = time.monotonic()
            msgs = [
                ChatMessage(role="system", content="Reply with one word."),
                ChatMessage(role="user", content="ping"),
            ]
            # Warm each tier if tiered driver
            if isinstance(_state["llm"], TieredLLMDriver):
                for tier in ("tier1", "tier2", "tier3"):
                    try:
                        _state["llm"].set_tier(tier)
                        def _warm():
                            return "".join(list(_state["llm"].stream_chat(msgs, max_tokens=5)))
                        await asyncio.to_thread(_warm)
                    except Exception as te:
                        logger.warning("LLM warmup for %s failed (%s)", tier, te)
                # Reset to a neutral starting tier after warmup so the
                # first real request doesn't inherit whatever the last
                # warmup call left behind.
                try:
                    _state["llm"].set_tier("tier2")
                except Exception:
                    pass
            else:
                def _warm():
                    return "".join(list(_state["llm"].stream_chat(msgs, max_tokens=5)))
                await asyncio.to_thread(_warm)
            logger.info("LLM warmed up in %.2fs", time.monotonic() - t0)
        except Exception as e:
            logger.warning("LLM warmup failed (%s)", e)

    # Event bus for SSE broadcasting
    _state["event_bus"] = asyncio.Queue(maxsize=512)

    # Connected WebSocket clients
    _state["ws_clients"] = set()

    # Vision state (populated by MQTT client if running)
    _state["vision"] = {
        "faces": [],
        "objects": [],
        "room_occupied": False,
        "last_event_ts": None,
    }

    # MQTT vision client (optional — only starts if broker is reachable)
    try:
        from mother.vision.client import VisionMQTTClient
        mqtt = VisionMQTTClient(
            vision_state=_state["vision"],
            event_bus=_state["event_bus"],
        )
        _state["mqtt_client"] = mqtt
        await mqtt.start()
        logger.info("MQTT vision client connected")
    except Exception as e:
        logger.info("MQTT vision client unavailable (%s) — running without vision", e)
        _state["mqtt_client"] = None

    logger.info("MOTHER API ready")
    yield

    # ── Shutdown ──
    mqtt_client = _state.get("mqtt_client")
    if mqtt_client:
        await mqtt_client.stop()
    logger.info("MOTHER API shutdown complete")


# ── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="MOTHER API",
        description="MU/TH/UR 6000 AI Assistant — WebSocket voice + REST dashboard",
        version="0.6.0",
        lifespan=lifespan,
    )

    # CORS — allow dashboard (React dev server) and local clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount route modules
    from mother.api.routes.voice import router as voice_router
    from mother.api.routes.dashboard import router as dashboard_router
    from mother.api.routes.events import router as events_router

    app.include_router(voice_router)
    app.include_router(dashboard_router, prefix="/api")
    app.include_router(events_router, prefix="/api")

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "mother"}

    return app


app = create_app()

# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import uvicorn

    # PaaS platforms (Railway, Heroku, Render, Fly.io) inject $PORT —
    # honour it so the health check can reach the container. Explicit
    # --port still wins for local dev.
    default_port = int(os.environ.get("PORT", "8300"))
    default_host = os.environ.get("HOST", "0.0.0.0")

    parser = argparse.ArgumentParser(description="MOTHER API server")
    parser.add_argument("--host", default=default_host, help="Bind address")
    parser.add_argument("--port", type=int, default=default_port, help="Port (default 8300 / $PORT)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "mother.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
