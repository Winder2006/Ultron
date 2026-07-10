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

    # Event bus for SSE broadcasting. The producer queue receives
    # events from anywhere in the app (`_emit_event`); the fan-out
    # drainer below reads from it and pushes to every subscriber's
    # personal queue so multiple /events connections can each see
    # every event (instead of competing for the next one).
    _state["event_bus"] = asyncio.Queue(maxsize=512)
    _state["event_subscribers"] = set()

    async def _event_fanout():
        """Drain the producer queue, broadcast to every subscriber.

        Slow subscriber? Their own queue fills up and they drop their
        copies. Other subscribers and the producer are unaffected.
        """
        bus: asyncio.Queue = _state["event_bus"]
        subs: set = _state["event_subscribers"]
        while True:
            try:
                event = await bus.get()
                # Copy is cheap — events are small JSON-y dicts. We
                # copy because the SSE handler mutates `ts`, and we
                # don't want one subscriber's mutation to bleed into
                # the next subscriber's view.
                for q in list(subs):
                    try:
                        q.put_nowait(dict(event))
                    except asyncio.QueueFull:
                        # Slow subscriber drops THEIR event. Log
                        # rate-limited so a stuck tab doesn't flood
                        # the log file.
                        pass
                    except Exception:
                        pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[event_fanout] iteration error: %s", e)

    _state["event_fanout_task"] = asyncio.create_task(_event_fanout())

    # Shared httpx.Client for tool dispatch — was previously created
    # per tool call, which leaked sockets and skipped TCP/TLS reuse.
    # One process-wide client gives connection pooling for free.
    import httpx as _httpx
    _state["tool_http_client"] = _httpx.Client(timeout=5.0)

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

    # ── Memory consolidation scheduler ──
    # Walks every known user every 30 minutes and runs a consolidation
    # pass for any user whose last run is overdue (default 6h). Pinned
    # to tier 2 because tier 1 is unreliable at structured-JSON output
    # and tier 3 is overkill. Best-effort: errors don't take down the
    # scheduler.
    async def _consolidation_scheduler():
        from mother.memory.consolidation import (
            consolidate_user, list_known_users, is_due,
        )
        from mother.llm.drivers import TieredLLMDriver
        # Wait a bit after startup so we don't compete with the warmup
        # path for first-call CPU/network.
        await asyncio.sleep(120.0)
        while True:
            try:
                llm_obj = _state.get("llm")
                if llm_obj is None:
                    await asyncio.sleep(1800)
                    continue

                # Build a tier-2-pinned chat callable for the consolidator.
                # This keeps the call out of any session-shared
                # _current_tier mutation path.
                if isinstance(llm_obj, TieredLLMDriver):
                    def _llm_call(messages, max_tokens=600):
                        return llm_obj.chat(
                            messages, max_tokens=max_tokens,
                            tier="tier2",
                        )
                else:
                    def _llm_call(messages, max_tokens=600):
                        return llm_obj.chat(messages, max_tokens=max_tokens)

                for uid in list_known_users():
                    if not is_due(uid):
                        continue
                    try:
                        # Run the (CPU + LLM) work on a thread so we
                        # don't block the asyncio loop while it's
                        # waiting on the network.
                        result = await asyncio.to_thread(
                            consolidate_user, uid, _llm_call,
                        )
                        if result.get("ran"):
                            logger.info(
                                "[consolidation] user=%s ok promoted=%d merged=%d dropped=%d",
                                uid, result["promoted"], result["merged"],
                                result["dropped"],
                            )
                    except Exception as ce:
                        logger.warning(
                            "[consolidation] user=%s failed: %s", uid, ce,
                        )
                    # Tiny gap between users so the scheduler is
                    # interruptible on shutdown.
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[consolidation] scheduler iteration error: %s", e)
            # Re-check every 30 minutes. is_due() will filter out
            # users that aren't yet overdue.
            await asyncio.sleep(1800)

    consolidation_task = asyncio.create_task(_consolidation_scheduler())
    _state["consolidation_task"] = consolidation_task

    logger.info("MOTHER API ready")
    yield

    # ── Shutdown ──
    fanout_task = _state.get("event_fanout_task")
    if fanout_task and not fanout_task.done():
        fanout_task.cancel()
        try:
            await fanout_task
        except (asyncio.CancelledError, Exception):
            pass
    consolidation_task = _state.get("consolidation_task")
    if consolidation_task and not consolidation_task.done():
        consolidation_task.cancel()
        try:
            await consolidation_task
        except (asyncio.CancelledError, Exception):
            pass
    mqtt_client = _state.get("mqtt_client")
    if mqtt_client:
        await mqtt_client.stop()
    tool_http = _state.get("tool_http_client")
    if tool_http is not None:
        try:
            tool_http.close()
        except Exception:
            pass
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
