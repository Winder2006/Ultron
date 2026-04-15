"""Server-Sent Events (SSE) endpoint for real-time MOTHER state updates.

The dashboard subscribes to /api/events and receives a continuous stream of
JSON events as they occur: STT results, LLM responses, vision changes, etc.
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from mother.core.logging_config import get_logger

logger = get_logger("mother.api.events")
router = APIRouter(tags=["events"])


@router.get("/events")
async def event_stream(request: Request):
    """SSE endpoint — streams JSON events to the dashboard.

    Event format (one per line):
        data: {"type": "query", "text": "what's the weather", "intent": "WEATHER", ...}

    Sends a heartbeat every 15s to keep the connection alive.
    """

    async def _generate():
        from mother.api.server import get_state
        state = get_state()
        event_bus: asyncio.Queue = state.get("event_bus")

        if event_bus is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Event bus not initialized'})}\n\n"
            return

        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'ts': time.time()})}\n\n"

        last_heartbeat = time.monotonic()
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            try:
                event = await asyncio.wait_for(event_bus.get(), timeout=15.0)
                event["ts"] = time.time()
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat', 'ts': time.time()})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
