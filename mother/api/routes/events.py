"""Server-Sent Events (SSE) endpoint for real-time MOTHER state updates.

The dashboard subscribes to /api/events and receives a continuous stream
of JSON events as they occur: STT results, LLM responses, vision
changes, etc.

Architecture: producer/consumer with FAN-OUT.

  ┌──────────────┐      ┌─────────────┐  ┌──────────────────┐
  │ voice route  │──┐   │   fan-out   │──│ subscriber queue │── /events #1
  │ etc.         │  ├──→│  drainer    │──│ subscriber queue │── /events #2
  └──────────────┘  │   │   (server)  │──│ subscriber queue │── /events #N
                    │   └─────────────┘  └──────────────────┘
                    │
              event_bus (single producer-side queue)

The earlier design had every /events connection reading from the same
producer queue — that's a competing-consumer pattern, so when two
browser tabs were open (main dashboard + /exec view), each event went
to whichever connection grabbed it first. The /exec tab silently
missed every event the main dashboard claimed.

Now: each connection gets its own bounded per-subscriber queue. The
fan-out task in server.py reads the producer queue and pushes copies
to every subscriber's queue. Slow subscribers drop their own events,
not anyone else's.
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


# Each subscriber queue is bounded so a slow client (or a tab that's
# been backgrounded) can't grow the buffer indefinitely. When the
# fan-out task tries to push to a full queue, it drops the event for
# THAT subscriber and moves on — the others still get it.
SUBSCRIBER_QUEUE_SIZE = 256


@router.get("/events")
async def event_stream(request: Request):
    """SSE endpoint — streams JSON events to one dashboard subscriber.

    Event format (one per line):
        data: {"type": "query", "text": "what's the weather", ...}

    Sends a heartbeat every 15s to keep the connection alive.
    """

    async def _generate():
        from mother.api.server import get_state
        state = get_state()
        subscribers: set = state.get("event_subscribers")

        if subscribers is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Event bus not initialized'})}\n\n"
            return

        # Per-connection queue. Bounded — if WE fall behind we lose
        # OUR events; other subscribers are unaffected.
        my_queue: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        subscribers.add(my_queue)
        logger.debug(
            "[events] new SSE subscriber (total=%d)", len(subscribers),
        )

        try:
            # Initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'ts': time.time()})}\n\n"

            while True:
                # Detect client disconnect promptly
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(my_queue.get(), timeout=15.0)
                    event["ts"] = time.time()
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat keeps proxies / browsers from killing the connection
                    yield f"data: {json.dumps({'type': 'heartbeat', 'ts': time.time()})}\n\n"
        finally:
            subscribers.discard(my_queue)
            logger.debug(
                "[events] SSE subscriber gone (remaining=%d)", len(subscribers),
            )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
