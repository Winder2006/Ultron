from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict, Iterator, List, Optional

import httpx


Role = str  # expected values: "system" | "user" | "assistant"


@dataclass
class ChatMessage:
    role: Role
    content: str


class LLMDriver:
    """Abstract base for LLM drivers.

    Implementations must provide a streaming chat interface for low latency.
    """

    def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
    ) -> Iterator[str]:
        raise NotImplementedError

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        chunks: List[str] = []
        for token in self.stream_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        ):
            chunks.append(token)
        return "".join(chunks)


class ClaudeLLMDriver(LLMDriver):
    """LLM driver for the Anthropic Claude API with streaming."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 150,
    ) -> None:
        import anthropic
        self.model = model
        self.max_tokens = max_tokens
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set it as an environment variable "
                "or pass api_key in config."
            )
        self._client = anthropic.Anthropic(api_key=key)

    def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Iterator[str]:
        # Separate system message from conversation messages
        system_text = ""
        api_messages = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            else:
                # Claude API requires alternating user/assistant roles
                # Map 'tool' role to 'user' for compatibility
                role = m.role if m.role in ("user", "assistant") else "user"
                api_messages.append({"role": role, "content": m.content})

        # Ensure messages start with 'user' and alternate properly
        if api_messages and api_messages[0]["role"] != "user":
            api_messages.insert(0, {"role": "user", "content": "Hello."})

        # Merge consecutive same-role messages
        merged = []
        for msg in api_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n" + msg["content"]
            else:
                merged.append(msg)
        api_messages = merged

        tok = max_tokens or self.max_tokens
        kwargs: Dict = {
            "model": self.model,
            "max_tokens": tok,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


class HybridLLMDriver(LLMDriver):
    """Routes queries to local Ollama for simple intents, Claude API for complex ones.

    Simple intents (weather, finance, reminders, identity) use the fast local model.
    Everything else goes to Claude for real reasoning.
    """

    def __init__(self, local: LLMDriver, cloud: LLMDriver) -> None:
        self.local = local
        self.cloud = cloud
        self._use_cloud = False  # flag set per-call by the router

    def route_to_cloud(self):
        """Signal that the next stream_chat call should use Claude."""
        self._use_cloud = True

    def route_to_local(self):
        """Signal that the next stream_chat call should use local Ollama."""
        self._use_cloud = False

    def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Iterator[str]:
        driver = self.cloud if self._use_cloud else self.local
        # Reset flag after routing
        self._use_cloud = False
        yield from driver.stream_chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
            tools=tools,
        )


# ---------------------------------------------------------------------------
# Tiered LLM routing via LiteLLM
# ---------------------------------------------------------------------------

TIER_MODELS = {
    "tier1": "gemini/gemini-2.5-flash-lite",
    "tier2": "groq/llama-3.3-70b-versatile",
    "tier3": "anthropic/claude-sonnet-4-20250514",
}


class TieredLLMDriver(LLMDriver):
    """Routes queries to the appropriate model tier via LiteLLM.

    All tiers use LiteLLM's unified OpenAI-compatible streaming interface.
    Tool calling uses the __TOOL_CALL__ sentinel pattern for backward compat.
    """

    def __init__(
        self,
        tier_models: Optional[Dict[str, str]] = None,
        max_tokens_by_tier: Optional[Dict[str, int]] = None,
    ) -> None:
        self._models = tier_models or dict(TIER_MODELS)
        self._max_tokens = max_tokens_by_tier or {
            "tier1": 80,
            "tier2": 150,
            "tier3": 300,
        }
        self._current_tier = "tier2"  # default

    def set_tier(self, tier: str):
        """Set the tier for the next stream_chat call."""
        if tier not in self._models:
            raise ValueError(f"Unknown tier: {tier}. Available: {list(self._models.keys())}")
        self._current_tier = tier

    @property
    def current_model(self) -> str:
        return self._models[self._current_tier]

    def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Iterator[str]:
        import litellm

        tier = self._current_tier
        model = self._models[tier]
        tok = max_tokens or self._max_tokens.get(tier, 150)

        # Build messages in OpenAI format. For Anthropic tiers, mark
        # the system message as cacheable — Claude's prompt caching
        # skips ~90% of the re-processing cost on turn 2+ when the
        # system prompt is stable, which saves 150-400ms of TTFT on
        # every non-first turn. The cache lives for 5 minutes by
        # default; subsequent conversational turns keep hitting it.
        is_anthropic = model.startswith("anthropic/") or "claude" in model.lower()
        api_messages = []
        for i, m in enumerate(messages):
            role = m.role
            if role == "tool":
                role = "user"
            if is_anthropic and role == "system" and i == 0:
                # Structured content array with cache_control so Claude
                # caches the system prompt. LiteLLM passes this through
                # to the Anthropic API unchanged.
                api_messages.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": m.content,
                        "cache_control": {"type": "ephemeral"},
                    }],
                })
            else:
                api_messages.append({"role": role, "content": m.content})

        # Ensure first message is from user
        if api_messages and api_messages[0]["role"] == "system" and len(api_messages) > 1:
            pass  # system + user is fine
        elif api_messages and api_messages[0]["role"] not in ("system", "user"):
            api_messages.insert(0, {"role": "user", "content": "Hello."})

        kwargs: Dict = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": tok,
            "stream": True,
        }

        # LiteLLM handles tools natively for providers that support it
        if tools and tier in ("tier2", "tier3"):
            # Convert Ollama tool format to OpenAI format if needed
            oai_tools = []
            for t in tools:
                if "function" in t:
                    oai_tools.append({"type": "function", "function": t["function"]})
                elif "type" in t:
                    oai_tools.append(t)
            if oai_tools:
                kwargs["tools"] = oai_tools

        try:
            response = litellm.completion(**kwargs)
            tool_call_buffer: Dict[int, Dict] = {}

            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Text content
                if delta.content:
                    yield delta.content

                # Tool calls — accumulate and emit as sentinel
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, "index") else 0
                        if idx not in tool_call_buffer:
                            tool_call_buffer[idx] = {
                                "name": "",
                                "arguments": "",
                            }
                        if tc.function:
                            if tc.function.name:
                                tool_call_buffer[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_call_buffer[idx]["arguments"] += tc.function.arguments

                # Check if stream is done
                finish = chunk.choices[0].finish_reason if chunk.choices else None
                if finish == "tool_calls" and tool_call_buffer:
                    # Emit tool calls as __TOOL_CALL__ sentinel for backward compat
                    for idx, tc_data in tool_call_buffer.items():
                        try:
                            args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        sentinel = {
                            "message": {
                                "tool_calls": [{
                                    "function": {
                                        "name": tc_data["name"],
                                        "arguments": args,
                                    }
                                }]
                            },
                            "done": True,
                        }
                        yield f"__TOOL_CALL__:{json.dumps(sentinel)}"
                    return

        except Exception as e:
            # Bounded fallback to a lower tier. The previous version
            # recursed into stream_chat, which meant a misconfigured
            # tier1 could loop (tier3 fails → tier2 fails → tier1 fails
            # → tier1 still in _models → recurse forever). We track
            # attempts per-call via a thread-local set so the recursion
            # can visit each tier at most once before raising.
            import logging
            _log = logging.getLogger("mother.llm")
            attempted = getattr(self, "_fallback_attempts", None)
            is_root = attempted is None
            if is_root:
                attempted = {tier}
                self._fallback_attempts = attempted
            try:
                fallback_order = ["tier3", "tier2", "tier1"]
                current_idx = fallback_order.index(tier) if tier in fallback_order else -1
                for fb_tier in fallback_order[current_idx + 1:]:
                    if fb_tier in attempted:
                        continue
                    if fb_tier not in self._models:
                        continue
                    attempted.add(fb_tier)
                    _log.warning(
                        "[LLM] %s failed (%s) — falling back to %s (%s)",
                        model, e, fb_tier, self._models[fb_tier],
                    )
                    self._current_tier = fb_tier
                    try:
                        yield from self.stream_chat(
                            messages, temperature, max_tokens, extra, tools
                        )
                        return
                    except Exception as fe:
                        _log.warning(
                            "[LLM] fallback tier %s also failed: %s", fb_tier, fe,
                        )
                        continue
                _log.error("[LLM] all tiers exhausted — re-raising original: %s", e)
                raise
            finally:
                if is_root:
                    self._fallback_attempts = None


class OllamaLLMDriver(LLMDriver):
    """LLM driver for Ollama's HTTP API.

    Uses /api/chat with stream=true to yield tokens as they arrive.
    """

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        request_timeout_s: float = 120.0,
        keep_alive: Optional[str] = None,
        num_thread: Optional[int] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.keep_alive = keep_alive
        self.num_thread = num_thread
        # Persistent client reused across all calls — avoids TCP handshake overhead
        self._client = httpx.Client(timeout=request_timeout_s)

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _to_ollama_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Iterator[str]:
        """Stream chat tokens. Yields text chunks.

        If *tools* is provided the schema is forwarded to Ollama. When the
        model emits a tool_call instead of text, a special sentinel string
        ``"__TOOL_CALL__:<json>"`` is yielded so the caller can detect and
        dispatch the call without breaking the streaming interface.
        """
        payload: Dict = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": 2048,      # explicit context window — avoids Ollama re-allocation overhead
                "top_k": 20,          # narrow candidate set → faster sampling
                "top_p": 0.9,         # nucleus sampling complement
                "repeat_penalty": 1.0, # disabled (=1.0 = no-op) → skips repeat-penalty pass
            },
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if self.num_thread is not None:
            payload["options"]["num_thread"] = self.num_thread
        if extra and "options" in extra and "num_thread" in extra["options"]:
            payload["options"]["num_thread"] = extra["options"]["num_thread"]
        if extra:
            opts = payload.setdefault("options", {})
            opts.update(extra.get("options", {}))
            for k, v in extra.items():
                if k != "options":
                    payload[k] = v
        if tools:
            payload["tools"] = tools

        headers: Dict[str, str] = {}
        url = f"{self.base_url}/api/chat"
        with self._client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if isinstance(data, dict):
                    message = data.get("message", {})
                    # Tool call: signal caller via sentinel token
                    if message.get("tool_calls"):
                        yield f"__TOOL_CALL__:{json.dumps(data)}"
                        if data.get("done", False):
                            break
                        continue
                    content = message.get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break


