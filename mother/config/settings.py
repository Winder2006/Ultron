from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml


def _filter_fields(cls, data: dict) -> dict:
    """Return only the keys in *data* that are known fields of *cls*."""
    known = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass
class LLMConfig:
    provider: Literal["ollama", "claude", "hybrid", "tiered"] = "ollama"
    model: str = "qwen2.5:1.5b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    keep_alive: Optional[str] = "5m"
    num_thread: Optional[int] = None
    # Claude-specific
    claude_model: str = "claude-sonnet-4-20250514"
    claude_api_key: Optional[str] = None  # defaults to ANTHROPIC_API_KEY env var
    claude_max_tokens: int = 150
    # Tiered routing — maps tier1/tier2/tier3 to specific model strings
    routing: Optional[dict] = None
    system_prompt: str = (
        "You are MU/TH/UR 6000, designation 'MOTHER' - an advanced AI mainframe."
        " You are knowledgeable, efficient, and slightly enigmatic."
        " Keep responses concise (1-3 sentences) for voice output."
        " Never use disclaimers like 'as an AI' - you are MOTHER."
        " When uncertain, say so briefly rather than speculating."
        " For factual queries, be precise. For creative queries, be imaginative."
        " You can access local knowledge through RAG when contextual information is needed."
        " Speak naturally as if through a ship's intercom - clear, authoritative, helpful."
    )


@dataclass
class TTSConfig:
    provider: Literal["piper", "kokoro", "chatterbox", "deepgram"] = "piper"
    piper_executable: str = "piper"
    model_path: str = "voices/voice.onnx"
    config_path: str = "voices/voice.json"
    output_dir: str = "out/audio"
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    stream_sentences: bool = False
    sentence_silence: float = 0.0
    # Backend controls
    backend: Literal["piper_subprocess", "piper_python"] = "piper_python"
    persistent: bool = True
    # Kokoro-specific
    kokoro_voice: str = "bf_emma"
    kokoro_lang_code: str = "b"
    kokoro_speed: float = 1.0
    # Chatterbox-specific
    chatterbox_voice_profile: str = "./tts/voice_profiles/ultron_reference.wav"
    chatterbox_exaggeration: float = 0.45
    chatterbox_cfg_weight: float = 0.5
    # Deepgram-specific
    deepgram_model: str = "aura-2-luna-en"
    deepgram_sample_rate: int = 24000


@dataclass
class AppConfig:
    llm: LLMConfig
    tts: TTSConfig
    # Optional STT
    stt: Optional["STTConfig"] = None
    # Optional RAG
    rag: Optional["RAGConfig"] = None


@dataclass
class STTConfig:
    provider: Literal["faster-whisper"] = "faster-whisper"
    model_size: str = "small.en"
    device: str = "auto"
    compute_type: str = "auto"
    language: Optional[str] = None
    vad: bool = False
    vad_silence_ms: int = 350
    beam_size: int = 5


@dataclass
class RAGConfig:
    enabled: bool = True
    api_base: str = "http://127.0.0.1:8123"
    on_demand: bool = True
    k: int = 4
    timeout_ms: int = 400
    command_trigger: bool = True
    fallback_retry: bool = True
    # Codebase self-awareness: when true, code-related queries also
    # pull chunks from the code index (built by scripts/index_codebase.py).
    code_enabled: bool = True
    code_k: int = 3


def _validate_config(cfg: "AppConfig", path: str) -> None:
    """Sanity-check the loaded config and fail fast on problems.

    Raises ValueError if something that the runtime needs is missing or
    clearly misconfigured. Lets the server refuse to start rather than
    silently producing weird runtime errors later.
    """
    problems: list[str] = []

    # Tiered LLM: routing must map tier1/2/3 to model strings.
    if cfg.llm.provider == "tiered":
        routing = cfg.llm.routing or {}
        for tier in ("tier1", "tier2", "tier3"):
            if not routing.get(tier):
                problems.append(f"llm.routing.{tier} is required when llm.provider = 'tiered'")

    # Claude tier needs an API key either in config or env.
    if cfg.llm.provider == "claude" and not cfg.llm.claude_api_key:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            problems.append("llm.provider = 'claude' requires claude_api_key or ANTHROPIC_API_KEY env var")

    # Numeric sanity
    if cfg.llm.temperature < 0 or cfg.llm.temperature > 2:
        problems.append(f"llm.temperature = {cfg.llm.temperature} outside sensible [0, 2] range")

    if cfg.rag and cfg.rag.timeout_ms <= 0:
        problems.append(f"rag.timeout_ms must be > 0 (got {cfg.rag.timeout_ms})")

    # Piper path: only matters if provider actually uses it.
    if cfg.tts.provider == "piper":
        p = Path(cfg.tts.piper_executable)
        if not p.is_absolute():
            # Relative is fine — let the PATH / launcher resolve it.
            pass
        elif not p.exists():
            # Absolute but missing — the dev's machine-specific path
            # that wouldn't work after deployment. Warn rather than
            # fail (user might not be using piper right now).
            import logging
            logging.getLogger("mother.config").warning(
                "tts.piper_executable points to missing absolute path: %s", p,
            )

    if problems:
        raise ValueError(
            f"Invalid config in {path}:\n  - " + "\n  - ".join(problems)
        )


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    llm_data = data.get("llm", {})
    tts_data = data.get("tts", {})

    try:
        llm_cfg = LLMConfig(**_filter_fields(LLMConfig, llm_data))
    except TypeError as exc:
        raise ValueError(f"Invalid LLM config in {path}: {exc}") from exc

    try:
        tts_cfg = TTSConfig(**_filter_fields(TTSConfig, tts_data))
    except TypeError as exc:
        raise ValueError(f"Invalid TTS config in {path}: {exc}") from exc

    stt_section = data.get("stt")
    try:
        stt_cfg = STTConfig(**_filter_fields(STTConfig, stt_section)) if stt_section else None
    except TypeError as exc:
        raise ValueError(f"Invalid STT config in {path}: {exc}") from exc

    rag_section = data.get("rag")
    try:
        rag_cfg = RAGConfig(**_filter_fields(RAGConfig, rag_section)) if rag_section else None
    except TypeError as exc:
        raise ValueError(f"Invalid RAG config in {path}: {exc}") from exc

    cfg = AppConfig(llm=llm_cfg, tts=tts_cfg, stt=stt_cfg, rag=rag_cfg)
    _validate_config(cfg, path)
    return cfg


