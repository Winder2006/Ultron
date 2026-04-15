from __future__ import annotations

from mother.llm.drivers import OllamaLLMDriver, ClaudeLLMDriver, HybridLLMDriver, TieredLLMDriver
from mother.tts.engine import (
    PiperConfig, PiperTTSEngine,
    KokoroConfig, KokoroTTSEngine,
    ChatterboxConfig, ChatterboxTTSEngine,
    DeepgramTTSConfig, DeepgramTTSEngine,
)
from mother.config.settings import AppConfig
from mother.audio.stt import FasterWhisperConfig, FasterWhisperSTT


def build_drivers(cfg: AppConfig):
    # LLM
    if cfg.llm.provider == "tiered":
        tier_models = {}
        routing = getattr(cfg.llm, "routing", None)
        if routing and isinstance(routing, dict):
            tier_models = {k: v for k, v in routing.items() if k.startswith("tier")}
        llm = TieredLLMDriver(tier_models=tier_models or None)
    elif cfg.llm.provider == "claude":
        llm = ClaudeLLMDriver(
            model=cfg.llm.claude_model,
            api_key=cfg.llm.claude_api_key,
            max_tokens=cfg.llm.claude_max_tokens,
        )
    elif cfg.llm.provider == "hybrid":
        local = OllamaLLMDriver(
            model=cfg.llm.model,
            base_url=cfg.llm.base_url,
            keep_alive=cfg.llm.keep_alive,
            num_thread=cfg.llm.num_thread,
        )
        cloud = ClaudeLLMDriver(
            model=cfg.llm.claude_model,
            api_key=cfg.llm.claude_api_key,
            max_tokens=cfg.llm.claude_max_tokens,
        )
        llm = HybridLLMDriver(local=local, cloud=cloud)
    elif cfg.llm.provider == "ollama":
        llm = OllamaLLMDriver(
            model=cfg.llm.model,
            base_url=cfg.llm.base_url,
            keep_alive=cfg.llm.keep_alive,
            num_thread=cfg.llm.num_thread,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {cfg.llm.provider}")

    # TTS
    if cfg.tts.provider == "deepgram":
        try:
            dg_cfg = DeepgramTTSConfig(
                model=getattr(cfg.tts, "deepgram_model", "aura-2-luna-en"),
                sample_rate=getattr(cfg.tts, "deepgram_sample_rate", 24000),
            )
            tts = DeepgramTTSEngine(dg_cfg)
        except Exception as _dg_err:
            import logging
            logging.getLogger("mother.tts").warning(
                "[TTS] Deepgram unavailable (%s) — falling back to Kokoro", _dg_err
            )
            kokoro_cfg = KokoroConfig(
                voice=cfg.tts.kokoro_voice,
                lang_code=cfg.tts.kokoro_lang_code,
                speed=cfg.tts.kokoro_speed,
            )
            tts = KokoroTTSEngine(kokoro_cfg)
    elif cfg.tts.provider == "chatterbox":
        try:
            cb_cfg = ChatterboxConfig(
                voice_profile=getattr(cfg.tts, "chatterbox_voice_profile", "./tts/voice_profiles/ultron_reference.wav"),
                exaggeration=getattr(cfg.tts, "chatterbox_exaggeration", 0.45),
                cfg_weight=getattr(cfg.tts, "chatterbox_cfg_weight", 0.5),
            )
            tts = ChatterboxTTSEngine(cb_cfg)
        except Exception as _cb_err:
            import logging
            logging.getLogger("mother.tts").warning(
                "[TTS] Chatterbox unavailable (%s) — falling back to Kokoro (bf_emma voice)", _cb_err
            )
            kokoro_cfg = KokoroConfig(
                voice=cfg.tts.kokoro_voice,
                lang_code=cfg.tts.kokoro_lang_code,
                speed=cfg.tts.kokoro_speed,
            )
            tts = KokoroTTSEngine(kokoro_cfg)
    elif cfg.tts.provider == "kokoro":
        kokoro_cfg = KokoroConfig(
            voice=cfg.tts.kokoro_voice,
            lang_code=cfg.tts.kokoro_lang_code,
            speed=cfg.tts.kokoro_speed,
        )
        tts = KokoroTTSEngine(kokoro_cfg)
    elif cfg.tts.provider == "piper":
        # Map app-level backend to PiperTTSEngine expected values
        backend = getattr(cfg.tts, "backend", "piper_subprocess")
        backend_mapped = "python" if backend == "piper_python" else "subprocess"
        piper_cfg = PiperConfig(
            piper_executable=cfg.tts.piper_executable,
            model_path=cfg.tts.model_path,
            config_path=cfg.tts.config_path,
            length_scale=cfg.tts.length_scale,
            noise_scale=cfg.tts.noise_scale,
            noise_w=cfg.tts.noise_w,
            sentence_silence=getattr(cfg.tts, "sentence_silence", 0.0),
            backend=backend_mapped,
            persistent=getattr(cfg.tts, "persistent", True),
        )
        tts = PiperTTSEngine(piper_cfg)
    else:
        raise ValueError(f"Unsupported TTS provider: {cfg.tts.provider}")

    stt = None
    if cfg.stt is not None:
        if cfg.stt.provider == "faster-whisper":
            stt_cfg = FasterWhisperConfig(
                model_size=cfg.stt.model_size,
                device=cfg.stt.device,
                compute_type=cfg.stt.compute_type,
                language=cfg.stt.language,
                beam_size=getattr(cfg.stt, "beam_size", 5),
            )
            stt = FasterWhisperSTT(stt_cfg)
        else:
            raise ValueError(f"Unsupported STT provider: {cfg.stt.provider}")

    return llm, tts, stt


