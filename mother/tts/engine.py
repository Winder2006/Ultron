from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import text normalizer for natural speech
try:
    from mother.tts.normalizer import normalize_for_speech
except ImportError:
    # Fallback if running standalone
    normalize_for_speech = lambda x: x  # type: ignore


class TTSEngine:
    """Abstract base for TTS engines."""

    def synthesize_to_file(self, text: str, output_wav_path: str) -> str:
        raise NotImplementedError

    def synthesize_to_bytes(self, text: str) -> bytes:
        raise NotImplementedError


@dataclass
class KokoroConfig:
    voice: str = "bf_emma"       # Kokoro voice ID (bf_=British female, af_=American female, etc.)
    lang_code: str = "b"         # 'a'=American English, 'b'=British English
    speed: float = 1.0
    sample_rate: int = 24000


class KokoroTTSEngine(TTSEngine):
    """Kokoro TTS engine — 82M parameter model, fast on CPU, high quality."""

    def __init__(self, cfg: KokoroConfig) -> None:
        self.cfg = cfg
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code=self.cfg.lang_code)

    def synthesize_to_bytes(self, text: str) -> bytes:
        import io
        import wave
        import numpy as np
        text = normalize_for_speech(text)
        self._ensure_pipeline()
        # Collect all audio chunks from the generator
        chunks = []
        for _gs, _ps, audio in self._pipeline(
            text, voice=self.cfg.voice, speed=self.cfg.speed
        ):
            if audio is not None:
                chunks.append(audio)
        if not chunks:
            return b""
        combined = np.concatenate(chunks)
        # Write WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            pcm16 = np.clip(combined, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
            wf.writeframes(pcm16)
        return buf.getvalue()

    def synthesize_to_file(self, text: str, output_wav_path: str) -> str:
        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = self.synthesize_to_bytes(text)
        output_path.write_bytes(wav_bytes)
        return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Deepgram TTS — cloud-based, sub-second latency
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeepgramTTSConfig:
    """Deepgram Aura 2 TTS configuration.

    Popular voice models (all English):
      aura-2-thalia-en   — warm, natural female
      aura-2-asteria-en  — clear, friendly female
      aura-2-luna-en     — calm, measured female (MOTHER-appropriate)
      aura-2-helena-en   — authoritative female
      aura-2-stella-en   — confident female
      aura-2-athena-en   — clinical, precise female
      aura-2-hera-en     — commanding female
      aura-2-orion-en    — resonant male
      aura-2-arcas-en    — warm male

    For MOTHER's "calm, clinical, deliberate" tone, `aura-2-luna-en` or
    `aura-2-athena-en` are the best fits.
    """
    model: str = "aura-2-luna-en"
    api_key: Optional[str] = None  # defaults to DEEPGRAM_API_KEY env var
    sample_rate: int = 24000


class DeepgramTTSEngine(TTSEngine):
    """Deepgram Aura 2 TTS — cloud API, ~300ms latency, high quality.

    Pros over Kokoro:
      • 10x faster (300ms vs 3s for a sentence on CPU)
      • No model loading, no CPU/GPU required
      • Multiple premium voice options
    Cons:
      • Requires internet + DEEPGRAM_API_KEY
      • Uses credits ($0.015 per 1000 characters)
    """

    def __init__(self, cfg: DeepgramTTSConfig) -> None:
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from deepgram import DeepgramClient
        key = self.cfg.api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY not set. Add it to .env or pass api_key in config."
            )
        self._client = DeepgramClient(api_key=key)

    def _generate(self, text: str, container: str = "wav"):
        """Call Deepgram TTS and return the chunk generator (or None for empty input).

        container: "wav" for standalone playback, "none" for raw PCM streaming.
        """
        text = normalize_for_speech(text)
        if not text.strip():
            return None
        self._ensure_client()
        kwargs = dict(
            text=text,
            model=self.cfg.model,
            encoding="linear16",
            sample_rate=self.cfg.sample_rate,
            container=container,  # "wav" or "none" — must be explicit for raw PCM
        )
        return self._client.speak.v1.audio.generate(**kwargs)

    def warmup(self) -> None:
        """Prime the Deepgram client and its connection pool.

        Without this, the first real synth pays a 100-200ms cold-start
        penalty while the SDK sets up the client, does DNS, and
        establishes TLS to api.deepgram.com. Running a tiny throwaway
        synth at server startup folds that cost into the launch
        sequence instead of the first user turn. Best-effort — any
        failure (network out at boot, bad key) is logged but not fatal.
        """
        try:
            self._ensure_client()
            # Tiny payload so warmup finishes fast and costs ~nothing
            # on Deepgram's per-character billing.
            response = self._generate(".", container="wav")
            if response is None:
                return
            # Drain the response so the underlying connection fully
            # completes — otherwise the warmup doesn't actually prime
            # the pool.
            if hasattr(response, "stream"):
                response.stream.getvalue()
            else:
                for _ in response:
                    pass
        except Exception:
            # Silent — warmup is an optimization, not a correctness
            # requirement. Logged at debug since this ran at startup
            # before any response-critical path.
            pass

    def synthesize_to_bytes(self, text: str) -> bytes:
        response = self._generate(text, container="wav")
        if response is None:
            return b""
        # Older SDK versions expose `.stream.getvalue()`; newer versions return a generator.
        if hasattr(response, "stream"):
            return response.stream.getvalue()
        return b"".join(chunk for chunk in response)

    def synthesize_stream_pcm(self, text: str):
        """Yield raw PCM (linear16 mono 24kHz) chunks as they arrive.

        No WAV header — the frontend plays chunks directly via Web Audio API.
        Time-to-first-chunk is ~150ms on warm connections (4× faster than
        waiting for the full WAV to download).
        """
        response = self._generate(text, container="none")
        if response is None:
            return
        if hasattr(response, "stream"):
            yield response.stream.getvalue()
            return
        for chunk in response:
            if chunk:
                yield chunk

    def synthesize_to_file(self, text: str, output_wav_path: str) -> str:
        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = self.synthesize_to_bytes(text)
        output_path.write_bytes(wav_bytes)
        return str(output_path)


@dataclass
class ChatterboxConfig:
    voice_profile: str = "./tts/voice_profiles/ultron_reference.wav"
    exaggeration: float = 0.45    # Emotion intensity: 0.0=flat, 1.0=dramatic. Ultron: 0.4-0.55
    cfg_weight: float = 0.5       # Voice clone adherence: higher=more like reference
    sample_rate: int = 24000


class ChatterboxTTSEngine(TTSEngine):
    """Chatterbox TTS engine — 350M param model with zero-shot voice cloning.

    Uses a reference WAV to clone any voice. Runs on CUDA if available (Jetson GPU),
    falls back to CPU. If Chatterbox fails to load, caller should fall back to Kokoro.
    """

    def __init__(self, cfg: ChatterboxConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from chatterbox.tts import ChatterboxTTS

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ChatterboxTTS.from_pretrained(device=self._device)

        # Pre-warm with a short synthesis to avoid cold-start latency
        ref = self.cfg.voice_profile
        if Path(ref).exists():
            try:
                self._model.generate(
                    "Initializing.", audio_prompt_path=ref,
                    exaggeration=self.cfg.exaggeration,
                    cfg_weight=self.cfg.cfg_weight,
                )
            except Exception:
                pass

    def synthesize_to_bytes(self, text: str) -> bytes:
        import io
        import wave
        import torch
        import numpy as np

        text = normalize_for_speech(text)
        self._ensure_model()

        ref_path = self.cfg.voice_profile
        kwargs = {
            "exaggeration": self.cfg.exaggeration,
            "cfg_weight": self.cfg.cfg_weight,
        }
        if Path(ref_path).exists():
            kwargs["audio_prompt_path"] = ref_path

        wav_tensor = self._model.generate(text, **kwargs)

        # Convert torch tensor to numpy
        if isinstance(wav_tensor, torch.Tensor):
            audio = wav_tensor.squeeze().cpu().numpy()
        else:
            audio = np.array(wav_tensor, dtype=np.float32)

        # Normalize to [-1, 1]
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        # Write WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.sample_rate)
            pcm16 = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
            wf.writeframes(pcm16)
        return buf.getvalue()

    def synthesize_to_file(self, text: str, output_wav_path: str) -> str:
        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav_bytes = self.synthesize_to_bytes(text)
        output_path.write_bytes(wav_bytes)
        return str(output_path)


@dataclass
class PiperConfig:
    piper_executable: str
    model_path: str
    config_path: str
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sentence_silence: float = 0.0
    backend: str = "subprocess"  # "subprocess" | "python"
    persistent: bool = True


class PiperTTSEngine(TTSEngine):
    """Piper TTS wrapper using subprocess for low-latency local synthesis.

    Requires a Piper binary (`piper.exe` on Windows) and voice files (.onnx and .json).
    """

    def __init__(self, cfg: PiperConfig) -> None:
        import threading
        self.cfg = cfg
        self._persistent_model = None
        self._persistent_process = None
        self._io_lock = threading.Lock()
        # Backend preference
        self._backend = (os.environ.get("PIPER_BACKEND") or cfg.backend or "subprocess").lower()
        if self._backend == "python":
            try:
                import piper  # type: ignore
            except Exception:
                self._backend = "subprocess"

    def _base_args(self) -> list[str]:
        return [
            self.cfg.piper_executable,
            "-m",
            self.cfg.model_path,
            "-c",
            self.cfg.config_path,
            "--sentence_silence",
            str(self.cfg.sentence_silence),
            "--length_scale",
            str(self.cfg.length_scale),
            "--noise_scale",
            str(self.cfg.noise_scale),
            "--noise_w",
            str(self.cfg.noise_w),
        ]

    def synthesize_to_file(self, text: str, output_wav_path: str) -> str:
        # Normalize text for natural speech (dates, numbers, abbreviations)
        text = normalize_for_speech(text)
        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # One-shot subprocess path produces a valid WAV via -f <path>.
        # We prefer this for file outputs to avoid container/format mismatches.
        args = [*self._base_args(), "-f", str(output_path)]
        proc = subprocess.run(
            args,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Piper failed ({proc.returncode}): {proc.stderr.decode(errors='ignore')}"
            )
        return str(output_path)

    def synthesize_to_bytes(self, text: str) -> bytes:
        # Normalize text for natural speech (dates, numbers, abbreviations)
        text = normalize_for_speech(text)
        # If we have a persistent python model, use it
        if self.cfg.persistent and self._backend == "python" and self._ensure_python_model():
            return self._python_synthesize(text)
        # Persistent subprocess path
        if self.cfg.persistent and self._backend == "subprocess" and self._ensure_persistent_proc():
            return self._subprocess_synthesize(text)
        # One-shot subprocess fallback
        args = self._base_args()
        proc = subprocess.run(
            args,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Piper failed ({proc.returncode}): {proc.stderr.decode(errors='ignore')}"
            )
        return proc.stdout

    # -------- python backend helpers --------
    def _ensure_python_model(self) -> bool:
        if self._persistent_model is not None:
            return True
        try:
            import piper  # type: ignore
            from piper.voice import PiperVoice  # type: ignore
            model_path = self.cfg.model_path
            config_path = self.cfg.config_path
            self._persistent_model = PiperVoice.load(model_path, config_path)  # type: ignore
            return True
        except Exception:
            self._persistent_model = None
            self._backend = "subprocess"
            return False

    def _python_synthesize(self, text: str) -> bytes:
        # Produce wav bytes using the loaded model
        import io
        import wave
        import numpy as np
        voice = self._persistent_model
        if voice is None:
            raise RuntimeError("Piper python backend not initialized")
        audio, sr = voice.synthesize(text, length_scale=self.cfg.length_scale, noise_scale=self.cfg.noise_scale, noise_w=self.cfg.noise_w)  # type: ignore
        # audio is float32 numpy array [-1,1]; write a minimal WAV header + data
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            pcm16 = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
            wf.writeframes(pcm16)
        return buf.getvalue()

    # -------- persistent subprocess helpers --------
    def _ensure_persistent_proc(self) -> bool:
        if self._persistent_process and self._persistent_process.poll() is None:
            return True
        try:
            # Output to stdout with -f - so we can read one WAV per input line
            args = [*self._base_args(), "-f", "-"]
            self._persistent_process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            return True
        except Exception:
            self._persistent_process = None
            return False

    def _subprocess_synthesize(self, text: str) -> bytes:
        import struct
        import io
        if not self._persistent_process or not self._persistent_process.stdin or not self._persistent_process.stdout:
            raise RuntimeError("Piper persistent process not running")
        proc = self._persistent_process
        with self._io_lock:
            # Send line
            proc.stdin.write((text + "\n").encode("utf-8"))
            proc.stdin.flush()
            # Read RIFF header: 4 bytes "RIFF" + 4 bytes size
            riff_header = proc.stdout.read(8)
            if not riff_header or len(riff_header) < 8:
                raise RuntimeError("Failed to read RIFF header from Piper")
            if riff_header[:4] != b"RIFF":
                raise RuntimeError(f"Invalid RIFF header: {riff_header[:4]!r}")
            total_size = struct.unpack("<I", riff_header[4:8])[0]
            # Read the rest of the WAV (total_size bytes after the 8-byte RIFF header)
            remaining = proc.stdout.read(total_size)
            if len(remaining) < total_size:
                raise RuntimeError("Truncated WAV data from Piper")
            return riff_header + remaining


