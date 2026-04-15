"""Minimal Chatterbox test with aggressive flushing so we see progress on CPU.

Usage:
    python -u scripts/test_chatterbox_minimal.py

(Note: use -u for unbuffered output so we see each step live.)
"""
import sys
import time
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("step 1: importing torch...")
    import torch
    log(f"  torch {torch.__version__}, cuda={torch.cuda.is_available()}")

    log("step 2: importing chatterbox...")
    from chatterbox.tts import ChatterboxTTS
    log("  imported")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"step 3: loading model on {device} (first run ~30-90s on CPU)...")
    t0 = time.monotonic()
    model = ChatterboxTTS.from_pretrained(device=device)
    log(f"  loaded in {time.monotonic() - t0:.1f}s")

    ref_path = "tts/voice_profiles/ultron_reference.wav"
    log(f"step 4: synthesizing with reference {ref_path!r}...")
    text = "Affirmative. Systems operational."
    log(f"  text: {text!r}")
    t0 = time.monotonic()
    wav = model.generate(
        text,
        audio_prompt_path=ref_path,
        exaggeration=0.45,
        cfg_weight=0.5,
    )
    log(f"  synth took {time.monotonic() - t0:.1f}s")
    log(f"  output shape: {wav.shape if hasattr(wav, 'shape') else type(wav)}")

    log("step 5: saving output to out/ultron_test.wav...")
    import torchaudio
    out_path = Path("out/ultron_test.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # wav is likely a torch tensor (1, N) at model.sr
    if hasattr(wav, 'shape') and wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(out_path), wav.cpu(), sample_rate=getattr(model, "sr", 24000))
    log(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
