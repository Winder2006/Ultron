/**
 * Browser-side voice activity detection using Silero VAD (ONNX).
 *
 * Used during ACTIVE RECORDING to detect end-of-speech and auto-stop
 * the mic without waiting for the server's STT to return a final
 * transcript. Fixes the "5-second ceiling fires every time" problem
 * on wake-word-triggered turns: we now know the user is done talking
 * ~500ms after their last syllable, not 1.5s after the server finishes
 * a batch REST call.
 *
 * Silero takes 30ms or 60ms frames at 16kHz (480 or 960 samples). We
 * use 32ms (512 samples) which matches what the model was trained on
 * per the official docs and is a good fit for our AudioWorklet render
 * quantum.
 *
 * State machine:
 *
 *     [idle]
 *        │  first frame with VAD prob > speech_thresh
 *        ▼
 *     [speaking] ─── silence < silence_thresh_ms ───► [speaking]
 *        │
 *        │  silence ≥ silence_thresh_ms AND
 *        │  had at least min_speech_ms of speech earlier
 *        ▼
 *     [ended] ── fires onEndOfSpeech callback
 *
 * Deliberately conservative: we won't fire end-of-speech until there's
 * been at least some real speech in the buffer (prevents firing on
 * the startup silence before the user's first word).
 */

import type { InferenceSession, Tensor as OrtTensor } from 'onnxruntime-web';

// Silero VAD model input spec (v5 ONNX export):
//   input: shape [batch, samples] — float32 audio at 16kHz
//   sr:    shape []               — int64 sample rate (16000)
//   h:     shape [2, batch, 64]   — float32 LSTM hidden state
//   c:     shape [2, batch, 64]   — float32 LSTM cell state
// It outputs:
//   output: shape [batch, 1]      — float32 speech probability 0..1
//   hn:     shape [2, batch, 64]  — updated hidden state
//   cn:     shape [2, batch, 64]  — updated cell state
const FRAME_SAMPLES = 512;  // 32ms at 16kHz — Silero-native frame size
const SAMPLE_RATE = 16000;

export interface VADConfig {
  /** Probability above which a frame counts as speech (0..1). 0.5 is Silero's suggested default. */
  speechThreshold?: number;
  /** Consecutive silence (ms) needed before end-of-speech fires. */
  silenceDurationMs?: number;
  /** Minimum speech duration (ms) before end-of-speech CAN fire. Prevents
   *  firing on mere background noise or a single fricative. */
  minSpeechMs?: number;
  /** Maximum recording duration (ms) regardless of VAD state — safety net. */
  maxDurationMs?: number;
  /** Callback when end-of-speech is detected. */
  onEndOfSpeech: () => void;
  /** Optional: callback whenever speech is first detected (for UI). */
  onSpeechStart?: () => void;
}


type OrtModule = typeof import('onnxruntime-web');

export class RecordingVAD {
  private session: InferenceSession | null = null;
  private ort: OrtModule | null = null;  // cached after load()
  private h: Float32Array;             // LSTM hidden state [2, 1, 64]
  private c: Float32Array;             // LSTM cell state   [2, 1, 64]
  private residual: Float32Array;      // samples carried between calls

  private speechThreshold: number;
  private silenceThresholdMs: number;
  private minSpeechMs: number;
  private maxDurationMs: number;
  private onEndOfSpeech: () => void;
  private onSpeechStart?: () => void;

  private isLoading = false;
  private loaded = false;

  // State machine
  private speakingMs = 0;
  private silenceMs = 0;
  private totalMs = 0;
  private everSpoke = false;
  private ended = false;

  constructor(cfg: VADConfig) {
    this.speechThreshold = cfg.speechThreshold ?? 0.5;
    this.silenceThresholdMs = cfg.silenceDurationMs ?? 800;
    this.minSpeechMs = cfg.minSpeechMs ?? 350;
    this.maxDurationMs = cfg.maxDurationMs ?? 10000;
    this.onEndOfSpeech = cfg.onEndOfSpeech;
    this.onSpeechStart = cfg.onSpeechStart;
    this.h = new Float32Array(2 * 1 * 64);
    this.c = new Float32Array(2 * 1 * 64);
    this.residual = new Float32Array(0);
  }

  /**
   * Preload the model. Safe to call once at app startup so the first
   * real recording doesn't pay the 50-150ms cold-start cost.
   */
  async load(): Promise<void> {
    if (this.loaded || this.isLoading) return;
    this.isLoading = true;
    try {
      const ort = await import('onnxruntime-web');
      this.ort = ort;
      // Point ORT at the .wasm files in /public/ort/ — same config the
      // wake-word uses. Setting it here too makes the VAD hook
      // independent of wake-word initialization order.
      if (!ort.env.wasm.wasmPaths) {
        ort.env.wasm.wasmPaths = {
          wasm: '/ort/ort-wasm-simd-threaded.jsep.wasm',
          mjs: '/ort/ort-wasm-simd-threaded.jsep.mjs',
        };
      }
      ort.env.wasm.numThreads = 1;
      this.session = await ort.InferenceSession.create(
        '/models/silero_vad.onnx',
        { executionProviders: ['wasm'] },
      );
      this.loaded = true;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Adjust the end-of-speech silence window on the fly. Used for
   * adaptive endpointing: the live transcript tells us whether the
   * utterance already reads as complete (stop sooner) or trails off
   * mid-clause (wait longer). Takes effect on the next frame.
   */
  setSilenceDurationMs(ms: number): void {
    this.silenceThresholdMs = ms;
  }

  /**
   * Swap the end-of-speech callback without rebuilding the model.
   * Lets the caller reuse a loaded RecordingVAD across recordings even
   * when the React closure identity changes — otherwise we'd
   * instantiate a new VAD (and re-download the ONNX) on every turn.
   */
  setCallback(fn: () => void): void {
    this.onEndOfSpeech = fn;
  }

  /** Reset state machine — call at the start of each new recording. */
  reset(): void {
    this.h.fill(0);
    this.c.fill(0);
    this.residual = new Float32Array(0);
    this.speakingMs = 0;
    this.silenceMs = 0;
    this.totalMs = 0;
    this.everSpoke = false;
    this.ended = false;
  }

  /**
   * Feed one chunk of Float32 PCM samples at 16kHz. The chunk can be
   * any length — we buffer until we have enough for a Silero frame
   * (512 samples) and run inference on each complete frame.
   *
   * Callback fires synchronously on each end-of-speech detection. After
   * the first fire, subsequent calls to process() are no-ops until you
   * reset().
   */
  async process(samples: Float32Array): Promise<void> {
    if (this.ended || !this.loaded || !this.session) return;

    // Concatenate residual + new samples.
    const combined = new Float32Array(this.residual.length + samples.length);
    combined.set(this.residual, 0);
    combined.set(samples, this.residual.length);

    // Process as many complete Silero frames as we can; carry the
    // remainder forward.
    let offset = 0;
    while (offset + FRAME_SAMPLES <= combined.length) {
      const frame = combined.subarray(offset, offset + FRAME_SAMPLES);
      offset += FRAME_SAMPLES;
      await this.runFrame(frame);
      if (this.ended) break;
    }
    this.residual = combined.subarray(offset);
  }

  private async runFrame(frame: Float32Array): Promise<void> {
    if (!this.session || !this.ort || this.ended) return;
    // Module reference is cached at load() time so we don't pay the
    // dynamic-import lookup tax on every 32ms frame (~156 forced
    // microtask yields per 5-second utterance otherwise).
    const ort = this.ort;

    const inputTensor = new ort.Tensor('float32', frame, [1, FRAME_SAMPLES]);
    const srTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(SAMPLE_RATE)]), []);
    const hTensor = new ort.Tensor('float32', this.h, [2, 1, 64]);
    const cTensor = new ort.Tensor('float32', this.c, [2, 1, 64]);

    let output: Record<string, OrtTensor>;
    try {
      output = await this.session.run({
        input: inputTensor,
        sr: srTensor,
        h: hTensor,
        c: cTensor,
      });
    } catch (err) {
      console.warn('[vad] inference error:', err);
      return;
    }

    // Update LSTM state for next frame.
    if (output.hn) this.h = output.hn.data as Float32Array;
    if (output.cn) this.c = output.cn.data as Float32Array;
    const probTensor = output.output;
    const prob = probTensor ? Number((probTensor.data as Float32Array)[0]) : 0;

    // Each frame is 32ms (512/16000). Advance the state machine.
    const frameMs = (FRAME_SAMPLES / SAMPLE_RATE) * 1000;
    this.totalMs += frameMs;

    if (prob >= this.speechThreshold) {
      if (!this.everSpoke) {
        this.everSpoke = true;
        this.onSpeechStart?.();
      }
      this.speakingMs += frameMs;
      this.silenceMs = 0;
    } else {
      this.silenceMs += frameMs;
    }

    // End conditions:
    //   (1) Long enough speech + enough trailing silence, OR
    //   (2) Max duration hit — always stop
    const speechFinished =
      this.everSpoke
      && this.speakingMs >= this.minSpeechMs
      && this.silenceMs >= this.silenceThresholdMs;
    const hardLimit = this.totalMs >= this.maxDurationMs;

    if (speechFinished || hardLimit) {
      this.ended = true;
      try {
        this.onEndOfSpeech();
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn('[vad] onEndOfSpeech callback error:', err);
      }
    }
  }
}
