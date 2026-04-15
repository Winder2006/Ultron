/**
 * Always-on wake word detection running entirely in the browser.
 *
 * Uses openwakeword-js (a JS/TS port of openWakeWord) via ONNX Runtime
 * Web. Pipeline:
 *
 *   getUserMedia (16kHz mono Float32)
 *     → AudioWorklet (frame accumulator → 1280-sample / 80ms chunks)
 *     → openwakeword-js Model.predict()
 *     → on score > threshold: fire onDetected()
 *
 * Deploys to Railway unchanged — 100% client-side, no server audio.
 *
 * Self-trigger guard:
 *   The hook takes an `isSuppressed` function so the caller can disable
 *   detection while Ultron is speaking (TTS playing) or while the user
 *   is already recording. Without this, Ultron's own voice saying
 *   "Hey Jarvis" in conversation (or any phonetically similar phrase)
 *   would cause a feedback loop.
 *
 * Model files expected at (served by Vite from dashboard/public/):
 *   /models/melspectrogram.onnx
 *   /models/embedding_model.onnx
 *   /models/silero_vad.onnx
 *   /models/hey_jarvis_v0.1.onnx
 *
 * ORT WASM files expected at /ort/ort-wasm-simd-threaded.{wasm,mjs}.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

// ONNX Runtime Web env config — set BEFORE openwakeword-js imports
// and initializes ORT. wasmPaths uses the object form so ORT fetches
// the files directly (the string-prefix form causes Vite to try to
// import() the .mjs as a module, which it refuses for files in
// public/).
async function configureORT(): Promise<void> {
  const ort = await import('onnxruntime-web');
  const wasmURL = new URL('/ort/ort-wasm-simd-threaded.jsep.wasm', window.location.origin).href;
  const mjsURL = new URL('/ort/ort-wasm-simd-threaded.jsep.mjs', window.location.origin).href;
  ort.env.wasm.wasmPaths = {
    wasm: wasmURL,
    mjs: mjsURL,
  };
  ort.env.wasm.numThreads = 1;
  // Diagnostic: verify what we set and confirm the files are reachable
  // with the right MIME. If either of these checks fails, that's the
  // real bug.
  // eslint-disable-next-line no-console
  console.log('[wake] ORT wasmPaths set to:', ort.env.wasm.wasmPaths);
  try {
    const r = await fetch(wasmURL, { method: 'HEAD' });
    // eslint-disable-next-line no-console
    console.log(
      `[wake] wasm HEAD → ${r.status} ${r.headers.get('content-type')} ` +
      `(content-length=${r.headers.get('content-length')})`,
    );
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error('[wake] wasm HEAD failed:', e);
  }
  try {
    const r = await fetch(mjsURL, { method: 'HEAD' });
    // eslint-disable-next-line no-console
    console.log(
      `[wake] mjs HEAD → ${r.status} ${r.headers.get('content-type')} ` +
      `(content-length=${r.headers.get('content-length')})`,
    );
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error('[wake] mjs HEAD failed:', e);
  }
}


export interface WakeWordOptions {
  /** Raised when the wake word fires. */
  onDetected?: (keyword: string, score: number) => void;
  /** Detection threshold 0..1. Default 0.5. Lower = more sensitive (more
   *  false alerts), higher = stricter (more misses). */
  threshold?: number;
  /** Seconds between consecutive detections. Default 2s — stops a
   *  single utterance from firing multiple times. */
  debounceSeconds?: number;
  /** Called before each predict() — if it returns true, the frame is
   *  skipped (no detection possible). Use this to suppress the wake
   *  word while Ultron is speaking or while already recording. */
  isSuppressed?: () => boolean;
}


export interface WakeWordState {
  /** True when the wake-word listener is actively running. */
  listening: boolean;
  /** True during the async model/mic initialization. */
  initializing: boolean;
  /** Last error, if any. Cleared when listening starts successfully. */
  error: string | null;
  /** Epoch seconds of the last detection, or 0 if none this session. */
  lastDetectionTs: number;
}


// Inline AudioWorklet source. Accumulates 16kHz Float32 samples into
// 1280-sample (80ms) frames and posts them to the main thread. Having
// this inline (rather than a separate .js file) means we don't need
// another static asset — the Blob URL approach works anywhere.
const WORKLET_SOURCE = `
class FrameAccumulatorProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 80ms @ 16kHz = 1280 samples. openWakeWord's required chunk size.
    this.frameSize = 1280;
    this.buffer = new Float32Array(this.frameSize);
    this.writeIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];
    let offset = 0;
    while (offset < channel.length) {
      const copy = Math.min(
        this.frameSize - this.writeIndex,
        channel.length - offset,
      );
      this.buffer.set(
        channel.subarray(offset, offset + copy),
        this.writeIndex,
      );
      this.writeIndex += copy;
      offset += copy;
      if (this.writeIndex >= this.frameSize) {
        // Copy because the underlying buffer gets overwritten on the
        // next frame — main thread needs its own buffer.
        this.port.postMessage({ type: 'frame', samples: this.buffer.slice() });
        this.writeIndex = 0;
      }
    }
    return true;
  }
}
registerProcessor('wake-frame-accumulator', FrameAccumulatorProcessor);
`;


function createWorkletBlobURL(): string {
  return URL.createObjectURL(
    new Blob([WORKLET_SOURCE], { type: 'application/javascript' }),
  );
}


/**
 * React hook for browser-side wake word detection.
 *
 * Usage:
 *   const wake = useWakeWord({
 *     threshold: 0.5,
 *     isSuppressed: () => voice.recording || voice.responding,
 *     onDetected: (kw, score) => voice.startRecording(),
 *   });
 *   wake.start();   // begin listening
 *   wake.stop();    // teardown
 */
export function useWakeWord(opts: WakeWordOptions = {}) {
  const {
    threshold = 0.5,
    debounceSeconds = 2.0,
    onDetected,
    isSuppressed,
  } = opts;

  const [state, setState] = useState<WakeWordState>({
    listening: false,
    initializing: false,
    error: null,
    lastDetectionTs: 0,
  });

  // All the live handles that need cleanup on stop().
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const blobURLRef = useRef<string | null>(null);
  // `any` because the openwakeword-js types export it but we import it
  // dynamically to keep it out of the initial bundle.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRef = useRef<any>(null);

  // Refs for anything the worklet-message closure reads. Using refs
  // avoids re-binding the handler every render.
  const thresholdRef = useRef(threshold);
  const debounceRef = useRef(debounceSeconds);
  const lastFireRef = useRef(0);
  const onDetectedRef = useRef(onDetected);
  const isSuppressedRef = useRef(isSuppressed);

  useEffect(() => { thresholdRef.current = threshold; }, [threshold]);
  useEffect(() => { debounceRef.current = debounceSeconds; }, [debounceSeconds]);
  useEffect(() => { onDetectedRef.current = onDetected; }, [onDetected]);
  useEffect(() => { isSuppressedRef.current = isSuppressed; }, [isSuppressed]);


  const start = useCallback(async () => {
    if (state.listening || state.initializing) return;
    setState((s) => ({ ...s, initializing: true, error: null }));

    try {
      // Always reconfigure — cheap, and ensures a retry after error
      // actually re-applies the WASM path settings.
      await configureORT();

      // Import openwakeword-js lazily so it doesn't bloat the main bundle.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { Model }: any = await import('openwakeword-js');

      // Build the model. openwakeword-js handles the full pipeline:
      // mel → embedding → per-word classifier, all async ONNX calls.
      // Pass wasmPaths into the Model constructor — the library sets
      // it on its OWN ORT instance (which differs from the one we get
      // via dynamic import when Vite code-splits). Without this, our
      // env-level configureORT() call is setting env on a different
      // instance than the one that actually loads the WASM.
      const wasmURL = new URL('/ort/ort-wasm-simd-threaded.jsep.wasm', window.location.origin).href;
      const mjsURL = new URL('/ort/ort-wasm-simd-threaded.jsep.mjs', window.location.origin).href;
      const model = new Model({
        wakewordModels: ['/models/hey_jarvis_v0.1.onnx'],
        melspectrogramModelPath: '/models/melspectrogram.onnx',
        embeddingModelPath: '/models/embedding_model.onnx',
        vadModelPath: '/models/silero_vad.onnx',
        // VAD threshold of 0.5 gates prediction on voice-likely frames.
        // Cuts CPU use and false positives from non-speech noise.
        vadThreshold: 0.5,
        inferenceFramework: 'onnx',
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        wasmPaths: { wasm: wasmURL, mjs: mjsURL } as any,
        debounceTime: debounceRef.current,
      });
      await model.init();
      modelRef.current = model;

      // Open the mic. 16kHz is critical — the melspec model expects it
      // and resampling in the browser is lossy.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      // Create a dedicated AudioContext at 16kHz. The default context
      // runs at 48kHz; forcing 16kHz avoids a resample step.
      const ctx = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = ctx;

      // Load the inline frame-accumulator worklet via a Blob URL.
      const blobURL = createWorkletBlobURL();
      blobURLRef.current = blobURL;
      await ctx.audioWorklet.addModule(blobURL);

      const source = ctx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(ctx, 'wake-frame-accumulator', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
      });
      workletNodeRef.current = workletNode;
      source.connect(workletNode);

      // Wire the main-thread frame handler. This is the hot path that
      // runs every 80ms while listening.
      workletNode.port.onmessage = async (ev) => {
        if (ev.data?.type !== 'frame') return;

        // Suppression: caller's way of saying "don't fire right now"
        // (TTS playing, recording active, etc.). Skip the whole
        // predict() call to save CPU.
        if (isSuppressedRef.current?.()) return;

        const samples: Float32Array = ev.data.samples;
        const m = modelRef.current;
        if (!m) return;

        try {
          const result = await m.predict(samples);
          // Diagnostic: log any non-trivial score so we can see
          // what the model hears even when below threshold. Logs
          // at most every ~1s (12 frames) to avoid spam.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (globalThis as any).__wakeDiagFrameIdx =
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ((globalThis as any).__wakeDiagFrameIdx || 0) + 1;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const idx = (globalThis as any).__wakeDiagFrameIdx;
          // Result is { 'hey_jarvis_v0.1': 0..1 } — scan for any
          // key that exceeds the threshold. Easier than knowing
          // the exact key name in case the library renames it.
          let bestScore = 0;
          let bestKey = '';
          for (const [keyword, rawScore] of Object.entries(result || {})) {
            const score = typeof rawScore === 'number' ? rawScore : 0;
            if (score > bestScore) {
              bestScore = score;
              bestKey = keyword;
            }
            if (score < thresholdRef.current) continue;
            const nowSec = Date.now() / 1000;
            if (nowSec - lastFireRef.current < debounceRef.current) break;
            lastFireRef.current = nowSec;
            setState((s) => ({ ...s, lastDetectionTs: nowSec }));
            // eslint-disable-next-line no-console
            console.log(`[wake] FIRED ${keyword}=${score.toFixed(3)}`);
            onDetectedRef.current?.(keyword, score);
            try { m.reset(); } catch { /* older versions lack reset */ }
            break;
          }
          // Log every ~1 second (12 frames) or whenever score > 0.1
          if (idx % 12 === 0 || bestScore > 0.1) {
            // eslint-disable-next-line no-console
            console.log(
              `[wake] frame ${idx} ${bestKey}=${bestScore.toFixed(3)} (threshold=${thresholdRef.current})`,
            );
          }
        } catch (err) {
          // Don't tear down on per-frame errors. Log and carry on —
          // one dropped prediction isn't worth killing the session.
          // eslint-disable-next-line no-console
          console.warn('[wake] predict error:', err);
        }
      };

      // eslint-disable-next-line no-console
      console.log('[wake] listening — say "Hey Jarvis"');
      setState({
        listening: true,
        initializing: false,
        error: null,
        lastDetectionTs: 0,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // eslint-disable-next-line no-console
      console.error('[wake] start failed:', err);
      setState({
        listening: false,
        initializing: false,
        error: message,
        lastDetectionTs: 0,
      });
    }
  }, [state.listening, state.initializing]);


  const stop = useCallback(() => {
    // Tear down in reverse creation order so we never leave a dangling
    // handle feeding audio into a disconnected graph.
    try { workletNodeRef.current?.disconnect(); } catch { /* ignore */ }
    workletNodeRef.current = null;

    try { audioContextRef.current?.close(); } catch { /* ignore */ }
    audioContextRef.current = null;

    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;

    if (blobURLRef.current) {
      URL.revokeObjectURL(blobURLRef.current);
      blobURLRef.current = null;
    }

    // openwakeword-js doesn't expose a destroy() — GC reclaims the
    // ONNX sessions when the Model ref drops.
    modelRef.current = null;

    setState({
      listening: false,
      initializing: false,
      error: null,
      lastDetectionTs: 0,
    });
  }, []);


  // Ensure we release the mic + ORT sessions on unmount.
  useEffect(() => {
    return () => { stop(); };
  }, [stop]);


  return { ...state, start, stop };
}
