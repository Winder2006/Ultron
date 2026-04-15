/**
 * Voice filter chain — robot character, stable by construction.
 *
 * Chain (in order):
 *   input
 *     → highpass (sub-rumble removal)
 *     → lowShelf (cut OR boost — negative = cut, positive = add body)
 *     → waveshaper (digital saturation)
 *     → presenceEQ (peaking boost around 2kHz for "through-a-speaker" bite)
 *     → bodyEQ (peaking boost around 200Hz for chest/menace)
 *     → chorusDryGain + chorusWet (delay + slight detune) — very short delay
 *       produces a metallic doubled-voice feel without being a comb filter
 *     → outputGain
 *     → limiter
 *     → destination
 *
 * Intensity-responsive: a caller can call setIntensity(0..1) per sentence
 * to scale the distortion/presence/body/chorus wetness around the base
 * params, which is how the heuristic intensity scaler in useVoice modulates
 * the voice per-chunk based on the text being spoken.
 *
 * No feedback paths — chorus uses a single non-recirculating delay so the
 * system stays stable regardless of parameter settings.
 */

export interface FilterParams {
  enabled: boolean;
  /** Pitch shift in semitones (applied via source playbackRate) */
  pitchSemitones: number;
  /** Distortion amount 0..1 */
  distortion: number;
  /** Presence boost at ~2kHz in dB (0..12) */
  presence: number;
  /** Low-shelf gain in dB at 250Hz. Negative cuts (thins the voice),
   *  positive adds body (deeper / more menacing). Range: -18..+12. */
  lowCut: number;
  /** Body/chest EQ boost around 200Hz in dB (0..8). Adds weight without
   *  muddying. Stacks with lowCut for heavy voices. */
  bodyBoost: number;
  /** Metallic-doubled-voice amount 0..1. Mixes in a 12-18ms detuned
   *  delayed copy. Low values: subtle shimmer. High values: distinct
   *  "synth chord" feel. Not a comb filter — single non-recirculating
   *  delay. */
  chorus: number;
  /** How much the per-sentence intensity signal scales distortion,
   *  presence, body, and chorus. 0 = no dynamic response (pure static
   *  preset); 1 = full scaling. Range: 0..1. */
  intensityResponsiveness: number;
  /** Output gain compensation */
  outputGain: number;

  // ── New effects ──

  /** Ring modulator wet mix, 0..1. Multiplies the signal by a sine
   *  carrier — gives Dalek/robot metallic ring. */
  ringModAmount: number;
  /** Ring modulator carrier frequency in Hz (30..800). Low = menacing
   *  tremolo, mid = metallic, high = alien. */
  ringModFreq: number;

  /** Bitcrusher wet mix, 0..1 (dry/wet blend). */
  bitcrushAmount: number;
  /** Bit depth target for the crusher (4..16 bits). Lower = more grit. */
  bitcrushBits: number;

  /** Reverb wet mix 0..1. Adds a synthetic plate/room tail — "speaking
   *  from inside a steel shell." */
  reverbAmount: number;
  /** Reverb decay length 0.1..4.0 seconds. Longer = bigger space. */
  reverbDecay: number;

  /** LFO wobble depth 0..1. Modulates the presence-EQ frequency to
   *  add unsettling drift/warble. */
  wobbleDepth: number;
  /** LFO wobble rate 0.1..8 Hz. Slow = creepy sway, fast = tremolo. */
  wobbleRate: number;
}

export type PresetName =
  | "Clean"
  | "Subtle"
  | "Ultron"
  | "Heavy Robot"
  | "Menacing";

// Zero values for all new-effect fields — spread into every preset
// so old preset tuning stays visually identical, and the user can
// dial the new effects in live from 0 without needing a new preset.
const NO_EFFECTS = {
  ringModAmount: 0,
  ringModFreq: 120,
  bitcrushAmount: 0,
  bitcrushBits: 16,
  reverbAmount: 0,
  reverbDecay: 1.2,
  wobbleDepth: 0,
  wobbleRate: 1.5,
} as const;

export const PRESETS: Record<PresetName, FilterParams> = {
  Clean: {
    enabled: false,
    pitchSemitones: 0,
    distortion: 0,
    presence: 0,
    lowCut: 0,
    bodyBoost: 0,
    chorus: 0,
    intensityResponsiveness: 0,
    outputGain: 1.0,
    ...NO_EFFECTS,
  },
  Subtle: {
    enabled: true,
    pitchSemitones: 0,
    distortion: 0.05,
    presence: 3,
    lowCut: -6,
    bodyBoost: 0,
    chorus: 0,
    intensityResponsiveness: 0.3,
    outputGain: 1.0,
    ...NO_EFFECTS,
  },
  // Tuned by ear. Slight pitch detune, heavy saturation, a deliberate
  // low-shelf cut to thin the voice (the "speaking through steel"
  // vibe), strong presence boost for through-a-speaker bite, moderate
  // chorus doubling, and high output compensation to keep perceived
  // loudness up after the cut.
  Ultron: {
    enabled: true,
    pitchSemitones: -0.25,
    distortion: 0.54,
    presence: 9.0,
    lowCut: -14.5,
    bodyBoost: 2.0,
    chorus: 0.54,
    intensityResponsiveness: 0.75,
    outputGain: 1.95,
    ...NO_EFFECTS,
  },
  "Heavy Robot": {
    enabled: true,
    pitchSemitones: -1,
    distortion: 0.25,
    presence: 9,
    lowCut: -12,
    bodyBoost: 0,
    chorus: 0.20,
    intensityResponsiveness: 0.4,
    outputGain: 1.2,
    ...NO_EFFECTS,
  },
  // Menacing: properly deep. Noticeably lower pitch (now actually
  // applied on the streaming path via the worklet resampler), stronger
  // metallic chorus doubling, pronounced body at 200Hz, a warmer
  // low-shelf boost for weight rather than the thin "cut" feel of
  // the older Ultron preset. Distortion is cranked to give the voice
  // real digital edge when intensity rises. Intensity responsiveness
  // near maximum so signature lines hit hard.
  Menacing: {
    enabled: true,
    pitchSemitones: -3.5,   // real depth — worklet resampler applies it
    distortion: 0.28,
    presence: 7.5,
    lowCut: +6,             // positive = added body weight
    bodyBoost: 7,
    chorus: 0.45,
    intensityResponsiveness: 0.85,
    outputGain: 1.15,
    ...NO_EFFECTS,
  },
};

export class VoiceFilterChain {
  private ctx: AudioContext;
  private params: FilterParams;
  // Per-sentence intensity, 0..1. 0.5 is the "normal" baseline; higher
  // scales distortion/presence/body/chorus up, lower scales them down.
  // Set via setIntensity() from useVoice on every `tts_start` event.
  private intensity: number = 0.5;

  private input: GainNode;
  private highpass: BiquadFilterNode;
  private lowShelf: BiquadFilterNode;
  private waveshaper: WaveShaperNode;
  private presenceEQ: BiquadFilterNode;
  private bodyEQ: BiquadFilterNode;     // Peaking boost ~200Hz for chest

  // Ring mod — a sine carrier multiplied with the signal via a GainNode
  // whose .gain audio-rate modulation tracks the carrier. Dry/wet mix
  // sums the unmodulated signal back in so the user can blend.
  private ringModDryGain: GainNode;
  private ringModWetGain: GainNode;
  private ringModMultiplier: GainNode;
  private ringModCarrier: OscillatorNode | null;

  // Bitcrusher — implemented as a WaveShaperNode with a quantisation
  // curve. Stays in pure Web Audio (no extra worklet). Dry/wet mix
  // lets the user dial in grit without hardclipping the signal.
  private crushDryGain: GainNode;
  private crushWetGain: GainNode;
  private crushShaper: WaveShaperNode;

  // Reverb — ConvolverNode with a synthetic exponentially-decaying
  // noise IR (generated on-demand from reverbDecay). Classic dry/wet
  // split so a tiny bit sounds like a room and a lot sounds like a
  // cathedral.
  private reverbDryGain: GainNode;
  private reverbWetGain: GainNode;
  private reverbConvolver: ConvolverNode;
  private lastReverbDecay: number = -1;

  // LFO wobble — modulates presenceEQ.frequency so the 2kHz bite
  // sweeps up and down. Creates an unsettling drift. Second oscillator,
  // separate from the chorus LFO.
  private wobbleLFO: OscillatorNode | null;
  private wobbleLFOGain: GainNode | null;

  private chorusDryGain: GainNode;      // Dry branch of chorus mixer
  private chorusWetGain: GainNode;      // Wet branch of chorus mixer
  private chorusDelay: DelayNode;       // Very short modulated delay
  private chorusLFO: OscillatorNode | null;
  private chorusLFOGain: GainNode | null;
  // Intonation compressor: flattens vocal peaks so Deepgram's natural
  // intonation spikes (exclamations, questions) feel less extreme. Does
  // the heavy lifting of "keep the voice sounding controlled / robotic."
  // Distinct from `limiter` which is only for brick-wall peak catching.
  private intonationCompressor: DynamicsCompressorNode;
  private outputGain: GainNode;
  private limiter: DynamicsCompressorNode;

  constructor(ctx: AudioContext, params: FilterParams = PRESETS.Clean) {
    this.ctx = ctx;
    this.params = { ...params };

    this.input = ctx.createGain();

    this.highpass = ctx.createBiquadFilter();
    this.highpass.type = "highpass";
    this.highpass.frequency.value = 80;
    this.highpass.Q.value = 0.7;

    this.lowShelf = ctx.createBiquadFilter();
    this.lowShelf.type = "lowshelf";
    this.lowShelf.frequency.value = 250;
    this.lowShelf.gain.value = 0;

    this.waveshaper = ctx.createWaveShaper();
    this.waveshaper.oversample = "2x";

    this.presenceEQ = ctx.createBiquadFilter();
    this.presenceEQ.type = "peaking";
    this.presenceEQ.frequency.value = 2000;
    this.presenceEQ.Q.value = 1.0;
    this.presenceEQ.gain.value = 0;

    this.bodyEQ = ctx.createBiquadFilter();
    this.bodyEQ.type = "peaking";
    this.bodyEQ.frequency.value = 200;
    this.bodyEQ.Q.value = 1.1;
    this.bodyEQ.gain.value = 0;

    // Ring mod chain — a GainNode whose .gain is audio-rate-modulated
    // by a sine oscillator. Multiplying by a zero-centred sine gives
    // classic Dalek AM. Dry/wet split so we don't lose the voice
    // itself when ringModAmount is small.
    this.ringModDryGain = ctx.createGain();
    this.ringModWetGain = ctx.createGain();
    this.ringModMultiplier = ctx.createGain();
    this.ringModMultiplier.gain.value = 0;  // driven by carrier
    this.ringModCarrier = null;              // lazy-started
    this.ringModDryGain.gain.value = 1;
    this.ringModWetGain.gain.value = 0;

    // Bitcrusher — a WaveShaper with a stepped quantisation curve.
    // Default to identity; rebuilt in applyAll() when bits change.
    this.crushDryGain = ctx.createGain();
    this.crushWetGain = ctx.createGain();
    this.crushShaper = ctx.createWaveShaper();
    this.crushShaper.oversample = "none"; // preserves the aliasing that makes it sound crunchy
    this.crushShaper.curve = this.identityCurve();
    this.crushDryGain.gain.value = 1;
    this.crushWetGain.gain.value = 0;

    // Reverb — synthetic exponentially-decaying noise IR. Rebuilt on
    // decay change. Dry branch passes the signal unchanged so
    // reverbAmount=0 is truly clean, not just quiet.
    this.reverbConvolver = ctx.createConvolver();
    this.reverbConvolver.buffer = this.makeReverbIR(1.2);
    this.lastReverbDecay = 1.2;
    this.reverbDryGain = ctx.createGain();
    this.reverbWetGain = ctx.createGain();
    this.reverbDryGain.gain.value = 1;
    this.reverbWetGain.gain.value = 0;

    // Wobble LFO — targets presenceEQ.frequency. Created lazily.
    this.wobbleLFO = null;
    this.wobbleLFOGain = null;

    // Chorus: a single DelayNode (no feedback) with a slow LFO modulating
    // the delay time. The modulated copy mixed back in produces a
    // detuned-doubled-voice effect — the metallic "Ultron-speaking-from-
    // inside-the-shell" character. Not a comb filter: no recirculation,
    // guaranteed stable.
    this.chorusDelay = ctx.createDelay(0.05);
    this.chorusDelay.delayTime.value = 0.015; // 15ms baseline

    this.chorusDryGain = ctx.createGain();
    this.chorusWetGain = ctx.createGain();
    this.chorusDryGain.gain.value = 1;
    this.chorusWetGain.gain.value = 0;

    // LFO drives delayTime modulation. Lazy-started on first use
    // because some browsers crash if you start too many oscillators
    // at AudioContext creation time.
    this.chorusLFO = null;
    this.chorusLFOGain = null;

    // Intonation compressor: medium ratio, musical attack/release.
    // Kicks in around -18 dBFS so only the *peaks* (the "raising his
    // voice" moments) get squashed, leaving the calm body uncompressed.
    // Result: Ultron's dynamic range narrows without sounding pumped
    // or pinched.
    this.intonationCompressor = ctx.createDynamicsCompressor();
    this.intonationCompressor.threshold.value = -18;
    this.intonationCompressor.ratio.value = 4;
    this.intonationCompressor.attack.value = 0.010;
    this.intonationCompressor.release.value = 0.15;
    this.intonationCompressor.knee.value = 6;

    this.outputGain = ctx.createGain();

    // Final brick-wall limiter. Different job from the compressor
    // above: this catches peaks that slip past the compressor and
    // saturation without audible release.
    this.limiter = ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -1;
    this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.001;
    this.limiter.release.value = 0.05;
    this.limiter.knee.value = 0;

    // Wire the graph.
    // Main path:
    //   input → highpass → lowShelf → waveshaper → presenceEQ → bodyEQ
    //     → ringMod (dry|wet×carrier) sum
    //     → bitcrush (dry|wet through quantiser) sum
    //     → reverb (dry|wet through convolver) sum
    //     → [split: chorus dry | chorus delay wet] sum
    //     → outputGain → intonationCompressor → limiter → destination
    //
    // Each effect has its own dry/wet split so at 0 the signal passes
    // through unchanged — no tone colour creep from just having the
    // nodes in the graph.
    this.input.connect(this.highpass);
    this.highpass.connect(this.lowShelf);
    this.lowShelf.connect(this.waveshaper);
    this.waveshaper.connect(this.presenceEQ);
    this.presenceEQ.connect(this.bodyEQ);

    // Ring mod stage — a summing GainNode collects the dry + wet legs.
    const ringModSum = ctx.createGain();
    this.bodyEQ.connect(this.ringModDryGain);
    this.bodyEQ.connect(this.ringModMultiplier);
    this.ringModMultiplier.connect(this.ringModWetGain);
    this.ringModDryGain.connect(ringModSum);
    this.ringModWetGain.connect(ringModSum);

    // Bitcrush stage
    const crushSum = ctx.createGain();
    ringModSum.connect(this.crushDryGain);
    ringModSum.connect(this.crushShaper);
    this.crushShaper.connect(this.crushWetGain);
    this.crushDryGain.connect(crushSum);
    this.crushWetGain.connect(crushSum);

    // Reverb stage
    const reverbSum = ctx.createGain();
    crushSum.connect(this.reverbDryGain);
    crushSum.connect(this.reverbConvolver);
    this.reverbConvolver.connect(this.reverbWetGain);
    this.reverbDryGain.connect(reverbSum);
    this.reverbWetGain.connect(reverbSum);

    // Chorus stage — dry + wet branches sum into outputGain.
    reverbSum.connect(this.chorusDryGain);
    reverbSum.connect(this.chorusDelay);
    this.chorusDelay.connect(this.chorusWetGain);

    this.chorusDryGain.connect(this.outputGain);
    this.chorusWetGain.connect(this.outputGain);

    // outputGain → intonationCompressor → limiter → destination.
    // Compressor sits here so it sees everything post-EQ/saturation
    // but before the final brick-wall. That order means we tame the
    // *post-effect* peaks that matter for perceived loudness.
    this.outputGain.connect(this.intonationCompressor);
    this.intonationCompressor.connect(this.limiter);
    this.limiter.connect(ctx.destination);

    this.applyAll();
  }

  /** Start the chorus LFO the first time chorus is needed. */
  private ensureChorusLFO() {
    if (this.chorusLFO) return;
    try {
      this.chorusLFO = this.ctx.createOscillator();
      this.chorusLFO.frequency.value = 0.8; // slow sweep, Hz
      this.chorusLFOGain = this.ctx.createGain();
      this.chorusLFOGain.gain.value = 0.003; // ±3ms around the 15ms baseline
      this.chorusLFO.connect(this.chorusLFOGain);
      this.chorusLFOGain.connect(this.chorusDelay.delayTime);
      this.chorusLFO.start();
    } catch {
      // If the oscillator can't start (very rare), chorus just runs at
      // fixed delay — still produces a doubled-voice effect, just no
      // shimmer.
    }
  }

  /** Start the ring-mod carrier oscillator. Drives the multiplier
   *  GainNode's .gain as an audio-rate signal, which is the standard
   *  Web Audio way to implement ring modulation. */
  private ensureRingMod() {
    if (this.ringModCarrier) return;
    try {
      this.ringModCarrier = this.ctx.createOscillator();
      this.ringModCarrier.type = "sine";
      this.ringModCarrier.frequency.value = this.params.ringModFreq;
      // Connect carrier → multiplier.gain. The carrier swings -1..+1;
      // the multiplier's GainNode multiplies its audio input by that
      // signal, producing AM / ring mod.
      this.ringModCarrier.connect(this.ringModMultiplier.gain);
      this.ringModCarrier.start();
    } catch {
      // Fallback: multiplier stays at 0 → wet branch is silent. User
      // just gets dry signal — acceptable failure mode.
    }
  }

  /** Start the wobble LFO targeting presenceEQ.frequency so the 2kHz
   *  "bite" drifts up and down. */
  private ensureWobbleLFO() {
    if (this.wobbleLFO) return;
    try {
      this.wobbleLFO = this.ctx.createOscillator();
      this.wobbleLFO.type = "sine";
      this.wobbleLFO.frequency.value = this.params.wobbleRate;
      this.wobbleLFOGain = this.ctx.createGain();
      this.wobbleLFOGain.gain.value = 0; // depth applied in applyAll
      this.wobbleLFO.connect(this.wobbleLFOGain);
      this.wobbleLFOGain.connect(this.presenceEQ.frequency);
      this.wobbleLFO.start();
    } catch {
      // No wobble — fine, static bite.
    }
  }

  /** Build a synthetic reverb IR: white noise × exponential decay.
   *  Not a real plate/spring sim, but close enough to give the
   *  "speaking from inside a steel shell" character without shipping a
   *  proper impulse response file. */
  private makeReverbIR(decaySeconds: number): AudioBuffer {
    const sr = this.ctx.sampleRate;
    const length = Math.max(1, Math.floor(sr * Math.max(0.05, decaySeconds)));
    const buf = this.ctx.createBuffer(2, length, sr);
    for (let ch = 0; ch < 2; ch++) {
      const data = buf.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        // (-1..+1) noise × exponential decay curve.
        // Slight early-reflections bias: first 30ms a touch louder.
        const t = i / length;
        const env = Math.pow(1 - t, 2.5);
        const early = i < sr * 0.03 ? 1.2 : 1.0;
        data[i] = (Math.random() * 2 - 1) * env * early;
      }
    }
    return buf;
  }

  /** Bitcrusher curve: step-quantise the -1..+1 input to 2^bits levels.
   *  Fewer bits → bigger quantisation steps → more digital grit. */
  private bitcrushCurve(bits: number): Float32Array {
    const n = 4096;
    const curve = new Float32Array(n);
    const levels = Math.pow(2, Math.max(1, Math.min(16, Math.round(bits))));
    const step = 2 / levels;
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      // Quantise to nearest step; centre the rounding so 0 stays 0.
      curve[i] = Math.round(x / step) * step;
    }
    return curve;
  }

  getInputNode(): AudioNode {
    return this.input;
  }

  getPitchRate(): number {
    if (!this.params.enabled) return 1;
    return Math.pow(2, this.params.pitchSemitones / 12);
  }

  setParams(params: Partial<FilterParams>) {
    this.params = { ...this.params, ...params };
    this.applyAll();
  }

  getParams(): FilterParams {
    return { ...this.params };
  }

  /**
   * Set the per-sentence intensity (0..1). Called from useVoice on each
   * incoming `tts_start` event after the heuristic scorer analyzes the
   * sentence text. Scales distortion/presence/body/chorus around their
   * baseline by up to `intensityResponsiveness`. Linear smoothing over
   * 80ms avoids audible parameter-jump clicks.
   */
  setIntensity(intensity: number) {
    this.intensity = Math.max(0, Math.min(1, intensity));
    this.applyAll();
  }

  private applyAll() {
    const p = this.params;
    const on = p.enabled;
    const now = this.ctx.currentTime;
    // Smoothing window — parameters cross-fade linearly over this interval.
    // Keeps intensity changes audibly musical instead of steppy.
    const smooth = 0.08;

    // Map intensity 0..1 to a scaling factor around the baseline. At
    // intensity=0.5 we use the preset verbatim. At intensity=1 we push
    // distortion/presence/body/chorus up by `intensityResponsiveness`.
    // At intensity=0 we pull them down by the same amount.
    const r = p.intensityResponsiveness;
    const kick = (this.intensity - 0.5) * 2 * r; // -r..+r

    const distortion = Math.max(0, Math.min(1, p.distortion + kick * 0.35));
    const presence = Math.max(-12, Math.min(14, p.presence + kick * 5));
    const body = Math.max(0, Math.min(12, p.bodyBoost + kick * 4));
    const chorus = Math.max(0, Math.min(1, p.chorus + kick * 0.25));

    // AudioParam automation for smooth transitions.
    if (on) {
      this.lowShelf.gain.cancelScheduledValues(now);
      this.lowShelf.gain.linearRampToValueAtTime(p.lowCut, now + smooth);
      this.presenceEQ.gain.cancelScheduledValues(now);
      this.presenceEQ.gain.linearRampToValueAtTime(presence, now + smooth);
      this.bodyEQ.gain.cancelScheduledValues(now);
      this.bodyEQ.gain.linearRampToValueAtTime(body, now + smooth);
      this.chorusDryGain.gain.cancelScheduledValues(now);
      this.chorusDryGain.gain.linearRampToValueAtTime(1 - chorus * 0.5, now + smooth);
      this.chorusWetGain.gain.cancelScheduledValues(now);
      this.chorusWetGain.gain.linearRampToValueAtTime(chorus, now + smooth);
      this.outputGain.gain.cancelScheduledValues(now);
      this.outputGain.gain.linearRampToValueAtTime(p.outputGain, now + smooth);

      if (chorus > 0.001) this.ensureChorusLFO();

      this.waveshaper.curve = distortion > 0.001
        ? this.distortionCurve(distortion)
        : this.identityCurve();

      // ── New effects ──

      // Ring mod: mix. Only start the carrier once we actually want it.
      this.ringModDryGain.gain.cancelScheduledValues(now);
      this.ringModDryGain.gain.linearRampToValueAtTime(
        1 - p.ringModAmount * 0.5, now + smooth,
      );
      this.ringModWetGain.gain.cancelScheduledValues(now);
      this.ringModWetGain.gain.linearRampToValueAtTime(
        p.ringModAmount, now + smooth,
      );
      if (p.ringModAmount > 0.001) {
        this.ensureRingMod();
        if (this.ringModCarrier) {
          this.ringModCarrier.frequency.cancelScheduledValues(now);
          this.ringModCarrier.frequency.linearRampToValueAtTime(
            Math.max(10, p.ringModFreq), now + smooth,
          );
        }
      }

      // Bitcrusher: rebuild the curve when bits change. Curve is small
      // (4096 floats) so rebuilding on every param write is fine.
      this.crushDryGain.gain.cancelScheduledValues(now);
      this.crushDryGain.gain.linearRampToValueAtTime(
        1 - p.bitcrushAmount, now + smooth,
      );
      this.crushWetGain.gain.cancelScheduledValues(now);
      this.crushWetGain.gain.linearRampToValueAtTime(
        p.bitcrushAmount, now + smooth,
      );
      if (p.bitcrushAmount > 0.001) {
        this.crushShaper.curve = this.bitcrushCurve(p.bitcrushBits);
      }

      // Reverb: regenerate IR only when decay actually changed — the
      // IR buffer is ~200k samples and costs a small spike to build.
      if (Math.abs(p.reverbDecay - this.lastReverbDecay) > 0.05) {
        this.reverbConvolver.buffer = this.makeReverbIR(p.reverbDecay);
        this.lastReverbDecay = p.reverbDecay;
      }
      this.reverbDryGain.gain.cancelScheduledValues(now);
      this.reverbDryGain.gain.linearRampToValueAtTime(1, now + smooth);
      this.reverbWetGain.gain.cancelScheduledValues(now);
      this.reverbWetGain.gain.linearRampToValueAtTime(
        p.reverbAmount, now + smooth,
      );

      // Wobble: modulates presence EQ frequency. Depth in Hz — we map
      // 0..1 wobbleDepth to 0..900Hz of swing around the 2kHz centre
      // so extreme settings are audibly dramatic without pushing the
      // filter below 100Hz (where it'd sound broken).
      if (p.wobbleDepth > 0.001) {
        this.ensureWobbleLFO();
        if (this.wobbleLFO) {
          this.wobbleLFO.frequency.cancelScheduledValues(now);
          this.wobbleLFO.frequency.linearRampToValueAtTime(
            Math.max(0.05, p.wobbleRate), now + smooth,
          );
        }
        if (this.wobbleLFOGain) {
          this.wobbleLFOGain.gain.cancelScheduledValues(now);
          this.wobbleLFOGain.gain.linearRampToValueAtTime(
            p.wobbleDepth * 900, now + smooth,
          );
        }
      } else if (this.wobbleLFOGain) {
        this.wobbleLFOGain.gain.cancelScheduledValues(now);
        this.wobbleLFOGain.gain.linearRampToValueAtTime(0, now + smooth);
      }
    } else {
      // Disabled: bypass with dry=1, wet=0, flat EQs, identity shaper.
      this.lowShelf.gain.linearRampToValueAtTime(0, now + smooth);
      this.presenceEQ.gain.linearRampToValueAtTime(0, now + smooth);
      this.bodyEQ.gain.linearRampToValueAtTime(0, now + smooth);
      this.chorusDryGain.gain.linearRampToValueAtTime(1, now + smooth);
      this.chorusWetGain.gain.linearRampToValueAtTime(0, now + smooth);
      this.outputGain.gain.linearRampToValueAtTime(1, now + smooth);
      this.waveshaper.curve = this.identityCurve();

      // New effects off: dry=1, wet=0.
      this.ringModDryGain.gain.linearRampToValueAtTime(1, now + smooth);
      this.ringModWetGain.gain.linearRampToValueAtTime(0, now + smooth);
      this.crushDryGain.gain.linearRampToValueAtTime(1, now + smooth);
      this.crushWetGain.gain.linearRampToValueAtTime(0, now + smooth);
      this.reverbDryGain.gain.linearRampToValueAtTime(1, now + smooth);
      this.reverbWetGain.gain.linearRampToValueAtTime(0, now + smooth);
      if (this.wobbleLFOGain) {
        this.wobbleLFOGain.gain.linearRampToValueAtTime(0, now + smooth);
      }
    }
  }

  private identityCurve(): Float32Array {
    const n = 512;
    const curve = new Float32Array(n);
    for (let i = 0; i < n; i++) curve[i] = (i * 2) / n - 1;
    return curve;
  }

  private distortionCurve(amount: number): Float32Array {
    const n = 1024;
    const curve = new Float32Array(n);
    const drive = 1 + amount * 6;
    const norm = Math.tanh(drive);
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1;
      curve[i] = Math.tanh(x * drive) / norm;
    }
    return curve;
  }

  destroy() {
    try { this.chorusLFO?.stop(); } catch { /* ignore */ }
    try { this.ringModCarrier?.stop(); } catch { /* ignore */ }
    try { this.wobbleLFO?.stop(); } catch { /* ignore */ }
    try { this.input.disconnect(); } catch { /* ignore */ }
    try { this.limiter.disconnect(); } catch { /* ignore */ }
  }
}
