/**
 * PCM Streaming AudioWorklet Processor.
 *
 * Keeps an internal Float32 ring buffer. The main thread pushes PCM samples
 * via port messages. The process() method drains the ring buffer at
 * sample-accurate timing on the audio thread.
 *
 * When pitchRate != 1.0, the worklet does linear-interpolated resampling
 * as it drains — reading the ring at `pitchRate` samples per output sample
 * effectively pitches the audio down (rate < 1) or up (rate > 1). At
 * 0.89 (≈ -2 semitones) a voice becomes noticeably deeper. Speech tempo
 * changes too, which suits Ultron's deliberate cadence at slow rates.
 *
 * This is the textbook correct way to stream PCM audio — no per-chunk
 * AudioBufferSourceNodes, no scheduling drift, no sample-rate resampling
 * mismatches. The worklet runs on the audio thread with sample-accurate timing.
 *
 * Sources:
 *   https://developer.chrome.com/blog/audio-worklet-design-pattern
 *   https://loke.dev/blog/stop-allocating-inside-audioworkletprocessor
 */

class PCMStreamProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    // Ring buffer sized to hold a full long response.
    // 1,048,576 samples @ 24kHz = ~43 seconds of audio.
    this.bufferSize = 1048576;  // 2^20
    this.mask = this.bufferSize - 1;
    this.buffer = new Float32Array(this.bufferSize);

    this.writeIndex = 0;
    // readCursor is a *float* because at pitchRate != 1.0 we advance
    // by a fractional amount per output sample and linearly interpolate
    // between the two surrounding integer indices.
    this.readCursor = 0.0;

    // 1.0 = native rate. <1.0 pitches DOWN (deeper, slower).
    //                    >1.0 pitches UP (higher, faster).
    // Main thread sends `{type:'pitch', rate:0.89}` messages to change it.
    this.pitchRate = 1.0;
    this.targetPitchRate = 1.0;
    // Rate smoothing — interpolate toward target over ~60ms so setting
    // pitch mid-playback doesn't produce an audible chirp.
    this.pitchSmoothSamplesRemaining = 0;

    this.droppedSamples = 0;
    this.lastDropReport = 0;

    this.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'push' && msg.samples) {
        this._push(msg.samples);
      } else if (msg.type === 'clear') {
        this._clear();
      } else if (msg.type === 'pitch' && typeof msg.rate === 'number') {
        this._setPitchRate(msg.rate);
      }
    };
  }

  _push(samples) {
    const available = this.writeIndex - this.readCursor;
    const free = this.bufferSize - available;
    const toWrite = Math.min(samples.length, Math.floor(free));

    for (let i = 0; i < toWrite; i++) {
      this.buffer[(this.writeIndex + i) & this.mask] = samples[i];
    }
    this.writeIndex += toWrite;

    if (toWrite < samples.length) {
      this.droppedSamples += samples.length - toWrite;
      if (this.droppedSamples - this.lastDropReport > 24000) {
        this.port.postMessage({
          type: 'overflow',
          dropped: this.droppedSamples,
        });
        this.lastDropReport = this.droppedSamples;
      }
    }
  }

  _clear() {
    this.readCursor = this.writeIndex;
  }

  _setPitchRate(rate) {
    // Clamp to a sane range; extreme rates can deadlock the reader.
    const clamped = Math.max(0.5, Math.min(2.0, rate));
    this.targetPitchRate = clamped;
    // Smooth over ~60ms worth of samples at 24kHz = ~1440 samples.
    // That's enough to avoid an audible step, short enough to feel
    // responsive.
    this.pitchSmoothSamplesRemaining = Math.round(sampleRate * 0.06);
  }

  _readSampleInterpolated(cursor) {
    // Linear interpolation between floor(cursor) and floor(cursor)+1.
    // Sufficient quality for speech; a higher-order filter would reduce
    // very-high-frequency artefacts but they aren't audible for speech.
    const i0 = Math.floor(cursor);
    const i1 = i0 + 1;
    const frac = cursor - i0;
    const s0 = this.buffer[i0 & this.mask];
    const s1 = this.buffer[i1 & this.mask];
    return s0 + (s1 - s0) * frac;
  }

  process(inputs, outputs) {
    const output = outputs[0];
    const channel = output[0];
    const quantumSize = channel.length;  // always 128 in current spec

    // How many source samples we need to read to produce `quantumSize`
    // output samples, given the current pitchRate.
    const neededSource = quantumSize * this.pitchRate;
    const available = this.writeIndex - this.readCursor;

    if (available >= neededSource + 1) {
      // Full quantum producible.
      for (let i = 0; i < quantumSize; i++) {
        channel[i] = this._readSampleInterpolated(this.readCursor);
        // Advance cursor, smoothing toward target rate.
        let rate = this.pitchRate;
        if (this.pitchSmoothSamplesRemaining > 0) {
          const t = 1 - (this.pitchSmoothSamplesRemaining / (sampleRate * 0.06));
          rate = this.pitchRate + (this.targetPitchRate - this.pitchRate) * t;
          this.pitchSmoothSamplesRemaining--;
          if (this.pitchSmoothSamplesRemaining === 0) {
            this.pitchRate = this.targetPitchRate;
          }
        }
        this.readCursor += rate;
      }
    } else if (available > 0) {
      // Partial buffer — play what we have at native rate, zero the rest.
      // Falling back to native rate here avoids weird stretching when
      // audio is just ending. Also applies a brief fade to avoid a click
      // into silence at the tail.
      const producible = Math.min(quantumSize, Math.floor(available));
      for (let i = 0; i < producible; i++) {
        channel[i] = this.buffer[((this.readCursor | 0) + i) & this.mask];
      }
      const fadeLen = Math.min(32, producible);
      const fadeStart = producible - fadeLen;
      for (let i = 0; i < fadeLen; i++) {
        const g = 1 - (i / fadeLen);
        channel[fadeStart + i] *= g;
      }
      for (let i = producible; i < quantumSize; i++) {
        channel[i] = 0;
      }
      this.readCursor += producible;
    } else {
      // Silence.
      channel.fill(0);
    }

    return true;
  }
}

registerProcessor('pcm-stream-processor', PCMStreamProcessor);
