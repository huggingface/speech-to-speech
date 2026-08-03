// @ts-check
/**
 * AudioWorkletProcessor that resamples the AudioContext rate (typically 48 kHz)
 * down to 16 kHz, packs the result as little-endian Int16 PCM, and posts it
 * back to the main thread in fixed-size chunks.
 *
 * The Hugging Face speech-to-speech WebSocket route expects the
 * `input_audio_buffer.append` payload at 16 kHz PCM16 mono.
 *
 * Design notes:
 *   - 48 -> 16 is an exact 3:1 ratio so we use a 3-tap boxcar average as a
 *     cheap low-pass before decimating. Good enough for voice STT; we lose
 *     a tiny bit of >8 kHz content which the pipeline discards anyway.
 *   - Output frames are emitted at the cadence dictated by `chunkMs`
 *     (default 40 ms = 640 samples = 1280 bytes). The OpenAI Realtime
 *     server batches incoming audio so the cadence is flexible; 20-100 ms
 *     is the sweet spot.
 *   - Float -> Int16 saturates to [-1, 1] before scaling.
 *   - Optional noise gate: per-chunk RMS decides open/closed against a
 *     threshold; the gain ramps (fast attack, hold, slow release) so word
 *     onsets aren't clipped and quiet tails don't click. The gate only
 *     affects the audio we SEND; the main-thread visualiser taps the raw
 *     mic separately. We post the chunk RMS up every frame so the Settings
 *     mic meter can show the live level against the threshold.
 */

const TARGET_RATE = 16000;
const DEFAULT_CHUNK_MS = 40;
// Gate envelope timing (fixed; only the threshold is user-tunable).
const GATE_ATTACK_MS = 5; // open almost instantly so word onsets survive
const GATE_HOLD_MS = 250; // stay open this long after the level drops back under
const GATE_RELEASE_MS = 80; // then fade closed over this long (no click)

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkMs = options?.processorOptions?.chunkMs ?? DEFAULT_CHUNK_MS;
    this._inputRate = sampleRate;
    this._ratio = this._inputRate / TARGET_RATE;
    this._chunkSamples16k = Math.round((TARGET_RATE * chunkMs) / 1000);
    this._scratch = new Float32Array(0);
    this._decimated = new Float32Array(this._chunkSamples16k);
    this._enabled = true;

    // Noise gate state. Disabled by default (pure passthrough).
    this._gateEnabled = false;
    this._thresholdLin = 0; // linear amplitude; signal RMS must exceed this to open
    this._gateGain = 1; // smoothed gain currently applied
    this._holdRemaining = 0; // samples left before the gate may start closing
    this._attackCoef = Math.exp(-1 / ((GATE_ATTACK_MS / 1000) * TARGET_RATE));
    this._releaseCoef = Math.exp(-1 / ((GATE_RELEASE_MS / 1000) * TARGET_RATE));
    this._holdSamples = Math.round((GATE_HOLD_MS / 1000) * TARGET_RATE);

    this.port.onmessage = (e) => {
      const data = e.data;
      if (data?.kind === "enable") this._enabled = !!data.value;
      else if (data?.kind === "gate") {
        this._gateEnabled = !!data.enabled;
        // dB -> linear amplitude. When off, threshold 0 keeps the gate open.
        this._thresholdLin = data.enabled ? Math.pow(10, data.thresholdDb / 20) : 0;
      }
    };
  }

  /**
   * Append `incoming` to the internal scratch buffer, then emit as many
   * full output chunks as we have material for.
   * @param {Float32Array} incoming
   */
  _ingest(incoming) {
    if (incoming.length === 0) return;
    const next = new Float32Array(this._scratch.length + incoming.length);
    next.set(this._scratch, 0);
    next.set(incoming, this._scratch.length);
    this._scratch = next;
    this._maybeEmit();
  }

  _maybeEmit() {
    const r = this._ratio;
    const n = this._chunkSamples16k;
    const needIn = Math.ceil(n * r);
    const dec = this._decimated;
    while (this._scratch.length >= needIn) {
      // 1. Decimate to 16 kHz floats and accumulate energy for the gate/meter.
      let sumSq = 0;
      if (Math.abs(r - 3) < 1e-6) {
        // 48 kHz -> 16 kHz fast path with boxcar lowpass.
        for (let i = 0; i < n; i++) {
          const idx = i * 3;
          const s = (this._scratch[idx] + this._scratch[idx + 1] + this._scratch[idx + 2]) / 3;
          dec[i] = s;
          sumSq += s * s;
        }
      } else {
        // Generic path: linear interpolation. Slower but works at any rate
        // (e.g. some Windows boxes report sampleRate=44100).
        for (let i = 0; i < n; i++) {
          const srcPos = i * r;
          const idx = Math.floor(srcPos);
          const frac = srcPos - idx;
          const a = this._scratch[idx];
          const b = this._scratch[idx + 1] ?? a;
          const s = a + (b - a) * frac;
          dec[i] = s;
          sumSq += s * s;
        }
      }
      const rms = Math.sqrt(sumSq / n);

      // 2. Decide the gate target for this chunk, then ramp sample-by-sample.
      let target = 1;
      if (this._gateEnabled) {
        if (rms >= this._thresholdLin) {
          this._holdRemaining = this._holdSamples; // re-arm the hold
        } else if (this._holdRemaining > 0) {
          this._holdRemaining -= n; // coasting through the hold window
        } else {
          target = 0;
        }
      }

      // 3. Apply the (smoothed) gain and pack to Int16.
      const out = new Int16Array(n);
      let gain = this._gateGain;
      for (let i = 0; i < n; i++) {
        const coef = target > gain ? this._attackCoef : this._releaseCoef;
        gain = target + (gain - target) * coef;
        const s = dec[i] * gain;
        const clamped = s < -1 ? -1 : s > 1 ? 1 : s;
        out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      }
      this._gateGain = gain;

      // Shift the scratch buffer to keep only the trailing unused samples.
      const consumed = Math.floor(n * r);
      this._scratch = this._scratch.slice(consumed);

      // Live input level for the Settings meter (raw RMS, pre-gate).
      this.port.postMessage({ kind: "level", rms });

      if (this._enabled) {
        this.port.postMessage(out.buffer, [out.buffer]);
      }
      // When disabled (mic muted) we silently consume input so the worklet
      // stays alive and the buffer never grows unbounded.
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;
    const mono = input[0];
    if (mono.length > 0) this._ingest(mono);
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
