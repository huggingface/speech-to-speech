// @ts-check
/**
 * Orb spectrum visualiser. Each animation frame it reads two AnalyserNodes (the
 * mic input and the TTS output) and maps the low-frequency speech energy onto
 * the orb's CSS custom properties:
 *   - `--bar0`..`--bar4`   the 5-band level meter
 *   - `--ai-audio-level`   the global "Reachy talks" glow / scale pulse
 *
 * The bottom of the FFT is where speech energy lives, so the band edges stay
 * low — that keeps the bars dancing on voice rather than on noise. While the AI
 * is speaking we source the bars from the OUTPUT analyser so the orb pulses with
 * Reachy's voice instead of sitting dead while the user is silent.
 */

// Exported so the client can size its AnalyserNodes to match our buffer.
export const VIS_FFT_SIZE = 256;
const VIS_BAND_COUNT = 5;
const VIS_BAND_EDGES = [2, 5, 9, 16, 28, 52];
const VIS_ATTACK = 0.6; // weight for new sample on upswing (snappy)
const VIS_RELEASE = 0.18; // weight for new sample on decay (gentle fade)

export class OrbVisualiser {
  /**
   * @param {AnalyserNode} micAnalyser
   * @param {AnalyserNode} outAnalyser
   * @param {() => boolean} isAiSpeaking Source the bars from the AI output when
   *   true, otherwise from the mic.
   */
  constructor(micAnalyser, outAnalyser, isAiSpeaking) {
    this._mic = micAnalyser;
    this._out = outAnalyser;
    this._isAiSpeaking = isAiSpeaking;
    this._buf = new Uint8Array(micAnalyser.frequencyBinCount);
    this._bands = new Float32Array(VIS_BAND_COUNT);
    this._aiLevel = 0;
    /** @type {number | null} */
    this._frame = null;
  }

  /** Begin the rAF loop (idempotent). */
  start() {
    if (this._frame !== null) return;
    const root = document.documentElement;
    const tick = () => {
      this._frame = requestAnimationFrame(tick);
      this._update(root);
    };
    this._frame = requestAnimationFrame(tick);
  }

  /** Stop the loop and clear the CSS vars so the orb returns to rest. */
  stop() {
    if (this._frame !== null) {
      cancelAnimationFrame(this._frame);
      this._frame = null;
    }
    const root = document.documentElement;
    for (let i = 0; i < VIS_BAND_COUNT; i++) root.style.removeProperty(`--bar${i}`);
    root.style.removeProperty("--ai-audio-level");
  }

  /** @param {HTMLElement} root */
  _update(root) {
    // Mic bars: split FFT into 5 log-ish bands, smooth, write CSS vars.
    const source = this._isAiSpeaking() ? this._out : this._mic;
    source.getByteFrequencyData(this._buf);

    for (let b = 0; b < VIS_BAND_COUNT; b++) {
      const lo = VIS_BAND_EDGES[b];
      const hi = VIS_BAND_EDGES[b + 1];
      let sum = 0;
      let n = 0;
      for (let i = lo; i < hi && i < this._buf.length; i++) {
        sum += this._buf[i];
        n += 1;
      }
      const target = n > 0 ? sum / (n * 255) : 0;
      const prev = this._bands[b];
      const k = target > prev ? VIS_ATTACK : VIS_RELEASE;
      const next = prev + (target - prev) * k;
      this._bands[b] = next;
      root.style.setProperty(`--bar${b}`, next.toFixed(3));
    }

    // Global AI audio level: peak of the output analyser, used by the CSS to
    // make the orb's glow / scale react to Reachy's voice.
    this._out.getByteFrequencyData(this._buf);
    let peak = 0;
    const limit = Math.min(this._buf.length, VIS_BAND_EDGES[VIS_BAND_COUNT]);
    for (let i = 0; i < limit; i++) {
      if (this._buf[i] > peak) peak = this._buf[i];
    }
    const aiTarget = peak / 255;
    const k = aiTarget > this._aiLevel ? VIS_ATTACK : VIS_RELEASE;
    this._aiLevel = this._aiLevel + (aiTarget - this._aiLevel) * k;
    root.style.setProperty("--ai-audio-level", this._aiLevel.toFixed(3));
  }
}
