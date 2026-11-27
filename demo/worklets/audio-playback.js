// @ts-check
/**
 * AudioWorkletProcessor that plays back Float32 mono samples received from
 * the main thread, upsampling whatever incoming rate the server uses
 * (typically 24 kHz PCM16) to the AudioContext rate (typically 48 kHz).
 *
 * Lifecycle / messaging:
 *
 *   main -> worklet:
 *     { kind: "config", inputRate: 24000 }              one-shot at startup
 *     { kind: "audio", samples: Float32Array, key }          (transferable) per chunk
 *     { kind: "clear", clearId? }                       wipe queue (barge-in)
 *     { kind: "pause" }                                 hold the queue (speech that may be a cough)
 *     { kind: "resume" }                                play on, replaying from the last gap between words
 *     { kind: "gain", value }                           playback volume (ducking); resume and clear reset it
 *
 *   worklet -> main:
 *     { kind: "cleared", clearId, played }             per-item rendered PCM counts
 *     { kind: "stats", queuedMs, played }               every ~250 ms
 *     { kind: "underrun" }                              every time the queue
 *                                                      runs dry mid-playback
 *
 * Underrun strategy: output silence. We do NOT hold the last sample (that
 * tends to produce audible clicks/buzzes when long gaps appear between
 * TTS chunks). A short ramp-out + ramp-in at boundaries would be nicer but
 * the server's 30 ms cadence makes underruns visible only at end of turn.
 */

const STATS_INTERVAL_FRAMES = 12000;
const FADE_FRAMES = 32;
// Resuming after a pause replays from the last quiet stretch of at least
// GAP_MIN_MS within the last REWIND_MAX_MS played, so speech does not pick up
// in the middle of a word. Same rule as the server's WebRTC track.
const REWIND_MAX_MS = 600;
const GAP_WINDOW_MS = 10;
const GAP_MIN_MS = 50;
const GAP_LEVEL = 0.1;

/**
 * Samples at the end of `played` to replay when playback resumes.
 * @param {Float32Array} played @param {number} rate
 */
function resumeRewindSamples(played, rate) {
  const window = Math.max(1, Math.floor(rate * GAP_WINDOW_MS / 1000));
  const count = Math.floor(Math.min(played.length, Math.floor(rate * REWIND_MAX_MS / 1000)) / window);
  if (count === 0) return 0;
  const start = played.length - count * window;
  const level = new Float32Array(count);
  let loudest = 0;
  for (let w = 0; w < count; w++) {
    let energy = 0;
    for (let i = start + w * window; i < start + (w + 1) * window; i++) energy += played[i] * played[i];
    level[w] = Math.sqrt(energy / window);
    loudest = Math.max(loudest, level[w]);
  }
  const minRun = Math.ceil(GAP_MIN_MS / GAP_WINDOW_MS);
  let run = 0;
  for (let w = count - 1; w >= 0; w--) {
    if (level[w] < loudest * GAP_LEVEL) {
      run += 1;
      continue;
    }
    if (run >= minRun) return w + run === count - 1 ? 0 : (count - (w + run)) * window;
    run = 0;
  }
  return 0;
}

class AudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._inputRate = 24000;
    this._stepRatio = this._inputRate / sampleRate;
    this._queue = [];
    this._playedByItem = new Map();
    this._readIdx = 0;
    this._fracPos = 0;
    this._playing = false;
    this._framesSinceStats = 0;
    this._totalPlayed = 0;
    this._fadeIn = 0;
    this._fadeOut = 0;
    this._lastSample = 0;
    this._gain = 1;
    this._paused = false;
    this._pauseFade = 0;
    this._history = new Float32Array(Math.ceil(sampleRate * REWIND_MAX_MS / 1000));
    this._historyLength = 0;
    this._historyEnd = 0;
    /** @type {Float32Array | null} */
    this._replay = null;
    this._replayIdx = 0;

    this.port.onmessage = (e) => {
      const data = e.data;
      if (!data || typeof data !== "object") return;
      switch (data.kind) {
        case "config":
          if (typeof data.inputRate === "number" && data.inputRate > 0) {
            this._inputRate = data.inputRate;
            this._stepRatio = this._inputRate / sampleRate;
          }
          break;
        case "audio":
          if (data.samples instanceof Float32Array && data.samples.length > 0) {
            this._queue.push({ samples: data.samples, key: data.key });
            // A clear may still be fading the previous response out. New
            // audio must cancel that stop or it will strand the new queue.
            if (!this._playing || this._fadeOut > 0) {
              this._playing = true;
              this._fadeIn = FADE_FRAMES;
              this._fadeOut = 0;
            }
          }
          break;
        case "clear":
          if (data.clearId !== undefined) {
            this.port.postMessage({ kind: "cleared", clearId: data.clearId, played: [...this._playedByItem] });
          }
          this._playedByItem.clear();
          this._queue.length = 0;
          this._readIdx = 0;
          this._fracPos = 0;
          this._fadeOut = FADE_FRAMES;
          this._paused = false;
          this._pauseFade = 0;
          this._replay = null;
          this._historyLength = 0;
          this._gain = 1;
          break;
        case "pause":
          if (!this._paused) {
            this._paused = true;
            this._pauseFade = this._playing ? FADE_FRAMES : 0;
          }
          break;
        case "resume":
          this._gain = 1;
          if (this._paused) {
            this._paused = false;
            this._pauseFade = 0;
            const played = this._played();
            const rewind = resumeRewindSamples(played, sampleRate);
            if (rewind > 0) {
              this._replay = played.slice(played.length - rewind);
              this._replayIdx = 0;
              this._historyLength -= rewind;
              this._historyEnd = (this._historyEnd - rewind + this._history.length) % this._history.length;
            }
            if (this._replay || this._queue.length) {
              this._playing = true;
              this._fadeIn = FADE_FRAMES;
              this._fadeOut = 0;
            }
          }
          break;
        case "gain":
          if (typeof data.value === "number" && data.value >= 0) this._gain = data.value;
          break;
      }
    };
  }

  /** @param {number} sample */
  _remember(sample) {
    this._history[this._historyEnd] = sample;
    this._historyEnd = (this._historyEnd + 1) % this._history.length;
    this._historyLength = Math.min(this._historyLength + 1, this._history.length);
  }

  /** The last samples played, oldest first. */
  _played() {
    const out = new Float32Array(this._historyLength);
    const start = (this._historyEnd - this._historyLength + this._history.length) % this._history.length;
    for (let i = 0; i < out.length; i++) out[i] = this._history[(start + i) % this._history.length];
    return out;
  }

  _queuedSamples() {
    let total = -this._readIdx;
    for (const buf of this._queue) total += buf.samples.length;
    return Math.max(0, total);
  }

  /** Linear-interp read at the current fractional position. */
  _readInterpolated() {
    if (this._queue.length === 0) return null;
    const head = this._queue[0].samples;
    const idx = this._readIdx;
    const frac = this._fracPos;

    let a = head[idx];
    let b;
    if (idx + 1 < head.length) {
      b = head[idx + 1];
    } else if (this._queue.length > 1) {
      b = this._queue[1].samples[0];
    } else {
      b = a;
    }
    return a + (b - a) * frac;
  }

  /** Advance the read position by `stepRatio`; pop consumed buffers. */
  _advance() {
    let remaining = this._stepRatio;
    while (remaining > 0 && this._queue.length) {
      const { key, samples } = this._queue[0];
      const consumed = Math.min(remaining, samples.length - this._readIdx - this._fracPos);
      this._playedByItem.set(key, (this._playedByItem.get(key) ?? 0) + consumed);
      const position = this._readIdx + this._fracPos + consumed;
      this._readIdx = Math.floor(position);
      this._fracPos = position - this._readIdx;
      remaining -= consumed;
      if (this._readIdx >= samples.length) {
        this._readIdx = 0;
        this._fracPos = 0;
        this._queue.shift();
      }
    }
  }

  process(_, outputs) {
    const channels = outputs[0];
    if (!channels || channels.length === 0) return true;
    const out = channels[0];
    const stereo = channels.length > 1 ? channels[1] : null;

    for (let i = 0; i < out.length; i++) {
      let sample = 0;
      const held = this._paused && this._pauseFade === 0;

      if (this._playing && !held) {
        // Replayed samples were already counted as played: they do not advance the queue.
        const replaying = this._replay !== null;
        let v = null;
        if (this._replay) {
          v = this._replay[this._replayIdx++];
          if (this._replayIdx >= this._replay.length) this._replay = null;
        } else {
          v = this._readInterpolated();
        }
        if (v === null) {
          // Underrun: try to ramp out cleanly to avoid clicks.
          sample = this._lastSample * Math.max(0, 1 - 1 / FADE_FRAMES);
          this._lastSample = sample;
          if (Math.abs(sample) < 1e-4) {
            this._playing = false;
            this._lastSample = 0;
            // What plays after a gap is a new utterance: never replay across it.
            this._historyLength = 0;
            this.port.postMessage({ kind: "underrun" });
          }
        } else {
          sample = v;
          this._lastSample = v;
          this._remember(v);
          if (!replaying) this._advance();
        }

        if (this._fadeIn > 0) {
          const gain = 1 - this._fadeIn / FADE_FRAMES;
          sample *= gain;
          this._fadeIn -= 1;
        }
        if (this._fadeOut > 0) {
          const gain = this._fadeOut / FADE_FRAMES;
          sample *= gain;
          this._fadeOut -= 1;
          if (this._fadeOut === 0) {
            this._playing = false;
            this._lastSample = 0;
          }
        }
        if (this._pauseFade > 0) {
          sample *= this._pauseFade / FADE_FRAMES;
          this._pauseFade -= 1;
        }

        this._totalPlayed += 1;
      }

      sample *= this._gain;
      out[i] = sample;
      if (stereo) stereo[i] = sample;
    }

    this._framesSinceStats += out.length;
    if (this._framesSinceStats >= STATS_INTERVAL_FRAMES) {
      this._framesSinceStats = 0;
      const queuedSamples = this._queuedSamples();
      const queuedMs = (queuedSamples / this._inputRate) * 1000;
      this.port.postMessage({ kind: "stats", queuedMs, played: this._totalPlayed });
    }

    return true;
  }
}

registerProcessor("audio-playback", AudioPlaybackProcessor);
