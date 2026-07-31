// @ts-check
/**
 * Keep a bounded copy of the PCM16 frames sent over the realtime WebSocket and
 * turn the backend's VAD boundaries into browser-playable WAV blobs.
 *
 * This records the post-resampling, post-noise-gate signal—not a second raw-mic
 * capture—so replay is as close as the browser can get to the audio delivered
 * to the backend. Nothing leaves the page beyond the existing realtime stream.
 */

export const USER_AUDIO_SAMPLE_RATE = 16000;
const BYTES_PER_SAMPLE = 2;
const DEFAULT_PREROLL_MS = 5000;
const DEFAULT_MAX_BUFFER_MS = 120000;

/** @param {DataView} view @param {number} offset @param {string} value */
function _writeAscii(view, offset, value) {
  for (let i = 0; i < value.length; i++) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

/**
 * Wrap little-endian mono PCM16 in a standard WAV container.
 * @param {Uint8Array} pcm
 * @param {number} [sampleRate]
 * @returns {Blob}
 */
export function pcm16ToWavBlob(pcm, sampleRate = USER_AUDIO_SAMPLE_RATE) {
  const dataLength = pcm.byteLength - (pcm.byteLength % BYTES_PER_SAMPLE);
  const wav = new ArrayBuffer(44 + dataLength);
  const view = new DataView(wav);
  _writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  _writeAscii(view, 8, "WAVE");
  _writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM fmt chunk length
  view.setUint16(20, 1, true); // linear PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * BYTES_PER_SAMPLE, true);
  view.setUint16(32, BYTES_PER_SAMPLE, true);
  view.setUint16(34, 16, true);
  _writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);
  new Uint8Array(wav, 44).set(pcm.subarray(0, dataLength));
  return new Blob([wav], { type: "audio/wav" });
}

/** @param {Uint8Array} first @param {Uint8Array} second */
function _concat(first, second) {
  const result = new Uint8Array(first.byteLength + second.byteLength);
  result.set(first, 0);
  result.set(second, first.byteLength);
  return result;
}

export class SentAudioRecorder {
  /**
   * @param {{ sampleRate?: number, preRollMs?: number, maxBufferMs?: number }} [options]
   */
  constructor(options = {}) {
    this.sampleRate = options.sampleRate ?? USER_AUDIO_SAMPLE_RATE;
    this._preRollSamples = Math.round(
      (this.sampleRate * (options.preRollMs ?? DEFAULT_PREROLL_MS)) / 1000,
    );
    this._maxBufferSamples = Math.round(
      (this.sampleRate * (options.maxBufferMs ?? DEFAULT_MAX_BUFFER_MS)) / 1000,
    );
    /** @type {{ startSample: number, endSample: number, bytes: Uint8Array }[]} */
    this._chunks = [];
    this._sentSamples = 0;
    /** @type {{ itemId: string, requestedStartSample: number } | null} */
    this._active = null;
    this._lastItemId = "";
    this._lastItemPcm = new Uint8Array(0);
  }

  /** Store one PCM16 frame that was actually sent to the backend.
   * @param {ArrayBuffer} buffer */
  append(buffer) {
    const evenLength = buffer.byteLength - (buffer.byteLength % BYTES_PER_SAMPLE);
    if (evenLength <= 0) return;
    const bytes = new Uint8Array(buffer.slice(0, evenLength));
    const startSample = this._sentSamples;
    const endSample = startSample + evenLength / BYTES_PER_SAMPLE;
    this._chunks.push({ startSample, endSample, bytes });
    this._sentSamples = endSample;
    this._prune();
  }

  /**
   * Remember where the backend says this speech item began. The event normally
   * arrives after confirmation, so the bounded pre-roll retains its onset.
   * @param {{ itemId?: string, audioStartMs?: number }} boundary
   */
  speechStarted(boundary) {
    const itemId = boundary.itemId || `audio_${this._sentSamples}`;
    const requestedStartSample = this._sampleAtMs(boundary.audioStartMs, this._sentSamples);
    this._active = { itemId, requestedStartSample };
    if (itemId !== this._lastItemId) {
      this._lastItemId = itemId;
      this._lastItemPcm = new Uint8Array(0);
    }
    this._prune();
  }

  /**
   * Finalize the active VAD segment. Reopened segments carrying the same
   * item_id replace the prior recording with their concatenation, matching the
   * chat view's one-row-per-item behavior.
   * @param {{ itemId?: string, audioEndMs?: number }} boundary
   * @returns {{ itemId: string, audio: Blob, durationMs: number, truncated: boolean } | null}
   */
  speechStopped(boundary) {
    const active = this._active;
    if (!active) return null;
    this._active = null;

    const itemId = boundary.itemId || active.itemId;
    const availableStart = this._chunks[0]?.startSample ?? this._sentSamples;
    const startSample = Math.max(active.requestedStartSample, availableStart);
    let endSample = this._sampleAtMs(boundary.audioEndMs, this._sentSamples);
    if (endSample <= startSample) endSample = this._sentSamples;
    endSample = Math.min(endSample, this._sentSamples);

    const segment = this._slice(startSample, endSample);
    if (segment.byteLength === 0) {
      this._prune();
      return null;
    }

    if (itemId !== this._lastItemId) {
      this._lastItemId = itemId;
      this._lastItemPcm = new Uint8Array(0);
    }
    this._lastItemPcm = _concat(this._lastItemPcm, segment);
    const durationMs =
      (this._lastItemPcm.byteLength / BYTES_PER_SAMPLE / this.sampleRate) * 1000;
    const result = {
      itemId,
      audio: pcm16ToWavBlob(this._lastItemPcm, this.sampleRate),
      durationMs,
      truncated: active.requestedStartSample < availableStart,
    };
    this._prune();
    return result;
  }

  reset() {
    this._chunks = [];
    this._sentSamples = 0;
    this._active = null;
    this._lastItemId = "";
    this._lastItemPcm = new Uint8Array(0);
  }

  /** @param {number | undefined} ms @param {number} fallback */
  _sampleAtMs(ms, fallback) {
    if (!Number.isFinite(ms) || Number(ms) < 0) return fallback;
    return Math.max(0, Math.min(Math.round((Number(ms) * this.sampleRate) / 1000), this._sentSamples));
  }

  /** @param {number} startSample @param {number} endSample */
  _slice(startSample, endSample) {
    /** @type {Uint8Array[]} */
    const parts = [];
    let length = 0;
    for (const chunk of this._chunks) {
      const overlapStart = Math.max(startSample, chunk.startSample);
      const overlapEnd = Math.min(endSample, chunk.endSample);
      if (overlapEnd <= overlapStart) continue;
      const from = (overlapStart - chunk.startSample) * BYTES_PER_SAMPLE;
      const to = (overlapEnd - chunk.startSample) * BYTES_PER_SAMPLE;
      const part = chunk.bytes.slice(from, to);
      parts.push(part);
      length += part.byteLength;
    }
    const result = new Uint8Array(length);
    let offset = 0;
    for (const part of parts) {
      result.set(part, offset);
      offset += part.byteLength;
    }
    return result;
  }

  _prune() {
    const hardFloor = Math.max(0, this._sentSamples - this._maxBufferSamples);
    const softFloor = this._active
      ? this._active.requestedStartSample
      : Math.max(0, this._sentSamples - this._preRollSamples);
    const floor = Math.max(hardFloor, softFloor);
    while (this._chunks.length && this._chunks[0].endSample <= floor) {
      this._chunks.shift();
    }
  }
}
