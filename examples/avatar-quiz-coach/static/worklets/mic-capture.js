/**
 * Mic capture: downsample the AudioContext rate to 24 kHz PCM16 mono and post
 * 40 ms chunks (ArrayBuffer) plus the chunk RMS to the main thread.
 *
 * While "muted" (the coach is talking in half-duplex mode) it still emits
 * chunks, but silent ones, so the server VAD keeps a steady stream and never
 * hears the coach's own voice leaking from the speakers.
 */
const TARGET_RATE = 24000;
const CHUNK_MS = 40;

class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.chunk = Math.round((TARGET_RATE * CHUNK_MS) / 1000);
    this.out = new Int16Array(this.chunk);
    this.fill = 0;
    this.pos = 0; // fractional read position into the input stream
    this.prev = 0;
    this.muted = false;
    this.port.onmessage = (e) => {
      if (e.data?.kind === "mute") this.muted = !!e.data.value;
    };
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    // Linear interpolation (one sample of delay via `prev`); speech carries
    // little energy above 12 kHz, so aliasing is negligible for STT.
    let i = this.pos;
    let sumSq = 0;
    let n = 0;
    while (i < input.length) {
      const idx = Math.floor(i);
      const frac = i - idx;
      const a = idx === 0 ? this.prev : input[idx - 1];
      const b = input[idx];
      let s = a + (b - a) * frac;
      sumSq += s * s;
      n++;
      if (this.muted) s = 0;
      s = Math.max(-1, Math.min(1, s));
      this.out[this.fill++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.fill === this.chunk) {
        const buf = this.out.slice().buffer;
        this.port.postMessage({ pcm: buf, rms: Math.sqrt(sumSq / Math.max(1, n)) }, [buf]);
        this.fill = 0;
        sumSq = 0;
        n = 0;
      }
      i += this.ratio;
    }
    this.pos = i - input.length;
    this.prev = input[input.length - 1];
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
