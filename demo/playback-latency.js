// @ts-check
/** Estimate speech-end to output-device rendering. No physical speaker claim. */
export class PlaybackLatency {
  constructor(rate = 24000) {
    this.rate = rate;
    this.sentSamples = 0;
    this.chunks = [];
    this.speech = null;
    this.responses = new Map();
  }
  capture(byteLength, endedAt) {
    const count = byteLength / 2;
    this.chunks.push({ start: this.sentSamples, end: this.sentSamples + count, endedAt });
    this.sentSamples += count;
    if (this.chunks.length > 3000) this.chunks.shift();
  }
  speechStopped(audioEndMs) {
    const sample = audioEndMs * this.rate / 1000;
    const chunk = [...this.chunks].reverse().find(x => sample > x.start && sample <= x.end);
    this.speech = chunk ? { endedAt: chunk.endedAt - (chunk.end - sample) * 1000 / this.rate, consumed: false } : null;
  }
  responseCreated(id) {
    if (id && this.speech) this.responses.set(id, this.speech);
    if (this.responses.size > 128) this.responses.delete(this.responses.keys().next().value);
  }
  rendered(id, audioTime, context) {
    const speech = this.responses.get(id);
    if (!speech || speech.consumed) return null;
    const stamp = context?.getOutputTimestamp?.();
    if (!stamp || !(stamp.performanceTime > 0) || !Number.isFinite(stamp.contextTime)) return null;
    const renderedAt = stamp.performanceTime + (audioTime - stamp.contextTime) * 1000;
    if (!Number.isFinite(renderedAt) || renderedAt < speech.endedAt) return null;
    speech.consumed = true;
    return { responseId: id, seconds: (renderedAt - speech.endedAt) / 1000 };
  }
  interrupt() {
    this.responses.clear();
    this.speech = null;
  }
}
