// @ts-check
// Silero runs on the existing, echo-cancelled mic stream. Local detection only
// ducks playback; server VAD still owns cancellation and turn boundaries.
const ASSETS = new URL("../vendor/local-vad/", import.meta.url).href;
const ASSET_VERSION = "vad-0.0.31-ort-1.22.0";
let libraries;

function loadScript(name) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    const timer = setTimeout(() => finish(new Error(`Timed out loading ${name}`)), 15_000);
    function finish(error = null) {
      clearTimeout(timer);
      script.onload = script.onerror = null;
      if (error) { script.remove(); reject(error); }
      else resolve(undefined);
    }
    script.src = `${ASSETS}${name}?v=${ASSET_VERSION}`;
    script.onload = () => finish();
    script.onerror = () => finish(new Error(`Could not load ${name}`));
    document.head.append(script);
  });
}

async function loadLibraries() {
  if (!libraries) {
    libraries = (async () => {
      await loadScript("ort.wasm.min.js");
      await loadScript("bundle.min.js");
      return globalThis.vad.MicVAD;
    })().catch((error) => { libraries = null; throw error; });
  }
  return libraries;
}

/** Fast candidate detector with hysteresis and recovery from stalled input. */
export class SpeechCandidate {
  constructor(onChange) {
    this.onChange = onChange;
    this.active = false;
    this.quietFrames = 0;
    this.lastFrameAt = 0;
  }

  update(probability, now) {
    this.lastFrameAt = now;
    if (!Number.isFinite(probability)) { this.reset(); return; }
    if (probability >= 0.5) {
      this.quietFrames = 0;
      this.setActive(true);
    } else if (probability < 0.35) {
      // v6 frames are 32 ms: release a rejected candidate after 64 ms.
      if (++this.quietFrames >= 2) this.setActive(false);
    } else {
      this.quietFrames = 0;
    }
  }

  checkStale(now) {
    if (now - this.lastFrameAt >= 500) this.reset();
  }

  setActive(active) {
    if (active === this.active) return;
    this.active = active;
    this.onChange(active);
  }

  reset() {
    this.quietFrames = 0;
    this.setActive(false);
  }
}

export class LocalVad {
  constructor({ audioContext, stream, onChange, isEnabled, load = loadLibraries }) {
    this.context = audioContext;
    this.stream = stream;
    this.isEnabled = isEnabled;
    this.load = load;
    this.candidate = new SpeechCandidate(onChange);
    this.instance = null;
    this.closed = false;
    this.timer = 0;
  }

  async start() {
    try {
      const MicVAD = await this.load();
      if (this.closed) return;
      const instance = await MicVAD.new({
        model: "v6",
        audioContext: this.context,
        baseAssetPath: ASSETS,
        onnxWASMBasePath: ASSETS,
        // Single-threaded WASM works without cross-origin isolation headers.
        ortConfig: (ort) => { ort.env.wasm.numThreads = 1; },
        getStream: async () => this.stream,
        pauseStream: async () => {}, // The conversation owns these tracks.
        resumeStream: async () => this.stream,
        processorType: "AudioWorklet",
        onFrameProcessed: ({ isSpeech }) => {
          if (this.closed || !this.isEnabled()) this.candidate.reset();
          else this.candidate.update(isSpeech, performance.now());
        },
        onSpeechStart: () => {},
        onSpeechRealStart: () => {},
        onSpeechEnd: () => {},
        onVADMisfire: () => {},
      });
      this.instance = instance;
      if (this.closed) { await this.close(); return; }
      this.timer = setInterval(() => {
        if (!this.isEnabled()) this.candidate.reset();
        this.candidate.checkStale(performance.now());
        if (instance.errored) {
          console.warn("[audio] Local VAD failed; using server VAD:", instance.errored);
          void this.close();
        }
      }, 100);
    } catch (error) {
      this.candidate.reset();
      if (!this.closed) console.warn("[audio] Local VAD unavailable; using server VAD:", error);
    }
  }

  async close() {
    this.closed = true;
    clearInterval(this.timer);
    this.candidate.reset();
    const instance = this.instance;
    this.instance = null;
    try { await instance?.destroy(); } catch { /* Context may already be closed. */ }
  }
}
