// @ts-check
/**
 * A photoreal coach: a Gaussian-splat head (LAM output) drawn by
 * @myned-ai/gsplat-flame-avatar-renderer and driven with ARKit blendshapes.
 *
 * Same interface and lip-sync sources as the TalkingHead avatar in avatar.js,
 * but this renderer has no audio of its own, so the voice plays through
 * TalkingHead's streaming worklet on our own AudioContext, and every frame the
 * renderer asks for the face's 52 ARKit weights: mouth shapes from the server
 * viseme cues (or HeadAudio), plus blinks, mood and reactions. The baked
 * clips in the asset move the head (idle, and six speaking clips).
 */

import { GaussianSplatRenderer } from "gsplat";
import { HeadAudio } from "headaudio";
import { coarticulate, cueShape, FORCE_HEADAUDIO, JAW_LEVEL, turnGap, Upsampler, VISEME_ARKIT } from "./lipsync.js";

const ARKIT = [
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "mouthCheekPuff",
  "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
  "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
  "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
  "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
  "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
  "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
  "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
  "tongueOut",
];

/** @typedef {Record<string, number>} Shape */

/** Resting faces per mood, and short reactions layered on top. */
const MOODS = /** @type {Record<string, Shape>} */ ({
  happy: { mouthSmileLeft: 0.2, mouthSmileRight: 0.2, cheekSquintLeft: 0.06, cheekSquintRight: 0.06 },
  neutral: { mouthSmileLeft: 0.08, mouthSmileRight: 0.08 },
});
const REACTIONS = /** @type {Record<string, Shape>} */ ({
  correct: {
    mouthSmileLeft: 0.55, mouthSmileRight: 0.55, cheekSquintLeft: 0.25, cheekSquintRight: 0.25,
    eyeSquintLeft: 0.15, eyeSquintRight: 0.15, browInnerUp: 0.15,
  },
  partial: { mouthSmileLeft: 0.25, mouthSmileRight: 0.25, browInnerUp: 0.3, browOuterUpLeft: 0.15, browOuterUpRight: 0.15 },
  incorrect: { browInnerUp: 0.35, mouthSmileLeft: 0.12, mouthSmileRight: 0.12, mouthPressLeft: 0.15, mouthPressRight: 0.15 },
  clue: { browOuterUpLeft: 0.3, browOuterUpRight: 0.3, browInnerUp: 0.2, mouthSmileLeft: 0.2, mouthSmileRight: 0.2 },
  skipped: { browInnerUp: 0.4, mouthShrugLower: 0.2, mouthPressLeft: 0.1, mouthPressRight: 0.1 },
  think: { browDownLeft: 0.2, browDownRight: 0.12, mouthPressLeft: 0.15, mouthPressRight: 0.15, mouthLeft: 0.06 },
  celebrate: {
    mouthSmileLeft: 0.75, mouthSmileRight: 0.75, cheekSquintLeft: 0.35, cheekSquintRight: 0.35,
    eyeSquintLeft: 0.2, eyeSquintRight: 0.2, browInnerUp: 0.2,
  },
});
REACTIONS.hint = REACTIONS.clue;

/** @param {number} x */
const smooth = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/**
 * The renderer draws on an opaque canvas. Grab the canvas as it is attached
 * and create its WebGL context with an alpha channel first, so three.js gets
 * that context and the page's backdrop shows through around the head.
 * @param {HTMLElement} container
 */
function transparentCanvas(container) {
  const obs = new MutationObserver((records) => {
    for (const r of records) {
      for (const node of r.addedNodes) {
        if (node instanceof HTMLCanvasElement) {
          node.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false });
          obs.disconnect();
          return;
        }
      }
    }
  });
  obs.observe(container, { childList: true });
  return obs;
}

export class SplatAvatar extends EventTarget {
  /**
   * @param {HTMLElement} container
   * @param {{ name: string, asset: string, lipGain: number, jawGain: number, camera?: { y: number, z: number, pitchDeg: number } }} coach
   */
  constructor(container, coach) {
    super();
    this.container = container;
    this.coach = coach;
    /** @type {any} */
    this.renderer = null;
    /** @type {AudioContext | null} */
    this.ctx = null;
    /** @type {AudioWorkletNode | null} */
    this.player = null;
    /** @type {DelayNode | null} */
    this.voiceDelay = null;
    /** @type {any} */
    this.headaudio = null;
    /** @type {Shape} visemes from HeadAudio, latest values */
    this.haValues = {};
    this.haLast = 0;
    this.headaudioDelay = 0.08; // s
    this.inRate = 24000;
    /** @type {Upsampler | null} */
    this.upsampler = null;
    this.responseId = "";
    this.responseSamples = 0;
    this.cursor = 0; // audio clock (ms) when the queued audio runs out
    /** @type {{ viseme: number, start_s: number, end_s: number }[]} */
    this.pendingCues = [];
    /** @type {{ v: string, level: number, jaw: number, t0: number, t1: number, t2: number }[]} */
    this.cues = [];
    this.serverVisemes = false;
    this.lipsyncSource = "headaudio";
    this.playing = false;
    this.playingSince = 0;
    this.stoppedAt = 0;
    /** @type {number[]} */
    this.lineTimers = [];
    /** @type {AnalyserNode | null} */
    this.outTap = null;
    /** @type {Float32Array | null} */
    this.outBuf = null;
    /** Audible start (ms, avatar clock) of each response's audio. @type {Map<string, number>} */
    this.lineStarts = new Map();
    // Face state
    this.mode = "Idle";
    this.mood = "happy";
    /** @type {{ shape: Shape, until: number } | null} */
    this.reaction = null;
    /** @type {Shape} */
    this.face = Object.fromEntries(ARKIT.map((k) => [k, 0]));
    this.lastFrame = performance.now();
    this.nextBlink = performance.now() + 1500;
    this.blinkAt = -1;
    this.gaze = { yawDeg: 0, pitchDeg: 0 };
    this.nextSaccade = performance.now() + 1200;
  }

  get audioCtx() {
    return this.ctx;
  }

  /** @param {(ev: { lengthComputable: boolean, loaded: number, total: number }) => void} [onProgress] */
  async init(onProgress) {
    const report = (/** @type {number} */ pct) => onProgress?.({ lengthComputable: true, loaded: Math.round(pct), total: 100 });
    await this.initAudio();
    const obs = transparentCanvas(this.container);
    try {
      this.renderer = await GaussianSplatRenderer.create(/** @type {HTMLDivElement} */ (this.container), this.coach.asset, {
        backgroundColor: "0xc2ccd6",
        downloadProgress: (/** @type {number} */ p) => report(p * 90),
        loadProgress: (/** @type {number} */ p) => report(90 + p * 10),
        getChatState: () => this.chatState(),
        getExpressionData: () => this.frame(),
        getGazeOffset: () => this.gaze,
      });
    } finally {
      obs.disconnect();
    }
    const viewer = this.renderer.viewer;
    viewer?.renderer?.setClearColor(0x000000, 0);
    // Close head-and-shoulders: the asset's default camera shows the torso
    // down to where the scan ends in a hard edge.
    const cam = this.coach.camera;
    if (viewer?.camera && cam) {
      viewer.camera.position.set(0, cam.y, cam.z);
      viewer.camera.rotation.set((cam.pitchDeg * Math.PI) / 180, 0, 0);
    }
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") this.headaudio?.start();
      else this.headaudio?.stop();
    });
  }

  async initAudio() {
    const ctx = new AudioContext({ latencyHint: "interactive" });
    this.ctx = ctx;
    await ctx.audioWorklet.addModule("vendor/talkinghead/playback-worklet.js");
    const player = new AudioWorkletNode(ctx, "playback-worklet", {
      processorOptions: { sampleRate: ctx.sampleRate, metrics: { enabled: false } },
    });
    player.port.onmessage = (ev) => {
      const type = ev.data?.type;
      if (type === "playback-started") this.setPlaying(true);
      else if (type === "playback-ended") this.setPlaying(false);
    };
    this.player = player;
    const voice = ctx.createGain();
    player.connect(voice);
    // HeadAudio's visemes trail the sound by ~80 ms, so the audible voice
    // goes through a delay that is switched on only when HeadAudio drives.
    this.voiceDelay = ctx.createDelay(0.5);
    voice.connect(this.voiceDelay).connect(ctx.destination);
    this.outTap = ctx.createAnalyser();
    this.outTap.fftSize = 1024;
    this.voiceDelay.connect(this.outTap);
    this.upsampler = new Upsampler(this.inRate, ctx.sampleRate);
    try {
      await ctx.audioWorklet.addModule("vendor/headaudio/headworklet.min.mjs");
      const ha = new HeadAudio(ctx, { visemeMinMs: 110 });
      await ha.loadModel("vendor/headaudio/model-en-mixed.bin");
      voice.connect(ha);
      ha.onvalue = (/** @type {string} */ key, /** @type {number} */ value) => {
        this.haValues[key.replace("viseme_", "")] = value;
      };
      ha.start?.();
      this.headaudio = ha;
    } catch (err) {
      console.warn("[avatar] HeadAudio unavailable, lips will only move with server visemes", err);
    }
  }

  /** @param {boolean} on */
  setPlaying(on) {
    if (this.playing === on) return;
    this.playing = on;
    if (on) this.playingSince = performance.now();
    else this.stoppedAt = performance.now();
    this.dispatchEvent(new CustomEvent("playback", { detail: { playing: on } }));
  }

  resume() {
    const ctx = this.ctx;
    if (ctx && ctx.state !== "running") return ctx.resume().catch(() => {});
    return Promise.resolve();
  }

  // ── Speech ──────────────────────────────────────────────────────────────

  /** @param {string} responseId */
  beginResponse(responseId) {
    if (responseId === this.responseId) return;
    this.responseId = responseId;
    this.responseSamples = 0;
    this.pendingCues = [];
    this.serverVisemes = false;
    this.upsampler?.reset();
    this.mode = "Idle";
  }

  /** @param {string} responseId @param {{ viseme: number, start_s: number, end_s: number }[]} cues */
  addVisemes(responseId, cues) {
    if (FORCE_HEADAUDIO) return;
    this.beginResponse(responseId);
    this.pendingCues.push(...coarticulate(cues));
    this.serverVisemes = true;
    this.lipsyncSource = "server";
  }

  /** @param {string} responseId @param {Int16Array} pcm */
  feed(responseId, pcm) {
    const ctx = this.ctx;
    if (!ctx || !this.player || !this.upsampler) return;
    this.beginResponse(responseId);
    if (!this.serverVisemes) this.lipsyncSource = "headaudio";
    if (this.responseSamples === 0 && this.voiceDelay) this.voiceDelay.delayTime.value = this.serverVisemes ? 0 : this.headaudioDelay;

    // The player is a FIFO that starts a chunk as soon as it arrives when its
    // buffer is empty, so a chunk starts at max(now, end of the previous one).
    const now = ctx.currentTime * 1000;
    const latencyMs = ((ctx.baseLatency || 0) + (ctx.outputLatency || 0)) * 1000 || 20;
    const durMs = (pcm.length / this.inRate) * 1000;
    const r0 = (this.responseSamples / this.inRate) * 1000;
    let start = Math.max(now + latencyMs, this.cursor);
    const gap = this.responseSamples === 0 ? turnGap(start, this.cursor) : 0;
    if (gap > 0) {
      const silence = this.upsampler.process(new Int16Array(Math.round((gap * this.inRate) / 1000))).slice();
      this.player.port.postMessage({ type: "audioData", data: silence.buffer }, [silence.buffer]);
      start += gap;
    }
    this.cursor = start + durMs;
    this.lineStarts.set(responseId, this.lineStarts.get(responseId) ?? start);
    if (this.responseSamples === 0) {
      const id = responseId;
      this.lineTimers.push(
        window.setTimeout(() => this.dispatchEvent(new CustomEvent("line-start", { detail: { responseId: id } })), start - now),
      );
    }
    this.responseSamples += pcm.length;
    if (this.pendingCues.length) {
      const later = [];
      for (const cue of this.pendingCues) {
        const ms = cue.start_s * 1000;
        if (ms < r0 + durMs) this.scheduleCue(cue, start + (ms - r0));
        else later.push(cue);
      }
      this.pendingCues = later;
    }
    const chunk = this.upsampler.process(pcm).slice();
    this.player.port.postMessage({ type: "audioData", data: chunk.buffer }, [chunk.buffer]);
  }

  /**
   * Same envelope as TalkingHead's: rise from 2/3 of a duration before the
   * cue, peak mid-cue, fall by 1.5 durations. Times on the audio clock (ms).
   * @param {{ viseme: number, start_s: number, end_s: number }} cue @param {number} t
   */
  scheduleCue(cue, t) {
    const s = cueShape(cue, this.coach);
    if (!s) return;
    this.cues.push({ v: s.v, level: s.level, jaw: s.jaw, t0: t - (2 * s.d) / 3, t1: t + s.d / 2, t2: t + s.d + s.d / 2 });
  }

  endResponse() {
    this.player?.port.postMessage({ type: "no-more-data" });
  }

  /** Loudness (RMS) of the voice being played right now, for the mic's echo gate. */
  outputLevel() {
    const tap = this.outTap;
    if (!tap) return 0;
    const buf = this.outBuf || (this.outBuf = new Float32Array(tap.fftSize));
    tap.getFloatTimeDomainData(buf);
    let e = 0;
    for (let i = 0; i < buf.length; i++) e += buf[i] * buf[i];
    return Math.sqrt(e / buf.length);
  }

  /**
   * How much of a response the student has heard (ms of its audio), or -1.
   * @param {string} responseId
   */
  heardMs(responseId) {
    const start = this.lineStarts.get(responseId);
    if (start === undefined) return -1;
    return Math.max(0, (this.ctx?.currentTime ?? 0) * 1000 - start);
  }

  interrupt() {
    this.lineTimers.forEach(clearTimeout);
    this.lineTimers = [];
    this.player?.port.postMessage({ type: "stop" });
    this.pendingCues = [];
    this.cues = [];
    this.responseId = "";
    this.cursor = 0;
    this.setPlaying(false);
  }

  // ── Per-frame face ──────────────────────────────────────────────────────

  chatState() {
    // Hold the speaking clips across the short gaps between lines.
    if (this.playing || performance.now() - this.stoppedAt < 700) return "Responding";
    return this.mode;
  }

  /** The 52 ARKit weights for this frame. */
  frame() {
    const t = performance.now();
    const dt = Math.min(100, t - this.lastFrame);
    this.lastFrame = t;
    /** @type {Shape} */
    const target = Object.fromEntries(ARKIT.map((k) => [k, 0]));
    const add = (/** @type {Shape} */ shape, /** @type {number} */ w) => {
      for (const k in shape) target[k] += shape[k] * w;
    };

    // Mouth
    const mouth = this.mouthShapes();
    let speaking = 0;
    for (const v in mouth) {
      const w = mouth[v].w;
      if (w <= 0) continue;
      add(VISEME_ARKIT[v] ?? {}, w);
      target.jawOpen += mouth[v].jaw;
      speaking = Math.max(speaking, w);
    }
    // Mood and reactions; a smile flattens while the lips form words.
    if (this.reaction && t > this.reaction.until) this.reaction = null;
    const rest = this.reaction?.shape ?? MOODS[this.mood] ?? MOODS.neutral;
    const smileScale = 1 - 0.5 * Math.min(1, speaking * 1.5);
    for (const k in rest) target[k] += rest[k] * (k.startsWith("mouthSmile") ? smileScale : 1);

    // Blinks every 2-6 s (and one right after a reaction starts).
    if (t >= this.nextBlink) {
      this.blinkAt = t;
      this.nextBlink = t + 2000 + Math.random() * 4000;
    }
    const b = (t - this.blinkAt) / 160;
    const blink = b >= 0 && b <= 1 ? Math.sin(Math.PI * b) : 0;
    target.eyeBlinkLeft += blink;
    target.eyeBlinkRight += blink;

    // Gaze: on the student, with small saccades; up and aside while thinking.
    if (t >= this.nextSaccade) {
      const thinking = this.mode === "Thinking";
      this.gaze = thinking
        ? { yawDeg: 6 + Math.random() * 4, pitchDeg: 5 + Math.random() * 3 }
        : { yawDeg: (Math.random() - 0.5) * 3, pitchDeg: (Math.random() - 0.5) * 2 };
      this.nextSaccade = t + (thinking ? 900 : 1200 + Math.random() * 2200);
    }

    // Ease toward the target: quick for the mouth, slower for the rest.
    const fast = 1 - Math.exp(-dt / 30);
    const slow = 1 - Math.exp(-dt / 140);
    for (const k of ARKIT) {
      const goal = Math.max(0, Math.min(1, target[k]));
      const rate = k.startsWith("eyeBlink") ? 1 : k.startsWith("mouth") || k.startsWith("jaw") || k === "tongueOut" ? fast : slow;
      this.face[k] += (goal - this.face[k]) * rate;
    }
    return { ...this.face };
  }

  /** Current weight and jaw opening per Oculus viseme. */
  mouthShapes() {
    /** @type {Record<string, { w: number, jaw: number }>} */
    const out = {};
    const put = (/** @type {string} */ v, /** @type {number} */ w, /** @type {number} */ jaw) => {
      const cur = out[v] ?? (out[v] = { w: 0, jaw: 0 });
      cur.w = Math.max(cur.w, w);
      cur.jaw = Math.max(cur.jaw, jaw);
    };
    if (this.serverVisemes || this.cues.length) {
      const now = (this.ctx?.currentTime ?? 0) * 1000;
      this.cues = this.cues.filter((c) => c.t2 > now);
      for (const c of this.cues) {
        if (now < c.t0) continue;
        const env = now < c.t1 ? smooth((now - c.t0) / (c.t1 - c.t0)) : 1 - smooth((now - c.t1) / (c.t2 - c.t1));
        put(c.v, c.level * env, c.jaw * env);
      }
      return out;
    }
    if (this.headaudio) {
      const t = performance.now();
      this.headaudio.update(Math.min(100, t - (this.haLast || t - 16)));
      this.haLast = t;
      let jaw = 0;
      for (const [v, x] of Object.entries(this.haValues)) jaw += (JAW_LEVEL[v] ?? 0) * x;
      jaw = Math.min(0.45, jaw * this.coach.jawGain * 1.5);
      for (const [v, x] of Object.entries(this.haValues)) put(v, Math.min(1, x * this.coach.lipGain * 1.2), 0);
      put("aa", out.aa?.w ?? 0, jaw);
    }
    return out;
  }

  // ── Body language ───────────────────────────────────────────────────────

  /** @param {string} name */
  setMood(name) {
    this.mood = name in MOODS ? name : "neutral";
  }

  /** @param {string} name @param {number} ms */
  showReaction(name, ms) {
    const shape = REACTIONS[name];
    if (shape) this.reaction = { shape, until: performance.now() + ms };
  }

  listen() {
    this.mode = "Listening";
    this.reaction = null;
    this.gaze = { yawDeg: 0, pitchDeg: 0 };
  }

  think() {
    this.mode = "Thinking";
    this.showReaction("think", 2500);
    this.nextSaccade = 0;
  }

  /** @param {string | null | undefined} verdict @param {{ streak?: number }} [ctx] */
  react(verdict, ctx = {}) {
    this.mode = "Idle";
    switch (verdict) {
      case "correct":
        this.setMood("happy");
        this.showReaction((ctx.streak ?? 0) >= 3 ? "celebrate" : "correct", 2200);
        break;
      case "partial":
      case "incorrect":
      case "clue":
      case "hint":
      case "skipped":
        this.setMood("neutral");
        this.showReaction(verdict, 2000);
        break;
      default:
        this.setMood("neutral");
    }
    this.nextBlink = performance.now() + 250;
  }

  celebrate() {
    this.setMood("happy");
    this.showReaction("celebrate", 3500);
  }
}
