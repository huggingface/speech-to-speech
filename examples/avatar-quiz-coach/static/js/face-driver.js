// @ts-check
/**
 * The part of a coach that doesn't depend on how it is drawn: it plays the
 * voice, times the mouth to it and decides the face and head pose for each
 * frame. A renderer extends it and draws `frame()` however it likes
 * (robot-avatar.js turns the mouth into head motion or a beak).
 *
 * Same audio and lip-sync path as the splat coach (splat-avatar.js): the
 * voice plays through TalkingHead's streaming worklet on our own
 * AudioContext; mouth shapes come from the server's viseme cues, or from
 * HeadAudio when the backend sends none. Every frame yields the 52 ARKit
 * weights (mouth, blinks, mood, reactions) plus a head pose and gaze.
 */

import { HeadAudio } from "headaudio";
import { FORCE_HEADAUDIO, JAW_LEVEL, Upsampler, VISEME_ARKIT, coarticulate, cueShape } from "./lipsync.js";

export const ARKIT = [
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
export const MOODS = /** @type {Record<string, Shape>} */ ({
  happy: { mouthSmileLeft: 0.32, mouthSmileRight: 0.32, cheekSquintLeft: 0.1, cheekSquintRight: 0.1, browInnerUp: 0.05 },
  neutral: { mouthSmileLeft: 0.14, mouthSmileRight: 0.14 },
});
export const REACTIONS = /** @type {Record<string, Shape>} */ ({
  correct: {
    mouthSmileLeft: 0.8, mouthSmileRight: 0.8, cheekSquintLeft: 0.5, cheekSquintRight: 0.5,
    eyeSquintLeft: 0.3, eyeSquintRight: 0.3, browInnerUp: 0.2, browOuterUpLeft: 0.3, browOuterUpRight: 0.3,
  },
  partial: { mouthSmileLeft: 0.35, mouthSmileRight: 0.2, browInnerUp: 0.35, browOuterUpLeft: 0.3, browOuterUpRight: 0.1, mouthLeft: 0.1 },
  incorrect: {
    browInnerUp: 0.6, mouthSmileLeft: 0.1, mouthSmileRight: 0.1, mouthPressLeft: 0.3, mouthPressRight: 0.3,
    mouthFrownLeft: 0.15, mouthFrownRight: 0.15,
  },
  clue: { browOuterUpLeft: 0.5, browOuterUpRight: 0.3, browInnerUp: 0.25, mouthSmileLeft: 0.35, mouthSmileRight: 0.25, eyeWideLeft: 0.2, eyeWideRight: 0.2 },
  skipped: { browInnerUp: 0.6, mouthShrugLower: 0.3, mouthPressLeft: 0.2, mouthPressRight: 0.2, mouthFrownLeft: 0.1, mouthFrownRight: 0.1 },
  think: { browDownLeft: 0.35, browDownRight: 0.1, browOuterUpRight: 0.3, mouthPressLeft: 0.25, mouthPressRight: 0.25, mouthLeft: 0.25, eyeSquintLeft: 0.15 },
  celebrate: {
    mouthSmileLeft: 1, mouthSmileRight: 1, cheekSquintLeft: 0.8, cheekSquintRight: 0.8, jawOpen: 0.25,
    eyeSquintLeft: 0.5, eyeSquintRight: 0.5, browInnerUp: 0.3, browOuterUpLeft: 0.5, browOuterUpRight: 0.5,
  },
  surprised: { eyeWideLeft: 0.6, eyeWideRight: 0.6, browInnerUp: 0.6, browOuterUpLeft: 0.6, browOuterUpRight: 0.6, jawOpen: 0.15, mouthFunnel: 0.3 },
});
REACTIONS.hint = REACTIONS.clue;

/** Emoji that `express()` understands, as reactions. */
const EMOJI = /** @type {Record<string, string>} */ ({
  "🙂": "partial", "😊": "correct", "😀": "correct", "😄": "celebrate", "🎉": "celebrate", "🤔": "think",
  "😮": "surprised", "😲": "surprised", "😕": "incorrect", "🙁": "incorrect", "🤷": "skipped",
});

/** @param {number} x */
export const smooth = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
/** @param {number} x @param {number} a @param {number} b */
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));

/**
 * A few cheap, smooth pseudo-random signals for idle motion.
 * @param {number} t seconds @param {number} seed
 */
export function wobble(t, seed) {
  return (
    Math.sin(t * 0.31 + seed * 1.7) * 0.5 +
    Math.sin(t * 0.73 + seed * 4.1) * 0.3 +
    Math.sin(t * 1.37 + seed * 2.3) * 0.2
  );
}

/**
 * Drawing-friendly mouth controls from ARKit weights.
 * @param {Shape} w
 */
export function mouthParams(w) {
  const avg = (/** @type {string} */ a, /** @type {string} */ b) => ((w[a] ?? 0) + (w[b] ?? 0)) / 2;
  const smile = avg("mouthSmileLeft", "mouthSmileRight");
  const frown = avg("mouthFrownLeft", "mouthFrownRight");
  const press = avg("mouthPressLeft", "mouthPressRight");
  const dimple = avg("mouthDimpleLeft", "mouthDimpleRight");
  const stretch = avg("mouthStretchLeft", "mouthStretchRight");
  const lowerDown = avg("mouthLowerDownLeft", "mouthLowerDownRight");
  const roll = ((w.mouthRollLower ?? 0) + (w.mouthRollUpper ?? 0)) / 2;
  const pucker = w.mouthPucker ?? 0;
  const funnel = w.mouthFunnel ?? 0;
  const jaw = w.jawOpen ?? 0;
  const round = clamp(pucker * 0.8 + funnel * 0.9);
  // Lips rolled in (p, b, m) or pressed shut the mouth whatever the jaw does.
  const shut = clamp(roll * 1.1 + (w.mouthClose ?? 0));
  const open = clamp((jaw * 1.35 + lowerDown * 0.35 + funnel * 0.25) * (1 - 0.85 * shut));
  return {
    open,
    // Width: 1 is the resting width; rounding narrows, smiles and e/i widen.
    width: clamp(1 + smile * 0.3 + stretch * 0.3 + dimple * 0.18 + press * 0.08 - round * 0.48 - shut * 0.06, 0.4, 1.6),
    round,
    smile: clamp(smile - frown * 1.2 - shut * 0.1, -1, 1),
    skew: clamp((w.mouthSmileLeft ?? 0) - (w.mouthSmileRight ?? 0) + (w.mouthLeft ?? 0) - (w.mouthRight ?? 0), -1, 1),
    // Upper teeth show on f/v and on spread vowels; the tongue on th, n, d, l.
    teeth: clamp((w.mouthRollLower ?? 0) * (w.mouthShrugUpper ?? 0) + dimple * 0.6 + press * 0.4),
    tongue: clamp((w.tongueOut ?? 0) * 2),
    shut,
  };
}

export class FaceDriver extends EventTarget {
  /**
   * @param {HTMLElement} container
   * @param {{ name: string, lipGain: number, jawGain: number }} coach
   */
  constructor(container, coach) {
    super();
    this.container = container;
    this.coach = coach;
    /** @type {AudioContext | null} */
    this.ctx = null;
    /** @type {AudioWorkletNode | null} */
    this.player = null;
    /** @type {DelayNode | null} */
    this.voiceDelay = null;
    /** @type {GainNode | null} */
    this.voice = null;
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
    /** @type {"Idle" | "Listening" | "Thinking"} */
    this.mode = "Idle";
    this.modeSince = performance.now();
    this.mood = "happy";
    /** @type {{ name: string, shape: Shape, start: number, until: number } | null} */
    this.reaction = null;
    /** @type {Shape} */
    this.face = Object.fromEntries(ARKIT.map((k) => [k, 0]));
    this.lastFrame = performance.now();
    this.nextBlink = performance.now() + 1500;
    this.blinkAt = -1;
    this.doubleBlink = false;
    /** Where the eyes look (degrees, + is the coach's left / up). */
    this.gazeGoal = { yaw: 0, pitch: 0 };
    this.gaze = { yaw: 0, pitch: 0 };
    this.nextSaccade = performance.now() + 1200;
    /** Head pose in radians (pitch + is nose up, yaw + to its left, roll + toward its right shoulder), and a vertical bounce. */
    this.head = { pitch: 0, yaw: 0, roll: 0, bounce: 0 };
    /** How much the coach is talking right now (0-1), smoothed. */
    this.energy = 0;
    this.energySlow = 0;
  }

  get audioCtx() {
    return this.ctx;
  }

  // ── Audio ───────────────────────────────────────────────────────────────

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
    this.voice = voice;
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
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") this.headaudio?.start();
      else this.headaudio?.stop();
    });
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
    const start = Math.max(now + latencyMs, this.cursor);
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
    this.onChunk(pcm, start);
    const chunk = this.upsampler.process(pcm).slice();
    this.player.port.postMessage({ type: "audioData", data: chunk.buffer }, [chunk.buffer]);
  }

  /**
   * A chunk of the voice was queued to play at `startMs` (audio clock), for
   * renderers that analyse the voice themselves.
   * @param {Int16Array} pcm @param {number} startMs
   */
  onChunk(pcm, startMs) {}

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

  /** Talking, or in the short gap between two lines. */
  get talking() {
    return this.playing || performance.now() - this.stoppedAt < 500;
  }

  /**
   * Advance the face by one frame: the 52 ARKit weights (in `this.face`),
   * the head pose (`this.head`) and gaze (`this.gaze`).
   */
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
    this.energy += (speaking - this.energy) * (1 - Math.exp(-dt / 60));
    this.energySlow += ((this.talking ? 1 : 0) - this.energySlow) * (1 - Math.exp(-dt / 400));

    // Mood and reactions; a smile flattens while the lips form words.
    if (this.reaction && t > this.reaction.until) this.reaction = null;
    const rest = MOODS[this.mood] ?? MOODS.neutral;
    const smileScale = 1 - 0.55 * Math.min(1, speaking * 1.5);
    let rw = 0;
    if (this.reaction) {
      const r = this.reaction;
      rw = Math.min(smooth((t - r.start) / 180), smooth((r.until - t) / 450));
    }
    for (const k in rest) target[k] += rest[k] * (1 - rw) * (k.startsWith("mouthSmile") ? smileScale : 1);
    if (this.reaction) {
      for (const k in this.reaction.shape) {
        const scale = k.startsWith("mouthSmile") || k === "jawOpen" ? smileScale : 1;
        target[k] += this.reaction.shape[k] * rw * scale;
      }
    }
    // Brows lift a little on stressed syllables.
    target.browInnerUp += this.energy * 0.12;
    target.browOuterUpLeft += this.energy * 0.08;
    target.browOuterUpRight += this.energy * 0.08;

    // Blinks every 2-6 s, sometimes doubled, and one right after a reaction starts.
    if (t >= this.nextBlink) {
      this.blinkAt = t;
      this.doubleBlink = Math.random() < 0.2;
      this.nextBlink = t + (this.doubleBlink ? 380 : 0) + 2000 + Math.random() * 4000;
    }
    const blinkAt = (/** @type {number} */ t0) => {
      const b = (t - t0) / 170;
      return b >= 0 && b <= 1 ? Math.sin(Math.PI * b) ** 0.7 : 0;
    };
    const blink = Math.max(blinkAt(this.blinkAt), this.doubleBlink ? blinkAt(this.blinkAt + 230) : 0);
    target.eyeBlinkLeft += blink;
    target.eyeBlinkRight += blink;

    this.updateGaze(t, dt);
    this.updateHead(t, dt);

    // Ease toward the target: quick for the mouth, slower for the rest.
    const fast = 1 - Math.exp(-dt / 30);
    const slow = 1 - Math.exp(-dt / 110);
    for (const k of ARKIT) {
      const goal = clamp(target[k]);
      const rate = k.startsWith("eyeBlink") ? 1 : k.startsWith("mouth") || k.startsWith("jaw") || k === "tongueOut" ? fast : slow;
      this.face[k] += (goal - this.face[k]) * rate;
    }
    return this.face;
  }

  /** @param {number} t @param {number} dt */
  updateGaze(t, dt) {
    if (t >= this.nextSaccade) {
      const thinking = this.mode === "Thinking";
      const idle = this.mode === "Idle" && !this.talking;
      if (thinking) this.gazeGoal = { yaw: 10 + Math.random() * 6, pitch: 9 + Math.random() * 4 };
      else if (idle && Math.random() < 0.25) this.gazeGoal = { yaw: (Math.random() - 0.5) * 22, pitch: (Math.random() - 0.3) * 8 };
      else this.gazeGoal = { yaw: (Math.random() - 0.5) * 4, pitch: (Math.random() - 0.5) * 3 };
      this.nextSaccade = t + (thinking ? 900 + Math.random() * 600 : 900 + Math.random() * 2400);
    }
    // Saccades are fast.
    const k = 1 - Math.exp(-dt / 35);
    this.gaze.yaw += (this.gazeGoal.yaw - this.gaze.yaw) * k;
    this.gaze.pitch += (this.gazeGoal.pitch - this.gaze.pitch) * k;
  }

  /**
   * Head motion: a slow idle drift, small nods with the speech, and a pose
   * per mode and reaction.
   * @param {number} t @param {number} dt
   */
  updateHead(t, dt) {
    const s = t / 1000;
    const goal = { pitch: 0, yaw: 0, roll: 0, bounce: 0 };
    const idle = 1 - 0.5 * this.energySlow;
    goal.pitch += wobble(s, 1) * 0.035 * idle;
    goal.yaw += wobble(s, 2) * 0.06 * idle;
    goal.roll += wobble(s, 3) * 0.03 * idle;
    // Speech: small nods with the syllables and a slow sway.
    goal.pitch += -this.energy * 0.05 + Math.sin(s * 2.1) * 0.02 * this.energySlow;
    goal.yaw += Math.sin(s * 0.9 + 1) * 0.05 * this.energySlow;
    goal.roll += Math.sin(s * 0.7 + 2) * 0.025 * this.energySlow;
    // Gaze pulls the head a little.
    goal.yaw += (this.gaze.yaw * Math.PI) / 180 * 0.3;
    goal.pitch += (this.gaze.pitch * Math.PI) / 180 * 0.25;
    if (this.mode === "Listening") {
      goal.roll += 0.07;
      goal.pitch -= 0.02;
    } else if (this.mode === "Thinking") {
      goal.roll -= 0.09;
      goal.pitch += 0.06;
    }
    const r = this.reaction;
    if (r) {
      const age = (t - r.start) / 1000;
      const fade = smooth((r.until - t) / 400);
      switch (r.name) {
        case "correct":
          // Two quick nods.
          goal.pitch -= Math.max(0, Math.sin(age * 11)) * 0.12 * Math.max(0, 1 - age / 0.9) * fade;
          goal.bounce += Math.max(0, Math.sin(age * 11)) * 0.02 * Math.max(0, 1 - age / 0.9);
          break;
        case "celebrate":
          goal.bounce += Math.abs(Math.sin(age * 9)) * 0.05 * Math.max(0, 1 - age / 1.6);
          goal.roll += Math.sin(age * 6) * 0.06 * Math.max(0, 1 - age / 1.6);
          break;
        case "incorrect":
          // A slow, sympathetic tilt.
          goal.roll -= 0.1 * fade * smooth(age / 0.4);
          goal.pitch -= 0.04 * fade * smooth(age / 0.4);
          break;
        case "partial":
          goal.roll += 0.08 * fade * smooth(age / 0.3);
          break;
        case "clue":
        case "hint":
          goal.roll += 0.06 * fade * smooth(age / 0.3);
          goal.pitch -= 0.05 * fade * smooth(age / 0.3);
          break;
        case "skipped":
          goal.roll += Math.sin(age * 5) * 0.06 * Math.max(0, 1 - age / 1.2);
          goal.bounce += 0.015 * fade * smooth(age / 0.2);
          break;
        case "surprised":
          goal.pitch += 0.08 * fade;
          goal.bounce += 0.01 * fade;
          break;
      }
    }
    const k = 1 - Math.exp(-dt / 160);
    const kb = 1 - Math.exp(-dt / 60);
    this.head.pitch += (goal.pitch - this.head.pitch) * k;
    this.head.yaw += (goal.yaw - this.head.yaw) * k;
    this.head.roll += (goal.roll - this.head.roll) * k;
    this.head.bounce += (goal.bounce - this.head.bounce) * kb;
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
    const t = performance.now();
    if (shape) this.reaction = { name, shape, start: t, until: t + ms };
  }

  /** @param {string} name @param {number} [dur] seconds */
  gesture(name, dur = 2) {
    const map = /** @type {Record<string, string>} */ ({ thumbup: "correct", handup: "celebrate", index: "clue", shrug: "skipped", ok: "correct" });
    this.showReaction(map[name] ?? name, dur * 1000);
  }

  /** @param {string} emoji */
  express(emoji) {
    const name = EMOJI[emoji];
    if (name) this.showReaction(name, 2000);
  }

  listen() {
    this.mode = "Listening";
    this.modeSince = performance.now();
    this.reaction = null;
    this.gazeGoal = { yaw: 0, pitch: 0 };
  }

  think() {
    this.mode = "Thinking";
    this.modeSince = performance.now();
    this.showReaction("think", 2500);
    this.nextSaccade = 0;
  }

  /** @param {string | null | undefined} verdict @param {{ streak?: number }} [ctx] */
  react(verdict, ctx = {}) {
    this.mode = "Idle";
    this.modeSince = performance.now();
    this.gazeGoal = { yaw: 0, pitch: 0 };
    switch (verdict) {
      case "correct":
        this.setMood("happy");
        this.showReaction((ctx.streak ?? 0) >= 3 ? "celebrate" : "correct", 2400);
        break;
      case "partial":
      case "incorrect":
      case "clue":
      case "hint":
      case "skipped":
        this.setMood("neutral");
        this.showReaction(verdict, 2200);
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
