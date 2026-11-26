// @ts-check
/**
 * The coach's face: a TalkingHead (three.js) avatar with real-time lip-sync.
 *
 * Audio arrives from the speech backend as 24 kHz PCM16 and is played through
 * TalkingHead's streaming worklet. Mouth shapes come from one of two sources:
 *
 * 1. Server visemes: when the backend runs with `--enable_visemes`, it sends
 *    phoneme-derived viseme cues (Microsoft IDs 0-21, seconds from the start of
 *    the response audio) ahead of the audio. They are mapped to Oculus visemes
 *    and scheduled on a playback cursor that follows the player through
 *    buffer underruns (a slow TTS leaves gaps; the mouth waits for the sound).
 * 2. HeadAudio: otherwise an AudioWorklet classifies the outgoing audio into
 *    Oculus visemes in the browser; the voice is delayed 60 ms to match.
 */

import { TalkingHead } from "talkinghead";
import { HeadAudio } from "headaudio";

const VOWELS = new Set(["aa", "E", "I", "O", "U"]);
// Peak strength per mouth shape. Real speakers fully form only lip closures
// (p, b, m, f, v) and open vowels; tongue sounds barely show on the lips, so
// driving every phoneme at full strength made the mouth flap.
const VISEME_LEVEL = /** @type {Record<string, number>} */ ({
  PP: 0.9, FF: 0.8, aa: 0.7, O: 0.7, U: 0.65, E: 0.6, I: 0.55,
  CH: 0.5, SS: 0.4, TH: 0.4, RR: 0.35, DD: 0.3, nn: 0.3, kk: 0.3,
});
// Vowels also open the jaw; the visemes alone barely part the lips on some rigs.
const JAW_LEVEL = /** @type {Record<string, number>} */ ({ aa: 0.3, O: 0.22, E: 0.15, U: 0.12, I: 0.1, CH: 0.05 });

// Microsoft viseme IDs (as emitted by the backend's STV stage) to Oculus names.
const MS_TO_OCULUS = [
  "sil", "aa", "aa", "O", "E", "RR", "I", "U", "O", "aa", "O",
  "aa", "kk", "RR", "nn", "SS", "CH", "TH", "FF", "DD", "kk", "PP",
];

// `?lipsync=headaudio` ignores server visemes (for comparing the two sources).
const FORCE_HEADAUDIO = new URLSearchParams(location.search).get("lipsync") === "headaudio";

/** Coaches: TalkingHead sample avatars (free for non-commercial use) and their tuning. */
export const COACHES = {
  ada: {
    name: "Ada",
    voice: "Serena",
    // How strongly this rig's blendshapes read: Avaturn's are strong.
    lipGain: 0.85,
    jawGain: 0.4,
    avatar: {
      url: "static/avatars/avaturn.glb",
      body: "F",
      avatarMood: "happy",
      retarget: {
        Hips: { y: 0.03 }, Spine: { y: 0.02 }, Spine1: { y: 0.02, z: 0.01 }, Spine2: { y: 0.02, z: 0.01 },
        Neck: { z: 0.02, y: 0.01 }, Head: { z: 0.02 }, LeftShoulder: { rx: -0.5 }, RightShoulder: { rx: -0.5 },
        scaleToHipsLevel: 1.0,
      },
      // Eyes open and on the student: the stock "look down" reads as sleepy.
      baseline: { headRotateX: -0.05, eyeBlinkLeft: 0, eyeBlinkRight: 0, eyesLookDown: 0 },
    },
  },
  max: {
    name: "Max",
    voice: "Aiden",
    lipGain: 1.35,
    jawGain: 1,
    avatar: {
      url: "static/avatars/avatarsdk.glb",
      body: "M",
      avatarMood: "happy",
      retarget: {
        Neck: { z: -0.01, rx: -0.15 }, Neck1: { z: -0.01, rx: -0.15 }, Neck2: { z: -0.01, rx: -0.15 },
        LeftShoulder: { rz: -0.3 }, RightShoulder: { rz: 0.3 }, scaleToEyesLevel: 1.0, origin: { y: -0.1 },
      },
      baseline: { headRotateX: -0.04, eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05, eyesLookDown: 0 },
    },
  },
};

/**
 * The phoneme recognizer emits each phoneme as a 20-40 ms blip with a
 * silence cue between neighbours, so played as-is the mouth snaps shut
 * between sounds of the same word. Drop the short in-word silences and hold
 * each shape until the next one starts (at most 300 ms); real pauses of
 * 150 ms or more still close the mouth.
 * @param {{ viseme: number, start_s: number, end_s: number }[]} cues
 */
export function coarticulate(cues) {
  const kept = [...cues]
    .sort((a, b) => a.start_s - b.start_s)
    .filter((c) => c.viseme !== 0 || c.end_s - c.start_s >= 0.15)
    .map((c) => ({ ...c }));
  for (let i = 0; i < kept.length; i++) {
    const c = kept[i];
    if (c.viseme === 0) continue;
    const next = kept[i + 1];
    const until = next ? next.start_s : c.start_s + 0.12;
    c.end_s = Math.max(c.end_s, Math.min(until, c.start_s + 0.3));
  }
  return kept.filter((c) => c.viseme !== 0);
}

class Upsampler {
  /** @param {number} inRate @param {number} outRate */
  constructor(inRate, outRate) {
    this.step = inRate / outRate;
    this.reset();
  }
  reset() {
    this.pos = 0;
    this.prev = 0;
  }
  /** @param {Int16Array} input */
  process(input) {
    const n = input.length;
    if (!n) return new Int16Array(0);
    const out = new Int16Array(Math.ceil((n - this.pos) / this.step) + 1);
    let p = this.pos;
    let k = 0;
    while (p < n) {
      const i = Math.floor(p);
      const a = i === 0 ? this.prev : input[i - 1];
      out[k++] = a + (input[i] - a) * (p - i);
      p += this.step;
    }
    this.pos = p - n;
    this.prev = input[n - 1];
    return out.subarray(0, k);
  }
}

export class Avatar extends EventTarget {
  /** @param {HTMLElement} container @param {keyof typeof COACHES} [coach] */
  constructor(container, coach = "max") {
    super();
    this.container = container;
    this.coach = COACHES[coach] || COACHES.max;
    /** @type {any} */
    this.head = null;
    /** @type {any} */
    this.headaudio = null;
    this.responseId = "";
    this.responseSamples = 0; // samples (input rate) fed for this response
    this.cursor = 0; // animClock time (ms) when the queued audio runs out
    /** @type {{ viseme: number, start_s: number, end_s: number }[]} */
    this.pendingCues = [];
    this.serverVisemes = false;
    this.lipsyncSource = "headaudio";
    this.playing = false;
    this.inRate = 24000;
    /** @type {Upsampler | null} */
    this.upsampler = null;
    /** @type {DelayNode | null} */
    this.voiceDelay = null;
    /** @type {number[]} */
    this.lineTimers = [];
  }

  /** @param {(ev: ProgressEvent) => void} [onProgress] */
  async init(onProgress) {
    const head = new TalkingHead(this.container, {
      ttsEndpoint: "N/A",
      lipsyncModules: [],
      // Close head-and-shoulders framing: lips readable, hands still in shot.
      cameraView: "upper",
      cameraDistance: -0.7,
      cameraX: 0,
      cameraY: 0.05,
      cameraRotateEnable: false,
      cameraPanEnable: false,
      cameraZoomEnable: false,
      modelFPS: 60,
      lightAmbientColor: 0xffffff,
      lightAmbientIntensity: 0.6,
      lightDirectColor: 0xfff1dc,
      lightDirectIntensity: 18,
      lightDirectPhi: 0.7,
      lightDirectTheta: 2.2,
      lightSpotIntensity: 0,
      avatarIdleEyeContact: 0.45,
      avatarIdleHeadMove: 0.4,
      avatarSpeakingEyeContact: 0.8,
      avatarSpeakingHeadMove: 0.6,
    });
    this.head = head;
    await head.showAvatar(this.coach.avatar, onProgress ?? null);
    this.polishMaterials();
    this.upsampler = new Upsampler(this.inRate, head.audioCtx.sampleRate);
    await this.initStream();
    await this.initHeadAudio();
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        head.start();
        this.headaudio?.start();
      } else {
        this.headaudio?.stop();
        head.stop();
      }
    });
  }

  /** The AvatarSDK sample outfit carries the vendor's logo: plain fabric instead. */
  polishMaterials() {
    if (this.coach !== COACHES.max) return;
    this.head.armature?.traverse((/** @type {any} */ obj) => {
      if (!obj.isMesh || !obj.name.startsWith("outfit_")) return;
      for (const m of Array.isArray(obj.material) ? obj.material : [obj.material]) {
        m.map = null;
        m.color.set(obj.name === "outfit_top" ? 0x2d3a4f : 0x1d2230);
        m.needsUpdate = true;
      }
    });
  }

  get audioCtx() {
    return this.head?.audioCtx ?? null;
  }

  async initStream() {
    const head = this.head;
    await head.streamStart(
      { sampleRate: head.audioCtx.sampleRate, lipsyncType: "visemes", waitForAudioChunks: true },
      () => {
        this.playing = true;
        this.dispatchEvent(new CustomEvent("playback", { detail: { playing: true } }));
      },
      () => {
        this.playing = false;
        this.dispatchEvent(new CustomEvent("playback", { detail: { playing: false } }));
      },
    );
    // TalkingHead routes the stream into both its stream gain and its analyser
    // (which feeds the speech gain); both reach the speakers. Keep one path so
    // the voice is not doubled, and keep the analyser path HeadAudio listens to.
    try {
      head.streamWorkletNode.disconnect(head.audioStreamGainNode);
    } catch {}
    // HeadAudio's visemes trail the sound by ~60 ms (measured), so the audible
    // voice goes through a delay that is switched on only when HeadAudio drives.
    this.voiceDelay = head.audioCtx.createDelay(0.5);
    head.audioSpeechGainNode.disconnect(head.audioReverbNode);
    head.audioSpeechGainNode.connect(this.voiceDelay);
    this.voiceDelay.connect(head.audioReverbNode);
  }

  /**
   * Queue one mouth shape on TalkingHead's animation clock (same envelope as
   * its own stream lip-sync, with per-shape strength).
   * @param {{ viseme: number, start_s: number, end_s: number }} cue @param {number} t
   */
  scheduleCue(cue, t) {
    const v = MS_TO_OCULUS[cue.viseme];
    if (!v || v === "sil") return;
    const d = Math.max(40, (cue.end_s - cue.start_s) * 1000);
    // Short vowels open the jaw less than long ones.
    const length = VOWELS.has(v) ? Math.min(1, Math.max(0.6, d / 120)) : 1;
    const level = Math.min(1, (VISEME_LEVEL[v] ?? 0.4) * this.coach.lipGain * length);
    /** @type {Record<string, (number | null)[]>} */
    const vs = { ["viseme_" + v]: [null, level, 0] };
    const jaw = (JAW_LEVEL[v] ?? 0) * this.coach.jawGain * length;
    if (jaw > 0) vs.jawOpen = [null, jaw, 0];
    this.head.animQueue.push({ template: { name: "viseme" }, ts: [t - (2 * d) / 3, t + d / 2, t + d + d / 2], vs });
  }

  async initHeadAudio() {
    const head = this.head;
    try {
      await head.audioCtx.audioWorklet.addModule("vendor/headaudio/headworklet.min.mjs");
      const ha = new HeadAudio(head.audioCtx, { visemeMinMs: 110 });
      await ha.loadModel("vendor/headaudio/model-en-mixed.bin");
      head.audioSpeechGainNode.connect(ha);
      // HeadAudio's shapes are faint on some rigs: apply the coach's lip gain
      // and open the jaw with the vowels, as for server visemes.
      /** @type {Record<string, number>} */
      const current = {};
      const set = (/** @type {string} */ key, /** @type {number} */ value) => {
        const mt = head.mtAvatar?.[key];
        if (mt) Object.assign(mt, { newvalue: value, needsUpdate: true });
      };
      ha.onvalue = (/** @type {string} */ key, /** @type {number} */ value) => {
        if (this.serverVisemes) return;
        current[key.replace("viseme_", "")] = value;
        set(key, Math.min(1, value * this.coach.lipGain * 1.2));
        let jaw = 0;
        for (const [v, x] of Object.entries(current)) jaw += (JAW_LEVEL[v] ?? 0) * x;
        set("jawOpen", Math.min(0.45, jaw * this.coach.jawGain * 1.5));
      };
      head.opt.update = ha.update.bind(ha);
      this.headaudio = ha;
    } catch (err) {
      console.warn("[avatar] HeadAudio unavailable, lips will only move with server visemes", err);
    }
  }

  resume() {
    this.head?.start();
    const ctx = this.audioCtx;
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
  }

  /** @param {string} responseId @param {{ viseme: number, start_s: number, end_s: number }[]} cues */
  addVisemes(responseId, cues) {
    if (FORCE_HEADAUDIO) return;
    this.beginResponse(responseId);
    this.pendingCues.push(...coarticulate(cues));
    if (!this.serverVisemes) {
      this.serverVisemes = true;
      this.lipsyncSource = "server";
      this.head?.resetLips?.();
    }
  }

  /** @param {string} responseId @param {Int16Array} pcm */
  feed(responseId, pcm) {
    const head = this.head;
    if (!head || !this.upsampler) return;
    this.beginResponse(responseId);
    if (!this.serverVisemes) this.lipsyncSource = "headaudio";
    // Server cues arrive before their audio, so the first chunk decides the source.
    if (this.responseSamples === 0 && this.voiceDelay) this.voiceDelay.delayTime.value = this.serverVisemes ? 0 : 0.06;

    // The player is a FIFO that starts a chunk as soon as it arrives when its
    // buffer is empty, so a chunk starts at max(now, end of the previous one).
    const ctx = head.audioCtx;
    const latencyMs = ((ctx.baseLatency || 0) + (ctx.outputLatency || 0)) * 1000 || 20;
    const durMs = (pcm.length / this.inRate) * 1000;
    const r0 = (this.responseSamples / this.inRate) * 1000;
    const start = Math.max(head.animClock + latencyMs, this.cursor);
    this.cursor = start + durMs;
    if (this.responseSamples === 0) {
      // Tell the page when this line becomes audible (it may queue behind the
      // previous line), so captions appear with the voice, not before it.
      const id = responseId;
      this.lineTimers.push(
        window.setTimeout(() => this.dispatchEvent(new CustomEvent("line-start", { detail: { responseId: id } })), start - head.animClock),
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
    head.streamAudio({ audio: this.upsampler.process(pcm) });
  }

  /** The reply's audio is complete; let the player end when it drains. */
  endResponse() {
    this.head?.streamNotifyEnd();
  }

  /** Barge-in or skip: drop queued audio and close the mouth now. */
  interrupt() {
    this.lineTimers.forEach(clearTimeout);
    this.lineTimers = [];
    this.head?.streamInterrupt();
    this.pendingCues = [];
    this.responseId = "";
    this.cursor = 0;
  }

  // ── Body language ───────────────────────────────────────────────────────

  /** @param {string} name */
  setMood(name) {
    try {
      this.head?.setMood(name);
    } catch {}
  }

  /** @param {string} name @param {number} [dur] */
  gesture(name, dur = 2.5) {
    try {
      this.head?.playGesture(name, dur);
    } catch {}
  }

  /** @param {string} emoji */
  express(emoji) {
    try {
      this.head?.speakEmoji(emoji);
    } catch {}
  }

  listen() {
    this.head?.lookAtCamera(600);
  }

  think() {
    this.express("🤔");
  }

  /** @param {string | null | undefined} verdict @param {{ streak?: number }} [ctx] */
  react(verdict, ctx = {}) {
    switch (verdict) {
      case "correct":
        this.setMood("happy");
        if ((ctx.streak ?? 0) >= 3) this.gesture("handup", 2);
        else this.gesture("thumbup", 2);
        break;
      case "partial":
        this.setMood("neutral");
        this.gesture("index", 1.8);
        break;
      case "incorrect":
        this.setMood("neutral");
        this.express("🙂");
        break;
      case "clue":
      case "hint":
        this.setMood("neutral");
        this.gesture("index", 1.6);
        break;
      case "skipped":
        this.setMood("neutral");
        this.gesture("shrug", 1.8);
        break;
      default:
        this.setMood("neutral");
    }
  }

  celebrate() {
    this.setMood("happy");
    this.gesture("handup", 3);
  }
}
