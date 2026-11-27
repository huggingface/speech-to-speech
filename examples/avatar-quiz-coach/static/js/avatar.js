// @ts-check
/**
 * Max's face: a TalkingHead (three.js) avatar with real-time lip-sync.
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
import { FORCE_HEADAUDIO, JAW_LEVEL, Upsampler, coarticulate, cueShape } from "./lipsync.js";
import { COACHES } from "./coaches.js";

export class Avatar extends EventTarget {
  /** @param {HTMLElement} container @param {typeof COACHES.max} coach */
  constructor(container, coach) {
    super();
    this.container = container;
    this.coach = coach;
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
    /** @type {AnalyserNode | null} */
    this.outTap = null;
    /** @type {Float32Array | null} */
    this.outBuf = null;
    /** Audible start (ms, avatar clock) of each response's audio. @type {Map<string, number>} */
    this.lineStarts = new Map();
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
      lightAmbientColor: 0xf4f7fb,
      lightAmbientIntensity: 1.15,
      lightDirectColor: 0xfff5e8,
      lightDirectIntensity: 9.5,
      lightDirectPhi: 0.85,
      lightDirectTheta: 2.15,
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
    this.outTap = head.audioCtx.createAnalyser();
    this.outTap.fftSize = 1024;
    this.voiceDelay.connect(this.outTap);
  }

  /**
   * Queue one mouth shape on TalkingHead's animation clock (same envelope as
   * its own stream lip-sync, with per-shape strength).
   * @param {{ viseme: number, start_s: number, end_s: number }} cue @param {number} t
   */
  scheduleCue(cue, t) {
    const s = cueShape(cue, this.coach);
    if (!s) return;
    /** @type {Record<string, (number | null)[]>} */
    const vs = { ["viseme_" + s.v]: [null, s.level, 0] };
    if (s.jaw > 0) vs.jawOpen = [null, s.jaw, 0];
    this.head.animQueue.push({ template: { name: "viseme" }, ts: [t - (2 * s.d) / 3, t + s.d / 2, t + s.d + s.d / 2], vs });
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
    this.lineStarts.set(responseId, this.lineStarts.get(responseId) ?? start);
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
    return Math.max(0, this.head.animClock - start);
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
