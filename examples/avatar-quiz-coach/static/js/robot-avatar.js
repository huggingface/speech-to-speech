// @ts-check
/**
 * Pollen Robotics' robots as coaches: Reachy Mini, Reachy 2 and Microduck.
 * Each robot's model and joints live in robot-bodies.js; this file gives them
 * a voice, speech motion and reactions, all expressed as one head pose in
 * Reachy Mini's SDK frame (head position and orientation, two antennas and
 * the body's yaw) that each body maps onto its own joints.
 *
 * They talk the way the real robots do: the head wobbles with the voice,
 * using the official SDK's speech wobble (head-wobbler.js); Microduck also
 * opens its beak with the lip-sync timing the other coaches use (server
 * visemes or HeadAudio, see face-driver.js). Between lines each robot holds
 * a pose per conversational state (idle breathing, listening, thinking,
 * speaking, after Pollen's reachy-mini-3d-voice Space), and reactions replay
 * emotions recorded on the real Reachy Mini (Pollen's emotions library,
 * baked by scripts/bake_reachy_moves.py), on whichever robot is on screen.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { BODIES } from "./robot-bodies.js";
import { FaceDriver, mouthParams, smooth } from "./face-driver.js";
import { HeadWobbler } from "./head-wobbler.js";
import { duckGreet, duckWaks, QUACK_RATE } from "./quack.js";
import { Stage } from "./stage.js";

/** @typedef {{ name: string, fps: number, frames: number[][] }} Move */
/** @typedef {import("./robot-bodies.js").Pose} Pose */

/** Seconds of silence before a sound starts. @param {AudioBuffer} buf */
function soundLead(buf) {
  const x = buf.getChannelData(0);
  const i = x.findIndex((v) => Math.abs(v) > 0.01);
  return i < 0 ? 0 : i / buf.sampleRate;
}

/** Quiz moments whose emotion also plays its recorded sound (not the frequent listening and thinking). */
const SOUND_KINDS = new Set(["hello", "correct", "streak", "celebrate", "partial", "incorrect", "skipped", "surprised"]);
const SOUND_DIR = "static/avatars/reachy-sounds";

/** Emotions per quiz moment; one is picked at random, avoiding the last one. */
const REACTION_MOVES = /** @type {Record<string, string[]>} */ ({
  hello: ["welcoming1", "welcoming2"],
  correct: ["success2", "proud2", "yes1", "cheerful1"],
  streak: ["success1", "enthusiastic1"],
  celebrate: ["dance1", "enthusiastic1", "laughing1"],
  partial: ["inquiring1", "understanding1"],
  incorrect: ["uncertain1", "oops2"],
  clue: ["helpful1"],
  hint: ["helpful1"],
  skipped: ["resigned1", "understanding1"],
  think: ["thoughtful1", "thoughtful2"],
  listen: ["attentive1", "attentive2"],
  surprised: ["amazed1"],
});

const TAU = 2 * Math.PI;
/** Antennas open and close together, 15 degrees at 0.5 Hz, as the robot breathes. @param {number} t */
const breathing = (t) => {
  const s = ((15 * Math.PI) / 180) * Math.sin(TAU * 0.5 * t);
  return [-s, s];
};
/**
 * Base pose per conversational state, in the SDK frame (metres, radians;
 * pitch + is nose down), after robot/motion.js in Pollen's
 * reachy-mini-3d-voice Space.
 * @type {Record<string, (t: number) => { x?: number, z: number, roll: number, pitch: number, yaw: number, bodyYaw?: number }>}
 */
const STATES = {
  idle: (t) => ({
    z: 0.004 * Math.sin((TAU * t) / 4.6),
    pitch: 0.02 * Math.sin((TAU * t) / 5.3 + 1.2),
    yaw: 0.08 * Math.sin(0.31 * t) + 0.05 * Math.sin(0.13 * t + 2.1),
    roll: 0.02 * Math.sin(0.17 * t + 0.7),
    bodyYaw: 0.05 * Math.sin(0.09 * t),
  }),
  listening: (t) => ({ z: 0.008 + 0.003 * Math.sin((TAU * t) / 3.8), pitch: -0.04, yaw: 0.03 * Math.sin(0.23 * t), roll: 0.015 * Math.sin(0.19 * t + 1) }),
  thinking: (t) => ({ z: 0.006, pitch: -0.12, yaw: 0.22 + 0.02 * Math.sin(1.1 * t), roll: 0.06 }),
  speaking: () => ({ z: 0.004, pitch: -0.02, yaw: 0, roll: 0 }),
};
/** Seconds to blend from one state's pose to the next. */
const STATE_BLEND_S = 0.7;

const newPose = () => /** @type {Pose} */ ({ p: new THREE.Vector3(), q: new THREE.Quaternion(), ant: [0, 0], yaw: 0 });
const _e = new THREE.Euler(0, 0, 0, "ZYX");
const _q = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();

/** @param {Pose} out @param {Pose} a @param {Pose} b @param {number} k */
function blendPose(out, a, b, k) {
  out.p.lerpVectors(a.p, b.p, k);
  out.q.slerpQuaternions(a.q, b.q, k);
  out.ant = [a.ant[0] + (b.ant[0] - a.ant[0]) * k, a.ant[1] + (b.ant[1] - a.ant[1]) * k];
  out.yaw = a.yaw + (b.yaw - a.yaw) * k;
  return out;
}

/** @param {Pose} dst @param {Pose} src */
function copyPose(dst, src) {
  dst.p.copy(src.p);
  dst.q.copy(src.q);
  dst.ant = [...src.ant];
  dst.yaw = src.yaw;
  return dst;
}

export class RobotAvatar extends FaceDriver {
  /** @param {HTMLElement} container @param {any} coach */
  constructor(container, coach) {
    super(container, coach);
    /** @type {Stage | null} */
    this.stage = null;
    /** @type {import("./robot-bodies.js").Body | null} */
    this.body = null;
    /** @type {Record<string, Move>} */
    this.moves = {};
    /** @type {{ move: Move, t: number, age: number, from: Pose, rate: number, fadeIn: number } | null} */
    this.playingMove = null;
    this.lastMove = "";
    this.pose = newPose();
    this.base = newPose();
    this.movePose = newPose();
    /** The mouth opening, for a robot with a beak. */
    this.talk = { open: 0 };
    /** Weight of each state's pose, blended over STATE_BLEND_S. @type {Record<string, number>} */
    this.stateW = { idle: 1, listening: 0, thinking: 0, speaking: 0 };
    this.wobbler = new HeadWobbler(this.inRate);
    /** Decoded move sounds, by move name. @type {Map<string, Promise<AudioBuffer | null>>} */
    this.sounds = new Map();
    /** The same, once decoded. @type {Map<string, AudioBuffer>} */
    this.decoded = new Map();
    /** Seconds each move holds still before it starts moving. @type {Map<string, number>} */
    this.leads = new Map();
    /** The body's clock, and its quacks still to come (Microduck's stand-up). */
    this.bodyT = 0;
    /** @type {{ at: number, buf: AudioBuffer }[]} */
    this.quackQueue = [];
    this.wob = { pitch: 0, yaw: 0, roll: 0, x: 0, y: 0, z: 0 };
    if (coach.mouth === "beak") {
      // The beak trails the sound by its smoothing; it runs that far ahead
      // (measured): server cues are scheduled earlier, and with HeadAudio the
      // voice waits that much longer.
      this.leadMs = coach.leadMs ?? 60;
      this.headaudioDelay += this.leadMs / 1000;
    } else {
      // No mouth to line up with HeadAudio's visemes: no voice delay.
      this.leadMs = 0;
      this.headaudioDelay = 0;
    }
  }

  /** The wobble follows the voice as it is heard. @param {Int16Array} pcm @param {number} startMs */
  onChunk(pcm, startMs) {
    // Plus the voice delay that lines HeadAudio's visemes up with the beak
    // (set for this response in feed(); the AudioParam reads back late).
    this.wobbler.feed(pcm, startMs + (this.serverVisemes ? 0 : this.headaudioDelay) * 1000);
  }

  interrupt() {
    super.interrupt();
    this.wobbler.reset();
  }

  /**
   * @param {{ viseme: number, start_s: number, end_s: number }} cue @param {number} t
   */
  scheduleCue(cue, t) {
    super.scheduleCue(cue, t - this.leadMs);
  }

  /** @param {(ev: { lengthComputable: boolean, loaded: number, total: number }) => void} [onProgress] */
  async init(onProgress) {
    await this.initAudio();
    // Fetch and decode the emotion sounds while the model loads, so each one
    // is ready to start with its move.
    if (this.coach.robot !== "duck") for (const kind of SOUND_KINDS) for (const n of REACTION_MOVES[kind] ?? []) this.moveSound(n);
    const draco = new DRACOLoader().setDecoderPath("vendor/three/examples/jsm/libs/draco/gltf/");
    const loader = new GLTFLoader().setDRACOLoader(draco);
    const load = BODIES[this.coach.robot];
    if (!load) throw new Error(`unknown robot ${this.coach.robot}`);
    const [body, moves] = await Promise.all([
      load(loader, this.coach, (/** @type {number} */ f) => onProgress?.({ lengthComputable: true, loaded: f * 95, total: 100 })),
      fetch(this.coach.moves)
        .then((r) => (r.ok ? r.json() : { moves: [] }))
        .catch(() => ({ moves: [] })),
    ]);
    for (const m of moves.moves) this.moves[m.name] = m;
    draco.dispose();
    this.body = body;
    const cam = body.camera;
    this.stage = new Stage(this.container, { focusY: cam.focusY, frameH: cam.frameH, frameW: cam.frameW, tiltDeg: cam.tiltDeg ?? 0, fov: 24 });
    this.stage.scene.add(body.group);
    body.light?.(this.stage);
    if (body.table) this.stage.addTable(body.group, body.table);
    this.stage.updaters.push((dt) => this.update(dt));
    onProgress?.({ lengthComputable: true, loaded: 100, total: 100 });
    // Wake up: from the resting pose (head down, antennas folded) into a hello.
    const sleep = this.moves.sleep;
    if (sleep) this.moveFrame(sleep, 0, this.pose);
    this.playFor("hello", 1.4);
    // Microduck quacks as it stands up, each time its beak opens.
    if (body.quacks?.length && this.ctx) {
      const ctx = this.ctx;
      const waks = duckWaks(body.quacks.length);
      this.quackQueue = body.quacks.map((at, i) => {
        const buf = ctx.createBuffer(1, waks[i].length, QUACK_RATE);
        buf.copyToChannel(/** @type {Float32Array<ArrayBuffer>} */ (waks[i]), 0);
        return { at, buf };
      });
      // Sound blocked at load: quack hello on the first click instead.
      if (ctx.state !== "running") this.whenAudible(() => this.quackQueue.length === 0 && this.playPcm(duckGreet(), 1));
    }
  }

  // ── Sounds ──────────────────────────────────────────────────────────────

  /** The sound recorded with a move, decoded once. @param {string} name */
  moveSound(name) {
    let sound = this.sounds.get(name);
    if (!sound) {
      const ctx = this.ctx;
      sound = fetch(`${SOUND_DIR}/${name}.ogg`)
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`${r.status}`))))
        .then((buf) => (ctx ? ctx.decodeAudioData(buf) : null))
        .then((buf) => {
          if (buf) this.decoded.set(name, buf);
          return buf;
        })
        .catch(() => null);
      this.sounds.set(name, sound);
    }
    return sound;
  }

  /** Seconds a move holds still before it starts moving. @param {string} name */
  moveLead(name) {
    let lead = this.leads.get(name);
    if (lead === undefined) {
      const m = this.moves[name];
      const f = m?.frames ?? [];
      const i = f.findIndex((fr) => fr.reduce((s, v, k) => s + Math.abs(v - f[0][k]), 0) > 0.02);
      lead = m && i > 0 ? i / m.fps : 0;
      this.leads.set(name, lead);
    }
    return lead;
  }

  /** @param {AudioBuffer | null} buf @param {number} gain @param {number} [offset] seconds into the sound */
  playBuffer(buf, gain, offset = 0) {
    const ctx = this.ctx;
    if (!buf || !ctx || ctx.state !== "running") return;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain();
    g.gain.value = gain;
    src.connect(g).connect(ctx.destination);
    src.start(0, Math.min(offset, buf.duration));
  }

  /** @param {Float32Array} pcm 24 kHz mono @param {number} gain */
  playPcm(pcm, gain) {
    const ctx = this.ctx;
    if (!ctx) return;
    const buf = ctx.createBuffer(1, pcm.length, QUACK_RATE);
    buf.copyToChannel(/** @type {Float32Array<ArrayBuffer>} */ (pcm), 0);
    this.playBuffer(buf, gain);
  }

  /**
   * Run `fn` once the page may play sound: now, as soon as the audio starts,
   * or at the first click or key press (browsers block sound until then).
   * @param {() => void} fn
   */
  whenAudible(fn) {
    const ctx = this.ctx;
    if (!ctx) return;
    if (ctx.state === "running") return fn();
    let done = false;
    const run = () => {
      if (done) return;
      done = true;
      ctx.removeEventListener("statechange", onState);
      window.removeEventListener("pointerdown", onGesture, true);
      window.removeEventListener("keydown", onGesture, true);
      fn();
    };
    const onState = () => ctx.state === "running" && run();
    const onGesture = () => this.resume().then(run);
    ctx.addEventListener("statechange", onState);
    window.addEventListener("pointerdown", onGesture, true);
    window.addEventListener("keydown", onGesture, true);
  }

  // ── Moves ───────────────────────────────────────────────────────────────

  /** @param {string} name @param {number} [fadeIn] seconds @param {number} [rate] @param {number} [from] seconds into the move */
  play(name, fadeIn = 0.35, rate = 1, from = 0) {
    const move = this.moves[name];
    if (!move) return false;
    this.playingMove = { move, t: from, age: 0, from: copyPose(newPose(), this.pose), rate, fadeIn };
    this.lastMove = name;
    return true;
  }

  /** @param {string} kind @param {number} [fadeIn] */
  playFor(kind, fadeIn) {
    const list = (REACTION_MOVES[kind] ?? []).filter((n) => this.moves[n]);
    if (!list.length) return;
    const pool = list.length > 1 ? list.filter((n) => n !== this.lastMove) : list;
    const name = pool[Math.floor(Math.random() * pool.length)];
    // Recorded moves and their sounds open with a still, silent stretch that
    // the real robot just waits through; here the robot already glides into
    // the move. Skip both, so the motion and the sound start together.
    const withSound = SOUND_KINDS.has(kind) && this.coach.robot !== "duck";
    const buf = withSound ? this.decoded.get(name) : undefined;
    const skip = Math.max(0, this.moveLead(name) - 0.08);
    const soundSkip = buf ? Math.max(0, soundLead(buf) - 0.08) : 0;
    this.play(name, fadeIn, 1, skip);
    // The Reachys' emotions come with their own sounds (Microduck has its quacks).
    if (!withSound) return;
    if (kind !== "hello") {
      // Under the coach's voice: a little quieter, and only if sound can play now.
      if (buf) this.playBuffer(buf, 0.6, soundSkip);
      else this.moveSound(name);
      return;
    }
    const startedAt = performance.now();
    this.whenAudible(() => {
      // Blocked until a click: wave hello again, with its sound this time.
      if (performance.now() - startedAt > 1500) this.play(name, 0.6, 1, skip);
      this.moveSound(name).then((b) => this.playBuffer(b, 1, b ? Math.max(0, soundLead(b) - 0.08) : 0));
    });
  }

  /** @param {Move} m @param {number} t @param {Pose} out */
  moveFrame(m, t, out) {
    const f = m.frames;
    const k = Math.min(t * m.fps, f.length - 1);
    const i = Math.floor(k);
    const a = f[i];
    const b = f[Math.min(i + 1, f.length - 1)];
    const u = k - i;
    const l = (/** @type {number} */ j) => a[j] + (b[j] - a[j]) * u;
    out.p.set(l(0), l(1), l(2));
    out.q.slerpQuaternions(_q.fromArray(a, 3), _q2.fromArray(b, 3), u);
    out.ant = [l(7), l(8)];
    out.yaw = l(9);
    return out;
  }

  // ── Per frame ───────────────────────────────────────────────────────────

  /** @param {number} dt */
  update(dt) {
    const w = this.frame();
    const t = performance.now() / 1000;
    const sdt = dt / 1000;

    // Base pose: each conversational state's pose, blended as the state
    // changes.
    const state = this.talking ? "speaking" : this.mode === "Thinking" ? "thinking" : this.mode === "Listening" ? "listening" : "idle";
    let total = 0;
    for (const k in this.stateW) {
      this.stateW[k] = Math.max(0, Math.min(1, this.stateW[k] + (k === state ? 1 : -1) * (sdt / STATE_BLEND_S)));
      total += this.stateW[k];
    }
    let x = 0, z = 0, roll = 0, pitch = 0, yaw = 0, bodyYaw = 0;
    for (const k in this.stateW) {
      const w = this.stateW[k] / (total || 1);
      if (!w) continue;
      const p = STATES[k](t);
      x += (p.x ?? 0) * w;
      z += p.z * w;
      roll += p.roll * w;
      pitch += p.pitch * w;
      yaw += p.yaw * w;
      bodyYaw += (p.bodyYaw ?? 0) * w;
    }
    _e.set(roll, pitch, yaw, "ZYX");
    this.base.q.setFromEuler(_e);
    this.base.p.set(x, 0, z);
    this.base.ant = breathing(t);
    this.base.yaw = bodyYaw;

    // Recorded move on top, faded in from wherever the robot was, and out
    // over its last 0.5 s.
    const pm = this.playingMove;
    if (pm) {
      pm.t += sdt * pm.rate;
      pm.age += sdt;
      const dur = pm.move.frames.length / pm.move.fps;
      this.moveFrame(pm.move, pm.t, this.movePose);
      const fadeIn = smooth(pm.age / pm.fadeIn);
      const fadeOut = smooth((dur - pm.t) / 0.5);
      blendPose(this.pose, this.base, this.movePose, fadeOut);
      if (fadeIn < 1) blendPose(this.pose, pm.from, this.pose, fadeIn);
      if (pm.t >= dur) this.playingMove = null;
    } else {
      copyPose(this.pose, this.base);
    }

    // Speech: the SDK's wobble on top of whatever the robot is doing.
    const o = this.wobbler.at((this.ctx?.currentTime ?? 0) * 1000, this.wob);
    const k = this.body?.talkScale ?? 1;
    this.pose.p.x += o.x * k;
    this.pose.p.y += o.y * k;
    this.pose.p.z += o.z * k;
    _e.set(o.roll * k, o.pitch * k, o.yaw * k, "ZYX");
    this.pose.q.multiply(_q.setFromEuler(_e));

    // A beak follows the mouth shapes.
    const T = this.talk;
    T.open = mouthParams(w).open;

    this.body?.apply(this.pose, T, sdt);
    // Microduck's stand-up quacks, a hair early for the audio output's delay.
    this.bodyT += sdt;
    const early = (this.ctx?.outputLatency || this.ctx?.baseLatency || 0.02) + 0.02;
    while (this.quackQueue.length && this.quackQueue[0].at <= this.bodyT + early) {
      const q = /** @type {{ at: number, buf: AudioBuffer }} */ (this.quackQueue.shift());
      this.playBuffer(q.buf, 1);
    }
  }

  // ── Body language ───────────────────────────────────────────────────────

  listen() {
    super.listen();
    if (!this.playingMove && Math.random() < 0.5) this.playFor("listen");
  }

  think() {
    super.think();
    this.playFor("think");
  }

  /** @param {string | null | undefined} verdict @param {{ streak?: number }} [ctx] */
  react(verdict, ctx = {}) {
    super.react(verdict, ctx);
    if (verdict === "correct") this.playFor((ctx.streak ?? 0) >= 3 ? "streak" : "correct");
    else if (verdict) this.playFor(verdict);
  }

  celebrate() {
    super.celebrate();
    this.playFor("celebrate");
  }

  /** @param {string} name @param {number} [dur] */
  gesture(name, dur = 2) {
    if (!this.play(name)) super.gesture(name, dur);
  }
}
