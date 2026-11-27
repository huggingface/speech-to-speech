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
import { Stage } from "./stage.js";

/** @typedef {{ name: string, fps: number, frames: number[][] }} Move */
/** @typedef {import("./robot-bodies.js").Pose} Pose */

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
    /** @type {{ move: Move, t: number, from: Pose, rate: number, fadeIn: number } | null} */
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
  }

  // ── Moves ───────────────────────────────────────────────────────────────

  /** @param {string} name @param {number} [fadeIn] seconds @param {number} [rate] */
  play(name, fadeIn = 0.35, rate = 1) {
    const move = this.moves[name];
    if (!move) return false;
    this.playingMove = { move, t: 0, from: copyPose(newPose(), this.pose), rate, fadeIn };
    this.lastMove = name;
    return true;
  }

  /** @param {string} kind @param {number} [fadeIn] */
  playFor(kind, fadeIn) {
    const list = (REACTION_MOVES[kind] ?? []).filter((n) => this.moves[n]);
    if (!list.length) return;
    const pool = list.length > 1 ? list.filter((n) => n !== this.lastMove) : list;
    this.play(pool[Math.floor(Math.random() * pool.length)], fadeIn);
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
      const dur = pm.move.frames.length / pm.move.fps;
      this.moveFrame(pm.move, pm.t, this.movePose);
      const fadeIn = smooth(pm.t / pm.fadeIn);
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
