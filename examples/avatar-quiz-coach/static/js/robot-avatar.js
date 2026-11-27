// @ts-check
/**
 * Pollen Robotics' robots as coaches: Reachy Mini, Reachy 2 and Microduck.
 * Each robot's model and joints live in robot-bodies.js; this file gives them
 * a voice, speech motion and reactions, all expressed as one head pose in
 * Reachy Mini's SDK frame (head position and orientation, two antennas and
 * the body's yaw) that each body maps onto its own joints.
 *
 * Mouths come from the same lip-sync timing as the other coaches (server
 * visemes or HeadAudio, see face-driver.js). Reachy Mini and Reachy 2 have
 * none, so they talk the way the real robots do, with their heads: open
 * vowels lift the head, lip closures and pauses let it settle, and the
 * antennas flick on syllables. Microduck opens its beak. Reactions replay
 * emotions recorded on the real Reachy Mini (Pollen's emotions library, baked
 * by scripts/bake_reachy_moves.py), on whichever robot is on screen.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { BODIES } from "./robot-bodies.js";
import { FaceDriver, mouthParams, smooth, wobble } from "./face-driver.js";
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
    /** Speech motion: a springy "jaw" the head rides on, and the mouth opening. */
    this.talk = { lift: 0, v: 0, ant: 0, antV: 0, lastOpen: 0, open: 0 };
    // Head motion on a spring trails the syllables by ~90 ms (measured), so
    // it runs that much ahead: server cues are scheduled earlier, and with
    // HeadAudio the voice waits that much longer. A robot with a real mouth
    // (Microduck's beak) sets its own lead.
    this.leadMs = coach.leadMs ?? 90;
    this.headaudioDelay += this.leadMs / 1000;
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
    this.stage = new Stage(this.container, { focusY: cam.focusY, frameH: cam.frameH, frameW: cam.frameW, fov: 24 });
    this.stage.scene.add(body.group);
    body.light?.(this.stage);
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

    // Base pose: the face driver's head (idle drift, gaze, listening and
    // thinking poses), in the SDK frame (x forward, y left, z up; pitch + is
    // nose down there).
    const H = this.head;
    _e.set(H.roll, -H.pitch, H.yaw * 1.2, "ZYX");
    this.base.q.setFromEuler(_e);
    this.base.p.set(0, 0, 0.002 * wobble(t, 7) + H.bounce * 0.15);
    const idleAnt = 0.12 * wobble(t * 0.8, 11);
    this.base.ant = [-0.18 + idleAnt, 0.18 + idleAnt * 0.8];
    this.base.yaw = H.yaw * 0.6 + 0.04 * wobble(t * 0.5, 5);

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

    // Speech: a springy lift driven by the mouth opening (the robot's
    // "jaw"), so the head rises on vowels and settles on closures and
    // pauses; the antennas flick on syllable onsets.
    const m = mouthParams(w);
    const T = this.talk;
    T.open = m.open;
    const drive = Math.min(1, m.open * 2.6);
    T.v += ((drive - T.lift) * 900 - T.v * 36) * sdt;
    T.lift += T.v * sdt;
    const onset = Math.max(0, drive - T.lastOpen);
    T.lastOpen = drive;
    T.antV += (-T.ant * 140 - T.antV * 8 + onset * 160) * sdt;
    T.ant += T.antV * sdt;
    const talkW = Math.min(1, this.energySlow * 1.2);
    const k = this.body?.talkScale ?? 1;
    this.pose.p.z += T.lift * 0.009 * k;
    this.pose.p.x += T.lift * 0.003 * k;
    _e.set(Math.sin(t * 1.7) * 0.05 * talkW, (-T.lift * 0.2 + 0.03 * talkW) * k, Math.sin(t * 0.9) * 0.09 * talkW, "ZYX");
    this.pose.q.multiply(_q.setFromEuler(_e));
    this.pose.ant = [this.pose.ant[0] - T.ant * 0.5, this.pose.ant[1] + T.ant * 0.5];

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
