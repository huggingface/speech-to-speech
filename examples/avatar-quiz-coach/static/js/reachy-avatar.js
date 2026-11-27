// @ts-check
/**
 * Reachy Mini as the coach: Pollen Robotics' open-source desk robot, drawn
 * from its Blender model with the Space rig (reachy-rig.js), which poses the
 * head on its Stewart platform, the antennas and the body like the SDK does.
 *
 * Reachy Mini has no mouth, so it talks the way the real robot does in its
 * conversation app: the head moves with the speech. The syllables come from
 * the same lip-sync timing as the other coaches (server visemes or
 * HeadAudio, see face-driver.js): open vowels lift the head a few
 * millimetres, lip closures and pauses let it settle, and the antennas
 * flick on stressed syllables. Reactions replay emotions recorded on the
 * real robot (Pollen's emotions library, baked by
 * scripts/bake_reachy_moves.py).
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { Reachy } from "./reachy-rig.js";
import { FaceDriver, mouthParams, smooth, wobble } from "./face-driver.js";
import { Stage } from "./stage.js";

/** @typedef {{ name: string, fps: number, frames: number[][] }} Move */
/** @typedef {{ p: THREE.Vector3, q: THREE.Quaternion, ant: number[], yaw: number }} Pose */

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
  skipped: ["resigned1"],
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

export class ReachyAvatar extends FaceDriver {
  /** @param {HTMLElement} container @param {any} coach */
  constructor(container, coach) {
    super(container, coach);
    /** @type {Stage | null} */
    this.stage = null;
    /** @type {any} */
    this.robot = null;
    /** @type {Record<string, Move>} */
    this.moves = {};
    /** @type {{ move: Move, t: number, from: Pose, rate: number, fadeIn: number } | null} */
    this.playingMove = null;
    this.lastMove = "";
    this.pose = newPose();
    this.base = newPose();
    this.movePose = newPose();
    /** Speech motion state: a springy "jaw" that the head rides on. */
    this.talk = { lift: 0, v: 0, ant: 0, antV: 0, lastOpen: 0 };
    // The head's spring trails the syllables by ~90 ms (measured), so the
    // motion runs that much ahead: server cues are scheduled earlier, and
    // with HeadAudio the voice waits that much longer.
    this.leadMs = 90;
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
    const [gltf, moves] = await Promise.all([
      loader.loadAsync(this.coach.asset, (e) => onProgress?.({ lengthComputable: true, loaded: e.loaded * 0.95, total: e.total || 1 })),
      fetch(this.coach.moves)
        .then((r) => (r.ok ? r.json() : { moves: [] }))
        .catch(() => ({ moves: [] })),
    ]);
    for (const m of moves.moves) this.moves[m.name] = m;
    draco.dispose();

    const robot = new Reachy(gltf.scene);
    this.robot = robot;
    robot.setPose(this.pose);
    robot.group.updateWorldMatrix(true, true);
    this.hideInternals(robot);
    this.simplifyMaterials(robot);
    // Scale to ~3 units tall, base on the floor of the frame, facing the camera.
    const box = new THREE.Box3().setFromObject(robot.group);
    const s = 3 / (box.max.y - box.min.y);
    robot.group.scale.setScalar(s);
    robot.group.position.y = -box.min.y * s - 2.15;
    const cam = this.coach.camera ?? {};
    this.stage = new Stage(this.container, { focusY: cam.focusY ?? 0.1, frameH: cam.frameH ?? 2.9, frameW: cam.frameW ?? 2.5, fov: 24 });
    this.stage.scene.add(robot.group);
    // A white shell: softer key, more wrap from the sky, so it isn't blown out.
    this.stage.renderer.toneMappingExposure = 0.82;
    this.stage.lights.key.intensity = 1.7;
    this.stage.lights.hemi.intensity = 0.9;
    this.stage.scene.environmentIntensity = 0.45;
    this.stage.updaters.push((dt) => this.update(dt));
    onProgress?.({ lengthComputable: true, loaded: 100, total: 100 });
    // Wake up: from the resting pose (head down, antennas folded) into a hello.
    const sleep = this.moves.sleep;
    if (sleep) this.moveFrame(sleep, 0, this.pose);
    this.playFor("hello", 1.4);
  }

  /**
   * The model is the full CAD assembly. The motors inside the body and the
   * electronics inside the closed head can never be seen (only the lenses,
   * what sits right behind them and the camera between them show through
   * the face), and they are most of its 460 draws and 260k triangles.
   * @param {any} robot
   */
  hideInternals(robot) {
    const box = new THREE.Box3();
    const lenses = robot.meshes.filter((/** @type {any} */ m) => m.material?.name === "Glass_Eyes");
    const centers = lenses.map((/** @type {any} */ l) => box.setFromObject(l).getCenter(new THREE.Vector3()));
    const gap = centers.length === 2 ? centers[0].distanceTo(centers[1]) : 0;
    // The camera bridge between the eyes shows too.
    if (gap) centers.push(centers[0].clone().lerp(centers[1], 0.5));
    /** @type {any[]} */
    this.hidden = [];
    const c = new THREE.Vector3();
    for (const p of robot.parts) {
      const m = p.node;
      if (!m.isMesh || lenses.includes(m)) continue;
      let hide = p.id === "motors";
      if (p.id === "headCore" && gap) {
        box.setFromObject(m).getCenter(c);
        hide = Math.min(...centers.map((/** @type {THREE.Vector3} */ e) => e.distanceTo(c))) > 0.3 * gap;
      }
      if (hide) {
        m.visible = false;
        this.hidden.push(m);
      }
    }
  }

  /**
   * Transmission (the glass eyes) costs an extra render pass a frame: plain
   * glossy black glass looks the same at this size.
   * @param {any} robot
   */
  simplifyMaterials(robot) {
    const glass = new THREE.MeshPhysicalMaterial({ color: 0x050508, roughness: 0.05, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.02, envMapIntensity: 1.6 });
    for (const m of robot.meshes) {
      const mats = Array.isArray(m.material) ? m.material : [m.material];
      const out = mats.map((/** @type {any} */ mat) => (mat.transmission > 0 || mat.name === "Glass_Eyes" ? glass : mat));
      m.material = Array.isArray(m.material) ? out : out[0];
    }
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

    // Recorded move on top, faded in over 0.35 s from wherever the robot
    // was, and out over its last 0.5 s.
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
    const drive = Math.min(1, m.open * 2.6);
    T.v += ((drive - T.lift) * 900 - T.v * 36) * sdt;
    T.lift += T.v * sdt;
    const onset = Math.max(0, drive - T.lastOpen);
    T.lastOpen = drive;
    T.antV += (-T.ant * 140 - T.antV * 8 + onset * 160) * sdt;
    T.ant += T.antV * sdt;
    const talkW = Math.min(1, this.energySlow * 1.2);
    this.pose.p.z += T.lift * 0.009;
    this.pose.p.x += T.lift * 0.003;
    _e.set(Math.sin(t * 1.7) * 0.05 * talkW, -T.lift * 0.2 + 0.03 * talkW, Math.sin(t * 0.9) * 0.09 * talkW, "ZYX");
    this.pose.q.multiply(_q.setFromEuler(_e));
    this.pose.ant = [this.pose.ant[0] - T.ant * 0.5, this.pose.ant[1] + T.ant * 0.5];

    this.robot.setPose(this.pose);
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
