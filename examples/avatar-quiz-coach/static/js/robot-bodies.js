// @ts-check
/**
 * The robots robot-avatar.js can drive, each loaded from Pollen Robotics'
 * open models and posed from one head pose in Reachy Mini's SDK frame
 * (x forward, y left, z up; metres and radians):
 *
 * - Reachy Mini: its Blender model on the Space rig (reachy-rig.js), which
 *   solves the Stewart platform under the head; the pose maps one to one.
 * - Reachy 2: its Blender export, whose parts are merged per material, so
 *   the head (and the antennas on it) are re-hung on a pivot at the Orbita
 *   neck joint. The neck only turns, so head height becomes a nod. The
 *   mobile base turns for the body's yaw.
 * - Microduck: its MuJoCo model (mjcf.js), standing up and idling with the
 *   recorded clip; the pose drives its neck and head joints, and the mouth
 *   opening drives its beak, hinged at the jaw servo.
 */

import * as THREE from "three";
import { Reachy } from "./reachy-rig.js";
import { loadMJCF, poseMJCF, setPackLoader } from "./mjcf.js";
import { smooth } from "./face-driver.js";

/** @typedef {{ p: THREE.Vector3, q: THREE.Quaternion, ant: number[], yaw: number }} Pose */
/** @typedef {{ lift: number, open: number }} Talk */
/**
 * @typedef {{
 *   group: THREE.Object3D,
 *   camera: { focusY: number, frameH: number, frameW: number },
 *   apply: (pose: Pose, talk: Talk, dt: number) => void,
 *   light?: (stage: import("./stage.js").Stage) => void,
 *   talkScale?: number,
 * }} Body
 */

const glassEyes = () =>
  new THREE.MeshPhysicalMaterial({ color: 0x050508, roughness: 0.05, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.02, envMapIntensity: 1.6 });

/**
 * Transmission (the glass eyes) costs an extra render pass a frame: plain
 * glossy black glass looks the same at this size.
 * @param {THREE.Object3D} root
 */
function simplifyMaterials(root) {
  const glass = glassEyes();
  root.traverse((/** @type {any} */ m) => {
    if (!m.isMesh) return;
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    const out = mats.map((/** @type {any} */ mat) => (mat.transmission > 0 || mat.name === "Glass_Eyes" ? glass : mat));
    m.material = Array.isArray(m.material) ? out : out[0];
  });
}

/** White shells: a softer key and more wrap from the sky, so they aren't blown out. @param {import("./stage.js").Stage} stage */
function whiteShellLight(stage) {
  stage.renderer.toneMappingExposure = 0.82;
  stage.lights.key.intensity = 1.7;
  stage.lights.hemi.intensity = 0.9;
  stage.scene.environmentIntensity = 0.45;
}

/** SDK frame (x forward, y left, z up) to the scene's (robot facing +z, +x its left, y up). */
const toScene = (/** @type {THREE.Quaternion} */ q, /** @type {THREE.Quaternion} */ out) => out.set(q.y, q.z, q.x, q.w);
const _q = new THREE.Quaternion();
const _e = new THREE.Euler(0, 0, 0, "ZYX");
const X = new THREE.Vector3(1, 0, 0);
const Z = new THREE.Vector3(0, 0, 1);

// ── Reachy Mini ───────────────────────────────────────────────────────────

/**
 * The model is the full CAD assembly. The motors inside the body and the
 * electronics inside the closed head can never be seen (only the lenses,
 * what sits right behind them and the camera between them show through the
 * face), and they are most of its 460 draws and 260k triangles.
 * @param {any} robot
 */
function hideMiniInternals(robot) {
  const box = new THREE.Box3();
  const lenses = robot.meshes.filter((/** @type {any} */ m) => m.material?.name === "Glass_Eyes");
  const centers = lenses.map((/** @type {any} */ l) => box.setFromObject(l).getCenter(new THREE.Vector3()));
  const gap = centers.length === 2 ? centers[0].distanceTo(centers[1]) : 0;
  // The camera bridge between the eyes shows too.
  if (gap) centers.push(centers[0].clone().lerp(centers[1], 0.5));
  const hidden = [];
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
      hidden.push(m);
    }
  }
  return hidden;
}

/** @param {any} loader @param {any} coach @param {(f: number) => void} progress @returns {Promise<Body>} */
async function loadMini(loader, coach, progress) {
  const gltf = await loader.loadAsync(coach.asset, (/** @type {ProgressEvent} */ e) => progress(e.total ? e.loaded / e.total : 0));
  const robot = new Reachy(gltf.scene);
  robot.setPose({ p: new THREE.Vector3(), q: new THREE.Quaternion(), ant: [0, 0], yaw: 0 });
  robot.group.updateWorldMatrix(true, true);
  const hidden = hideMiniInternals(robot);
  simplifyMaterials(robot.group);
  // ~3 units tall, base at the bottom of the frame, facing the camera.
  const box = new THREE.Box3().setFromObject(robot.group);
  const s = 3 / (box.max.y - box.min.y);
  robot.group.scale.setScalar(s);
  robot.group.position.y = -box.min.y * s - 2.15;
  robot.group.userData.hidden = hidden;
  return {
    group: robot.group,
    camera: { focusY: 0.45, frameH: 4.9, frameW: 3.6 },
    apply: (pose) => robot.setPose(pose),
    light: whiteShellLight,
  };
}

// ── Reachy 2 ──────────────────────────────────────────────────────────────

/** @param {any} loader @param {any} coach @param {(f: number) => void} progress @returns {Promise<Body>} */
async function loadReachy2(loader, coach, progress) {
  const gltf = await loader.loadAsync(coach.asset, (/** @type {ProgressEvent} */ e) => progress(e.total ? e.loaded / e.total : 0));
  const scene = gltf.scene;
  simplifyMaterials(scene);
  // The cube of its grasp demo stays out.
  scene.traverse((/** @type {any} */ o) => {
    if (o.isMesh && o.material?.name === "Grasp_Cube") o.visible = false;
  });
  const root = new THREE.Group();
  root.add(scene);
  root.updateWorldMatrix(true, true);

  // Head: the camera head group and the two antennas on it.
  const top = (/** @type {string} */ prefix) => scene.children.filter((o) => o.name.startsWith(prefix));
  const headParts = top("DD2090_4CAM");
  const antennas = top("Moumousse").sort((a, b) => new THREE.Box3().setFromObject(a).getCenter(new THREE.Vector3()).x - new THREE.Box3().setFromObject(b).getCenter(new THREE.Vector3()).x);
  const headBox = new THREE.Box3();
  for (const o of headParts) headBox.expandByObject(o);
  const neck = new THREE.Group();
  // Orbita's centre sits just under the head shell, behind the eyes.
  neck.position.set((headBox.min.x + headBox.max.x) / 2, headBox.min.y - 0.012, (headBox.min.z + headBox.max.z) / 2 - 0.02);
  root.add(neck);
  neck.updateWorldMatrix(true, false);
  for (const o of headParts) neck.attach(o);
  /** @type {THREE.Group[]} right (scene -x) first, then left */
  const antPivots = antennas.map((a) => {
    let base = new THREE.Box3().setFromObject(a);
    a.traverse((/** @type {any} */ m) => {
      if (m.isMesh && m.material?.name === "Shell") base = new THREE.Box3().setFromObject(m);
    });
    const pivot = new THREE.Group();
    neck.add(pivot);
    pivot.position.copy(neck.worldToLocal(new THREE.Vector3((base.min.x + base.max.x) / 2, base.min.y, (base.min.z + base.max.z) / 2)));
    pivot.updateWorldMatrix(true, false);
    pivot.attach(a);
    return pivot;
  });

  return {
    group: root,
    camera: { focusY: 1.22, frameH: 0.82, frameW: 0.66 },
    apply: (pose) => {
      toScene(pose.q, neck.quaternion);
      // The neck only turns: head height becomes a nod (up when it rises).
      neck.quaternion.premultiply(_q.setFromAxisAngle(X, -pose.p.z * 5));
      // Antennas turn in the face's plane, like Reachy Mini's (whose angles
      // run the other way round).
      if (antPivots[0]) antPivots[0].quaternion.setFromAxisAngle(Z, -pose.ant[0]);
      if (antPivots[1]) antPivots[1].quaternion.setFromAxisAngle(Z, -pose.ant[1]);
      root.rotation.y = pose.yaw * 0.5;
    },
    light: whiteShellLight,
  };
}

// ── Microduck ─────────────────────────────────────────────────────────────

/**
 * Its beak is fixed to the head in the model: hinge the lower beak at the
 * jaw servo, about the head's lateral axis (from the keynote's scene).
 * @param {any} duck
 */
function hingeBeak(duck) {
  const jaw = new THREE.Group();
  const head = duck.userData.bodies.jaw_soft.children[0];
  jaw.position.set(0.003, 0, -0.018);
  for (const n of ["jaw", "jaw_soft"]) {
    const m = head.children.find((/** @type {any} */ c) => c.name === n);
    if (!m) continue;
    m.position.sub(jaw.position);
    jaw.add(m);
  }
  head.add(jaw);
  return jaw;
}

/** Joint directions, so that + is: neck up, nose down, turn left, tilt right, beak open. */
const DUCK = { neck: 1, pitch: 1, yaw: 1, roll: -1, jaw: 1 };
/** Three-quarter view: the body turned toward the quiz, the head back to the student. */
const DUCK_TURN = 0.55;

/** @param {any} loader @param {any} coach @param {(f: number) => void} progress @returns {Promise<Body>} */
async function loadDuck(loader, coach, progress) {
  const dir = coach.asset;
  setPackLoader((/** @type {string} */ url) => loader.loadAsync(url, (/** @type {ProgressEvent} */ e) => progress(e.total ? e.loaded / e.total : 0)));
  const [duck, clip] = await Promise.all([
    loadMJCF(`${dir}/robot.xml`, {
      meshDir: `${dir}/assets`,
      keep: ["jaw", "jaw_soft"],
      recolor: (/** @type {string} */ n) => (n === "xl330_material" ? { color: 0x161618, roughness: 0.4 } : null), // black XL330 servos
    }),
    fetch(`${dir}/clip.json`).then((r) => r.json()),
  ]);
  const jaw = hingeBeak(duck);
  const B = duck.userData.bodies;
  const joint = (/** @type {string} */ body) => B[body].children[0];
  const neck = joint("neck");
  const headPitch = joint("neck_pitch");
  const headYaw = joint("yaw_roll_motion");
  const headRoll = joint("jaw_soft");

  // Stand facing the camera (MJCF's +x forward), measured on its idle pose.
  const root = new THREE.Group();
  root.add(duck);
  duck.rotation.y = -Math.PI / 2;
  const idle0 = clip.segments.idle[0];
  poseMJCF(duck, clip, idle0);
  root.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(root);
  const h = box.max.y - box.min.y;
  // ~3 units tall like the other robots, feet at the bottom of the frame.
  const s = 3 / h;
  root.scale.setScalar(s);
  root.position.y = -box.min.y * s - 1.6;

  // Clock: stand up (the clip's first segment, which ends with a quack),
  // then hold that stance. The clip's idle looks around on its own, so from
  // there the coach's pose drives the head.
  let t = 0;
  const clipJaw = (/** @type {number} */ x) => {
    const k = Math.max(0, Math.min(clip.jaw.length - 1, x * clip.fps));
    const i = Math.floor(k);
    return clip.jaw[i] + (clip.jaw[Math.min(clip.jaw.length - 1, i + 1)] - clip.jaw[i]) * (k - i);
  };
  const turn = (/** @type {THREE.Object3D} */ j, /** @type {number} */ a) => j.quaternion.setFromAxisAngle(Z, a);
  return {
    group: root,
    camera: { focusY: 0.25, frameH: 4, frameW: 3.3 },
    talkScale: 0.5,
    apply: (pose, talk, dt) => {
      t += dt;
      const ct = Math.min(t, idle0);
      poseMJCF(duck, clip, ct);
      const up = smooth(t / idle0);
      _e.setFromQuaternion(pose.q, "ZYX");
      turn(neck, up * DUCK.neck * (pose.p.z * 9 - _e.y * 0.25));
      turn(headPitch, up * DUCK.pitch * _e.y * 0.85);
      turn(headYaw, up * DUCK.yaw * (_e.z - DUCK_TURN * 0.85));
      turn(headRoll, up * DUCK.roll * _e.x);
      jaw.rotation.y = 0.55 * clipJaw(ct) + DUCK.jaw * Math.min(1, talk.open * 2.4) * 0.5;
      root.rotation.y = DUCK_TURN + pose.yaw * 0.5 * up;
    },
  };
}

/** @type {Record<string, (loader: any, coach: any, progress: (f: number) => void) => Promise<Body>>} */
export const BODIES = { mini: loadMini, reachy2: loadReachy2, duck: loadDuck };
