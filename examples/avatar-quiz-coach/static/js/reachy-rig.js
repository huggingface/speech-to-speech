// Reachy Mini rig (pose, Stewart-platform legs, antennas, exploded view), unmodified apart from this header.
// From Pollen Robotics' reachy-mini-yo-its-your-boy Space (https://huggingface.co/spaces/pollen-robotics/reachy-mini-yo-its-your-boy),
// as redistributed in https://github.com/FabienDanieau/reachy_mini_intro (assets/robot.js). Apache-2.0.

// Reachy Mini rig, ported from the mobile app's ReachyModel.tsx without the
// exponential smoothing: a pose is applied exactly, so a frame only depends on
// the time it is rendered at.
import * as THREE from 'three';
import { clone as skeletonClone } from 'three/addons/utils/SkeletonUtils.js';

const MODEL_SCALE = 0.5;
const DISPLAY_YAW = -Math.PI;
const UNITS_PER_M = 1.7;
const HEAD_FIX = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
const HEAD_FIX_INV = HEAD_FIX.clone().invert();
const ANT_AXIS = new THREE.Vector3(0, 0, 1);
const ANT_SIGN = -1;

const BONE = { body: 'Core', head: 'Core.001', antL: 'Antenna.L.002', antR: 'Antenna.R.002' };
const norm = s => s.replace(/[^a-z0-9]/gi, '').toLowerCase();

// Sub-assembly offsets (robot frame, scene units; the robot is ~0.3 tall),
// start order (0 leads) and label pin (shift of the label anchor from the
// sub-assembly's centre). The stack comes apart along the vertical axis, top
// first, and every layer rises at least as far and as early as the one below
// it, so no two layers ever cross. The body shell lifts around the Stewart
// rods (both start together), which stay visible through its phantom look.
export const EXPLODE = {
  base: { off: new THREE.Vector3(0, 0, 0), order: 1, label: 'BASE' },
  motors: { off: new THREE.Vector3(0, 0.05, 0), order: 0.7, label: 'MOTORS' },
  neck: { off: new THREE.Vector3(0, 0.1, 0), order: 0.52, label: 'HEAD BASE\n6-DOF STEWART PLATFORM', pin: new THREE.Vector3(0, 0.125, 0) },
  bodyShell: { off: new THREE.Vector3(0, 0.15, 0), order: 0.5, label: 'BODY SHELL', pin: new THREE.Vector3(0, -0.035, 0) },
  headBot: { off: new THREE.Vector3(0, 0.185, 0), order: 0.4, label: '' },
  headCore: { off: new THREE.Vector3(0, 0.21, 0), order: 0.3, label: 'EYES + ELECTRONICS' },
  headFront: { off: new THREE.Vector3(0, 0.29, 0.05), order: 0.2, label: 'FACE' },
  headBack: { off: new THREE.Vector3(0, 0.37, -0.05), order: 0.12, label: 'HEAD SHELL' },
  headTop: { off: new THREE.Vector3(0, 0.44, 0), order: 0, label: 'MICROPHONES' },
  antennas: { off: new THREE.Vector3(0, 0.44, -0.05), order: 0.05, label: 'ANTENNAS' },
};
// Shell parts: the ones that turn into a phantom while the robot is exploded.
const SHELLS = ['coquetop', 'coquebot', 'shellhead'];

const LEG_IDS = ['A', 'B', 'C', 'D', 'E', 'F'];
const LEG_IK_ITERATIONS = 12;
const LEG_DOF = ['locked', 'hingeX', 'locked', 'free'];
const HINGE_AXIS = new THREE.Vector3(1, 0, 0);

function setupLeg(id, byName, head) {
  const chain = ['001', '002', '003', '004'].map(s => byName[norm(`Neck.${id}.${s}`)]);
  const ikTarget = byName[norm(`Neck.loc.IK.${id}`)];
  if (chain.some(b => !b) || !ikTarget) return null;
  const attach = ikTarget.getWorldPosition(new THREE.Vector3());
  const effector = new THREE.Object3D();
  chain[3].add(effector);
  effector.position.copy(chain[3].worldToLocal(attach.clone()));
  const target = new THREE.Object3D();
  head.add(target);
  target.position.copy(head.worldToLocal(attach.clone()));
  return { chain, restQ: chain.map(b => b.quaternion.clone()), effector, target };
}

const _jp = new THREE.Vector3();
const _ep = new THREE.Vector3();
const _tp = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _hw = new THREE.Vector3();
const _cx = new THREE.Vector3();
const _qr = new THREE.Quaternion();
const _qw = new THREE.Quaternion();
const _qp = new THREE.Quaternion();

function solveLeg(leg) {
  for (let i = 0; i < leg.chain.length; i++) leg.chain[i].quaternion.copy(leg.restQ[i]);
  leg.chain[0].updateWorldMatrix(false, true);
  leg.target.getWorldPosition(_tp);
  for (let it = 0; it < LEG_IK_ITERATIONS; it++) {
    for (let i = 0; i < leg.chain.length; i++) {
      const dof = LEG_DOF[i];
      if (dof === 'locked') continue;
      const joint = leg.chain[i];
      leg.effector.getWorldPosition(_ep);
      if (_ep.distanceToSquared(_tp) < 1e-8) return;
      joint.getWorldPosition(_jp);
      joint.getWorldQuaternion(_qw);
      if (dof === 'free') {
        _v1.subVectors(_ep, _jp).normalize();
        _v2.subVectors(_tp, _jp).normalize();
        _qr.setFromUnitVectors(_v1, _v2);
      } else {
        _hw.copy(HINGE_AXIS).applyQuaternion(_qw).normalize();
        _v1.subVectors(_ep, _jp).projectOnPlane(_hw);
        _v2.subVectors(_tp, _jp).projectOnPlane(_hw);
        if (_v1.lengthSq() < 1e-12 || _v2.lengthSq() < 1e-12) continue;
        _v1.normalize();
        _v2.normalize();
        _qr.setFromAxisAngle(_hw, Math.atan2(_cx.crossVectors(_v1, _v2).dot(_hw), _v1.dot(_v2)));
      }
      _qr.multiply(_qw);
      joint.parent?.getWorldQuaternion(_qp);
      joint.quaternion.copy(_qp.invert().multiply(_qr));
      joint.updateWorldMatrix(false, true);
    }
  }
}

// The real Reachy Mini's head limits (reachy_mini SDK), degrees and mm:
// - tilt: head pitch / roll within +-40 deg (docs/source/SDK/core-concept.md,
//   "Safety Limits"), enforced here as a cone on the total tilt so combined
//   pitch + roll stays reachable too;
// - rel / body: |head yaw - body yaw| <= 65 deg and |body yaw| <= 160 deg
//   (kinematics/analytical_kinematics.py, inverse_kinematics_safe
//   max_relative_yaw / max_body_yaw). Like the SDK, the body is dragged
//   along rather than the head held back;
// - x / y / z: head translation, in the body (Stewart platform) frame, within
//   the analytical IK's reach under the URDF motor limits (neutral
//   orientation: x -46..50, y +-47, z -50..23).
// Every limit saturates softly (tanh knee of width `knee`) so a clamped move
// never kinks; poses inside `limit - knee` are applied exactly (SLEEP, INIT).
export const HEAD_LIMITS = { tilt: 40, rel: 65, body: 160, x: [-38, 38], y: [-40, 40], z: [-50, 23] };
const KNEE = { tilt: 10, rel: 15, body: 20, xy: 8, z: 5 };
const DEG = Math.PI / 180;

function soft(v, lo, hi, k) {
  if (v > hi - k) return hi - k + k * Math.tanh((v - hi + k) / k);
  if (v < lo + k) return lo + k - k * Math.tanh((lo + k - v) / k);
  return v;
}

const _Z = new THREE.Vector3(0, 0, 1);
const _tw = new THREE.Quaternion();

// Heading / tilt split of a base-frame head quaternion: q = Rz(heading) * tilt,
// where tilt turns about a horizontal axis of the heading frame (swing-twist
// about the vertical).
export function splitHead(q, tilt) {
  _tw.set(0, 0, q.z, q.w);
  const n = Math.hypot(q.z, q.w);
  if (n < 1e-9) _tw.identity();
  else _tw.set(0, 0, q.z / n, q.w / n);
  const heading = 2 * Math.atan2(_tw.z, _tw.w);
  tilt?.copy(_tw.invert()).multiply(q);
  return heading;
}

const _tilt = new THREE.Quaternion();

// Base-frame pose -> what the rig shows, relative to the body, within
// HEAD_LIMITS: { yaw: body yaw, q, p: head orientation / translation in the
// body frame }.
export function limitHead(pose, out) {
  const L = HEAD_LIMITS;
  const heading = splitHead(pose.q, _tilt);
  const s = Math.hypot(_tilt.x, _tilt.y, _tilt.z);
  const angle = 2 * Math.atan2(s, Math.abs(_tilt.w));
  if (s > 1e-9) {
    const a = soft(angle / DEG, -1e9, L.tilt, KNEE.tilt) * DEG;
    const k = Math.sin(a / 2) / s;
    _tilt.set(_tilt.x * k, _tilt.y * k, _tilt.z * k, Math.sign(_tilt.w || 1) * Math.cos(a / 2));
  }
  const rel = Math.atan2(Math.sin(heading - pose.yaw), Math.cos(heading - pose.yaw));
  const relL = soft(rel / DEG, -L.rel, L.rel, KNEE.rel) * DEG;
  // The body follows the head past the relative limit, within its own range.
  out.yaw = soft((pose.yaw + rel - relL) / DEG, -L.body, L.body, KNEE.body) * DEG;
  out.q.setFromAxisAngle(_Z, relL).multiply(_tilt);
  out.p.copy(pose.p).applyAxisAngle(_Z, -out.yaw);
  out.p.set(
    soft(out.p.x * 1000, L.x[0], L.x[1], KNEE.xy) / 1000,
    soft(out.p.y * 1000, L.y[0], L.y[1], KNEE.xy) / 1000,
    soft(out.p.z * 1000, L.z[0], L.z[1], KNEE.z) / 1000,
  );
  return out;
}

// Head / body-shell clearance, the rig's stand-in for the SDK's head-torso
// self-collision constraint (kinematics/placo_kinematics.py,
// add_avoid_self_collisions_constraint): some poses inside HEAD_LIMITS still
// push the head's skirt through the shell (a lean + a turn, a big nod). The
// head is then translated out of contact, toward the yaw axis or up,
// whichever frees the contact sooner: under Placo's 1 cm ~ 1 deg task
// weights a translation is the cheapest fix and it keeps the orientation.
// The shell's inner wall is tabulated once in the model frame (distance from
// the yaw axis per heading and height, plus the rim height per heading) and
// the lower head shell is sampled into points. Depth is counted beyond the
// rest pose (where the model already touches the rim) and resolved with a
// Huber knee, so contact eases in and poses that don't touch (INIT, SLEEP)
// are left exactly as they are. The push is a continuous function of the
// pose (no per-frame state, no branch that flips between the two ways out),
// so a head sliding along the rim moves smoothly instead of trembling.
// Distances in mm: knee (Huber), soft (contact rounding), gap (overhang
// clearance), blend (width of the axis / up handover), far (search range).
const CLEAR = { knee: 0.5, soft: 0.5, gap: 3, blend: 6, far: 40, stall: 4, iterations: 16, headings: 180, dz: 1 };
const _clearCache = new WeakMap();
function shellClearance(source, model, meshes, hr) {
  if (_clearCache.has(source)) return _clearCache.get(source);
  const U = UNITS_PER_M / 1000;
  const toModel = model.matrixWorld.clone().invert();
  const shell = meshes.find(m => norm(m.name).startsWith('coquetop'));
  const v = new THREE.Vector3();
  const NA = CLEAR.headings;
  const binOf = (x, y) => ((((Math.atan2(y, x) / (2 * Math.PI)) * NA) % NA) + NA) % NA;
  const sp = shell.geometry.attributes.position;
  const sv = [];
  for (let k = 0; k < sp.count; k++) sv.push(new THREE.Vector3().fromBufferAttribute(sp, k).applyMatrix4(shell.matrixWorld).applyMatrix4(toModel));
  const z0 = Math.min(...sv.map(p => p.z));
  const zTop = Math.max(...sv.map(p => p.z));
  const DZ = CLEAR.dz * U;
  const NY = Math.ceil((zTop - z0) / DZ) + 1;
  // Slice every shell triangle with each row's plane and keep, per heading,
  // the nearest crossing to the axis (the inner wall).
  const wall = new Float32Array(NA * NY).fill(Infinity);
  const si = shell.geometry.index;
  const tri = si ? Array.from({ length: si.count }, (_, k) => si.getX(k)) : sv.map((_, k) => k);
  const cut = [];
  for (let f = 0; f < tri.length; f += 3) {
    const [A, B, C] = [sv[tri[f]], sv[tri[f + 1]], sv[tri[f + 2]]];
    const j0 = Math.max(0, Math.ceil((Math.min(A.z, B.z, C.z) - z0) / DZ));
    const j1 = Math.min(NY - 1, Math.floor((Math.max(A.z, B.z, C.z) - z0) / DZ));
    for (let j = j0; j <= j1; j++) {
      const z = z0 + j * DZ;
      cut.length = 0;
      for (const [P, Q] of [[A, B], [B, C], [C, A]]) {
        if ((P.z - z) * (Q.z - z) > 0 || P.z === Q.z) continue;
        const k = (z - P.z) / (Q.z - P.z);
        cut.push(P.x + (Q.x - P.x) * k, P.y + (Q.y - P.y) * k);
      }
      if (cut.length < 4) continue;
      const steps = Math.ceil(Math.hypot(cut[2] - cut[0], cut[3] - cut[1]) / (0.5 * U)) + 1;
      for (let i = 0; i <= steps; i++) {
        const x = cut[0] + ((cut[2] - cut[0]) * i) / steps;
        const y = cut[1] + ((cut[3] - cut[1]) * i) / steps;
        const idx = (Math.round(binOf(x, y)) % NA) * NY + j;
        wall[idx] = Math.min(wall[idx], Math.hypot(x, y));
      }
    }
  }
  // Rim height per heading: the highest row the wall reaches.
  const top = new Float32Array(NA);
  const lip = new Float32Array(NA);
  for (let a = 0; a < NA; a++) {
    let j = NY - 1;
    while (j > 0 && wall[a * NY + j] === Infinity) j--;
    top[a] = z0 + (j + 0.5) * DZ;
    lip[a] = wall[a * NY + j];
  }
  const pts = [];
  for (const m of meshes.filter(x => norm(x.name).startsWith('shellhead'))) {
    const p = m.geometry.attributes.position;
    const P = k => new THREE.Vector3().fromBufferAttribute(p, k).applyMatrix4(m.matrixWorld).applyMatrix4(toModel);
    const keep = x => x.z < zTop + 30 * U && pts.push(x.sub(hr.p));
    for (let k = 0; k < p.count; k++) keep(P(k));
    const idx = m.geometry.index;
    // Triangle centroids and edge midpoints too: the outer shells are low-poly.
    for (let f = 0; idx && f < idx.count; f += 3) {
      const [a, b, c] = [P(idx.getX(f)), P(idx.getX(f + 1)), P(idx.getX(f + 2))];
      keep(a.clone().add(b).add(c).divideScalar(3));
      keep(a.clone().lerp(b, 0.5));
      keep(b.clone().lerp(c, 0.5));
      keep(c.clone().lerp(a, 0.5));
    }
  }
  const stride = Math.max(1, Math.floor(pts.length / 6000));
  const hull = pts.filter((_, k) => k % stride === 0);
  const n = hull.length;
  const wallAt = (a, z) => {
    const y = (z - z0) / DZ;
    const j = Math.floor(y);
    if (j < 0 || j >= NY - 1) return Infinity;
    const a0 = Math.floor(a) % NA;
    const a1 = (a0 + 1) % NA;
    const u = a - Math.floor(a);
    const w = y - j;
    let s = 0;
    let ws = 0;
    for (const [c, k] of [[wall[a0 * NY + j], (1 - u) * (1 - w)], [wall[a1 * NY + j], u * (1 - w)], [wall[a0 * NY + j + 1], (1 - u) * w], [wall[a1 * NY + j + 1], u * w]]) {
      if (c !== Infinity) {
        s += c * k;
        ws += k;
      }
    }
    return ws > 0.25 ? s / ws : Infinity;
  };
  const T = CLEAR.soft * U;
  const GAP = CLEAR.gap * U;
  const pts3 = new Float32Array(n * 3);
  const dk = new Float32Array(n);
  const rad = new THREE.Vector3();
  // Deepest contact (model units, negative when clear) for the head placed at
  // pts3 + c. Each point's depth is its shorter way out, toward the axis or
  // up; a hard minimum makes it a sharp tent where the two cost the same, and
  // a skirt sliding along the sloped rim then reads as a sawtooth
  // (neighbouring points take turns being deepest) that the push would copy
  // as a tremble. The minimum is rounded upward instead (exact where the two
  // ways tie, at most `soft` * ln 2 deeper elsewhere), so it never reads
  // shallower than the real contact. With `dir`, `rad` gets the horizontal
  // way toward the axis (a softmax over the contacts, so it turns smoothly as
  // the deepest one moves) and `coh` how much the contacts agree on it.
  let coh = 0;
  const contact = (c, dir = false) => {
    let worst = -Infinity;
    for (let k = 0; k < n; k++) {
      const x = pts3[3 * k] + c.x;
      const y = pts3[3 * k + 1] + c.y;
      const z = pts3[3 * k + 2] + c.z;
      const a = binOf(x, y);
      const a0 = Math.floor(a) % NA;
      const u = a - Math.floor(a);
      const t = top[a0] * (1 - u) + top[(a0 + 1) % NA] * u;
      let w = wallAt(a, z);
      if (w === Infinity && z < t) w = lip[a0] * (1 - u) + lip[(a0 + 1) % NA] * u;
      // Overhanging the rim (past its lip), the skirt keeps a visible gap.
      const lipR = lip[a0] * (1 - u) + lip[(a0 + 1) % NA] * u;
      const out = Math.min(1, Math.max(0, (Math.hypot(x, y) - lipR) / GAP));
      const up = t - z + GAP * out * out * (3 - 2 * out);
      let d = up;
      if (w !== Infinity) {
        const r = Math.hypot(x, y) - w;
        d = Math.min(r, up) + T * (Math.LN2 - Math.log1p(Math.exp(-Math.abs(r - up) / T)));
      }
      dk[k] = d;
      if (d > worst) worst = d;
    }
    if (!dir) return worst;
    let ws = 0;
    rad.set(0, 0, 0);
    for (let k = 0; k < n; k++) {
      if (dk[k] < worst - 14 * T) continue;
      const g = Math.exp((dk[k] - worst) / T);
      ws += g;
      const x = pts3[3 * k] + c.x;
      const y = pts3[3 * k + 1] + c.y;
      const h = Math.hypot(x, y) || 1;
      rad.x -= (g * x) / h;
      rad.y -= (g * y) / h;
    }
    const l = rad.length();
    coh = l / ws;
    if (l > 1e-9) rad.divideScalar(l);
    return worst;
  };
  const place = (q, wp) => {
    for (let k = 0; k < n; k++) {
      v.copy(hull[k]).applyQuaternion(q).add(wp);
      pts3[3 * k] = v.x;
      pts3[3 * k + 1] = v.y;
      pts3[3 * k + 2] = v.z;
    }
  };
  const zero = new THREE.Vector3();
  place(new THREE.Quaternion(), hr.p);
  const allow = Math.max(0, contact(zero));
  const K = CLEAR.knee * U;
  const B = CLEAR.blend * U;
  const FAR = CLEAR.far * U;
  const UP = new THREE.Vector3(0, 0, 1);
  const toAxis = new THREE.Vector3();
  const cs = new THREE.Vector3();
  let cost = 0;
  // Distance along the unit direction u that brings the contact back to the
  // rest allowance (e0: the depth before moving). Stepping by the depth left
  // never passes the first root (depth changes at most one-for-one with a
  // unit move). `cost` rates the direction: the best distance moved plus
  // `stall` times the depth still left over the steps, so a way out that
  // only just fails (the far wall catches the head) rates close to one that
  // only just succeeds, rather than jumping from one to the other.
  const reach = (u, e0) => {
    let s = 0;
    let e = e0;
    cost = CLEAR.stall * e0;
    for (let i = 0; i < CLEAR.iterations && e > 0 && s < FAR; i++) {
      s += e;
      e = contact(cs.copy(u).multiplyScalar(s)) - allow;
      cost = Math.min(cost, s + CLEAR.stall * Math.max(0, e));
    }
    return s;
  };
  // Translation (model frame, model units) that takes the head out of the
  // shell, up to the Huber residual. Both ways out are rated and the push
  // goes along their blend, weighted toward the cheaper one, then solved
  // along that blend: when the cheaper way changes, the push turns over a
  // few mm of travel instead of flipping from one frame to the next. The
  // way toward the axis only counts as far as the contacts agree on it
  // (none when they sit on both sides of the head).
  const clear = (q, wp, c) => {
    c.set(0, 0, 0);
    place(q, wp);
    const e0 = contact(c, true) - allow;
    if (e0 <= 0) return c;
    toAxis.copy(rad);
    const k = coh * coh;
    let sr = Infinity;
    if (k > 1e-3) {
      reach(toAxis, e0);
      sr = cost;
    }
    reach(UP, e0);
    const w = k / (1 + Math.exp(Math.max(-50, Math.min(50, (sr - cost) / B))));
    c.copy(toAxis).multiplyScalar(w).addScaledVector(UP, 1 - w).normalize();
    const s = reach(c, e0);
    return c.multiplyScalar(s < K ? (s * s) / (2 * K) : s - K / 2);
  };
  const out = { clear };
  _clearCache.set(source, out);
  return out;
}

export class Reachy {
  constructor(source) {
    this.model = skeletonClone(source);
    this.model.rotation.x = -Math.PI / 2;
    this.root = new THREE.Group();
    this.root.scale.setScalar(MODEL_SCALE);
    this.root.rotation.y = DISPLAY_YAW;
    this.root.add(this.model);
    this.group = new THREE.Group();
    this.group.add(this.root);

    this.model.updateWorldMatrix(true, true);
    const modelInv = new THREE.Matrix4().copy(this.model.matrixWorld).invert();
    const byName = {};
    this.model.traverse(o => {
      if (o.name) byName[norm(o.name)] = o;
    });
    this.bones = {};
    this.rest = {};
    for (const [key, name] of Object.entries(BONE)) {
      this.bones[key] = byName[norm(name)];
      this.rest[key] = this.bones[key].quaternion.clone();
    }
    const body = this.bones.body;
    const pq = new THREE.Quaternion();
    new THREE.Matrix4().multiplyMatrices(modelInv, body.parent.matrixWorld).decompose(new THREE.Vector3(), pq, new THREE.Vector3());
    this.bodyYawAxis = new THREE.Vector3(0, 0, 1).applyQuaternion(pq.invert()).normalize();

    const head = this.bones.head;
    const headModel = new THREE.Matrix4().multiplyMatrices(modelInv, head.matrixWorld);
    const parentModel = new THREE.Matrix4().multiplyMatrices(modelInv, head.parent.matrixWorld);
    const hr = { q: new THREE.Quaternion(), p: new THREE.Vector3(), s: new THREE.Vector3() };
    headModel.decompose(hr.p, hr.q, hr.s);
    hr.parentInv = parentModel.invert();
    this.headRest = hr;
    this.legs = LEG_IDS.map(id => setupLeg(id, byName, head)).filter(Boolean);

    this.meshes = [];
    this.model.traverse(o => {
      if (o.isMesh) {
        // Skinned antennas: their bounds are computed once, so they'd be culled after the explode moves them.
        if (o.isSkinnedMesh) o.frustumCulled = false;
        o.userData.origMaterial = o.material;
        this.meshes.push(o);
      }
    });
    this.clear = shellClearance(source, this.model, this.meshes, hr);
    this.#setupExplode();
    this._tmp = { p: new THREE.Vector3(), q: new THREE.Quaternion(), wq: new THREE.Quaternion(), wp: new THREE.Vector3(), m: new THREE.Matrix4() };
    this._lim = { p: new THREE.Vector3(), q: new THREE.Quaternion(), yaw: 0 };
  }

  // Assembly-drawing exploded view: parts are grouped into the robot's real
  // sub-assemblies, and each sub-assembly translates as one rigid block (the
  // antennas via their bones, since they are skinned). Offsets are in the
  // robot's own frame (y up, +z = face), converted to each node's parent frame.
  #setupExplode() {
    const layerOf = mesh => {
      if (mesh.isSkinnedMesh) return null;
      const n = norm(mesh.name);
      if (n.startsWith('shellheadtop')) return 'headTop';
      if (n.startsWith('shellheadfront')) return 'headFront';
      if (n.startsWith('shellheadback')) return 'headBack';
      if (n.startsWith('shellheadbot')) return 'headBot';
      if (n.startsWith('coquetop')) return 'bodyShell';
      if (n.startsWith('coquebot')) return 'base';
      if (n.startsWith('boule')) return 'neck'; // head-plate ball joints stay on their rods
      const up = [];
      for (let o = mesh.parent; o && o !== this.model; o = o.parent) up.push(norm(o.name));
      if (up.some(a => a.startsWith('antenna'))) return null;
      if (up.some(a => a.startsWith('neck'))) return 'neck';
      if (up.includes('core001')) return 'headCore';
      if (['core', 'corebot', 'coretop'].includes(up[0])) return 'motors';
      return 'base';
    };
    const toLocal = (node, v) => {
      const pq = new THREE.Quaternion();
      const ps = new THREE.Vector3();
      node.parent.matrixWorld.decompose(new THREE.Vector3(), pq, ps);
      return v.clone().applyQuaternion(pq.invert()).divide(ps);
    };
    const box = new THREE.Box3();
    const centers = {};
    this.parts = [];
    for (const mesh of this.meshes) {
      mesh.userData.shell = SHELLS.some(s => norm(mesh.name).startsWith(s));
      const id = layerOf(mesh);
      if (!id) continue;
      const L = EXPLODE[id];
      box.setFromObject(mesh);
      (centers[id] ??= new THREE.Box3()).union(box);
      this.parts.push({ id, node: mesh, rest: mesh.position.clone(), dir: toLocal(mesh, L.off), order: L.order });
    }
    for (const side of ['L', 'R']) {
      const bone = this.model.getObjectByName(`Antenna${side}001`) ?? this.model.getObjectByName(`Antenna.${side}.001`);
      if (bone) this.parts.push({ node: bone, rest: bone.position.clone(), dir: toLocal(bone, EXPLODE.antennas.off), order: EXPLODE.antennas.order });
    }
    // Label anchors: rest centre of each sub-assembly, stored in the frame of
    // one of its parts' parent (plus that part's offset direction), so the
    // anchor rides along when the robot dances while exploded.
    this.layers = Object.entries(centers).map(([id, b]) => {
      const part = this.parts.find(p => p.node.isMesh && layerOf(p.node) === id);
      const parent = part.node.parent;
      const c = b.getCenter(new THREE.Vector3());
      if (EXPLODE[id].pin) c.add(EXPLODE[id].pin);
      return { id, ...EXPLODE[id], parent, local: parent.worldToLocal(c), dir: part.dir };
    });
    // Antennas are skinned: anchor their label on the left antenna bone.
    const ant = this.parts.find(p => !p.node.isMesh);
    if (ant) this.layers.push({ id: 'antennas', ...EXPLODE.antennas, parent: ant.node.parent, local: ant.rest.clone(), dir: ant.dir });
  }

  // Per-part progress: a smootherstep over a window of the explode amount
  // that starts later for higher orders. Its first two derivatives vanish at
  // both ends, so a part driven by a C2 explode curve moves C2 too: no part
  // starts, stops or reverses with a kick.
  static explodeProgress(k, order) {
    const W = 0.7;
    const x = Math.max(0, Math.min(1, (k - order * (1 - W)) / W));
    return x * x * x * (x * (6 * x - 15) + 10);
  }

  setExplode(k) {
    if (k === this._explode) return;
    this._explode = k;
    for (const p of this.parts) p.node.position.copy(p.rest).addScaledVector(p.dir, Reachy.explodeProgress(k, p.order));
  }

  // World-space centre of each sub-assembly at the current explode amount and
  // pose (call after the group's world matrices are up to date).
  explodeAnchors() {
    const k = this._explode ?? 0;
    return this.layers.map(l => ({
      ...l,
      world: l.parent.localToWorld(l.local.clone().addScaledVector(l.dir, Reachy.explodeProgress(k, l.order))),
    }));
  }

  setYaw(yaw) {
    this.root.rotation.y = DISPLAY_YAW + yaw;
  }

  // pose: { p: Vector3 (m), q: Quaternion, ant: [right, left], yaw }, head in
  // the robot's base frame like the SDK's head pose. The head bone hangs off
  // the body, so the pose is brought into the body frame (limitHead).
  setPose(pose) {
    const { body, head, antL, antR } = this.bones;
    const t = this._tmp;
    const lim = limitHead(pose, this._lim);
    t.q.setFromAxisAngle(this.bodyYawAxis, lim.yaw);
    body.quaternion.copy(t.q).multiply(this.rest.body);

    const hr = this.headRest;
    t.p.copy(lim.p);
    t.q.copy(lim.q).premultiply(HEAD_FIX).multiply(HEAD_FIX_INV);
    t.p.applyQuaternion(HEAD_FIX);
    t.wq.copy(t.q).multiply(hr.q);
    t.wp.copy(hr.p).addScaledVector(t.p, UNITS_PER_M);
    t.wp.add(this.clear.clear(t.q, t.wp, t.p));
    t.m.compose(t.wp, t.wq, hr.s).premultiply(hr.parentInv);
    t.m.decompose(head.position, head.quaternion, head.scale);

    this.group.updateWorldMatrix(true, true);
    for (const leg of this.legs) solveLeg(leg);

    t.q.setFromAxisAngle(ANT_AXIS, ANT_SIGN * pose.ant[1]);
    antL.quaternion.copy(this.rest.antL).multiply(t.q);
    t.q.setFromAxisAngle(ANT_AXIS, ANT_SIGN * pose.ant[0]);
    antR.quaternion.copy(this.rest.antR).multiply(t.q);
  }
}
