// MuJoCo (MJCF) loader for the Microduck coach, trimmed from Fabien Danieau's
// Reachy Mini talk (https://github.com/FabienDanieau/reachy_mini_intro,
// assets/robots.js, CC BY 4.0, (c) Fabien Danieau) as adapted in
// https://github.com/andimarafioti/agentic-days-keynote. Unmodified apart from
// this header and the removal of the parts the coach doesn't use.

// Minimal MJCF loader (microduck): kinematic tree + visual STL meshes, posed by
// an optional { jointName: angle } map. Returned group is Y-up (MJCF is Z-up).
// ponytail: hinge joints about the body origin, no default-class inheritance.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// Three tries, so a network hiccup doesn't lose a robot.
async function getText(url) {
    for (let i = 1; ; i++) {
        try { const r = await fetch(url, { cache: 'no-cache' }); if (!r.ok) throw new Error(`${url}: ${r.status}`); return await r.text(); }
        catch (e) { if (i >= 3) throw e; await new Promise(ok => setTimeout(ok, 700 * i)); }
    }
}
const nums = (s, d) => (s ? s.trim().split(/\s+/).map(Number) : d);
const kids = (el, tag) => [...el.children].filter(c => c.tagName === tag);

// Smooth normals across faces under 30 deg, sharp CAD edges above, so decimated parts don't look faceted.
// Same rule as three's toCreasedNormals, which takes about a minute on these 3M triangles (a string key
// per vertex): corners are welded on a numeric key, and each corner averages the normals of the faces
// around its vertex that are within the crease angle of its own face.
function creaseNormals(geo, crease = Math.PI / 6) {
    const p = geo.attributes.position.array, n = p.length / 3, nf = n / 3, cos = Math.cos(crease);
    const fn = new Float32Array(nf * 3);
    for (let f = 0; f < nf; f++) {
        const i = f * 9, ax = p[i + 3] - p[i], ay = p[i + 4] - p[i + 1], az = p[i + 5] - p[i + 2], bx = p[i + 6] - p[i], by = p[i + 7] - p[i + 1], bz = p[i + 8] - p[i + 2];
        const x = ay * bz - az * by, y = az * bx - ax * bz, z = ax * by - ay * bx, l = Math.hypot(x, y, z) || 1;
        fn[f * 3] = x / l; fn[f * 3 + 1] = y / l; fn[f * 3 + 2] = z / l;
    }
    geo.computeBoundingBox();
    const { min, max } = geo.boundingBox, q = (max.distanceTo(min) || 1) * 2e-5, K = 65536; // < 2^53 for any part
    const ids = new Map(), vid = new Uint32Array(n);
    for (let c = 0; c < n; c++) {
        const key = (Math.round((p[c * 3] - min.x) / q) * K + Math.round((p[c * 3 + 1] - min.y) / q)) * K + Math.round((p[c * 3 + 2] - min.z) / q);
        let v = ids.get(key);
        if (v === undefined) ids.set(key, (v = ids.size));
        vid[c] = v;
    }
    const start = new Uint32Array(ids.size + 1), list = new Uint32Array(n); // corners around each vertex
    for (let c = 0; c < n; c++) start[vid[c] + 1]++;
    for (let v = 0; v < ids.size; v++) start[v + 1] += start[v];
    const fill = start.slice(0, ids.size);
    for (let c = 0; c < n; c++) list[fill[vid[c]]++] = c;
    const out = new Float32Array(n * 3);
    for (let c = 0; c < n; c++) {
        const f = (c / 3) | 0, fx = fn[f * 3], fy = fn[f * 3 + 1], fz = fn[f * 3 + 2], v = vid[c];
        let x = 0, y = 0, z = 0;
        for (let k = start[v]; k < start[v + 1]; k++) {
            const g = ((list[k] / 3) | 0) * 3, gx = fn[g], gy = fn[g + 1], gz = fn[g + 2];
            if (gx * fx + gy * fy + gz * fz >= cos) { x += gx; y += gy; z += gz; }
        }
        const l = Math.hypot(x, y, z) || 1;
        out[c * 3] = x / l; out[c * 3 + 1] = y / l; out[c * 3 + 2] = z / l;
    }
    geo.setAttribute('normal', new THREE.BufferAttribute(out, 3));
    return geo;
}

// Each robot's parts come in one pack, meshes.glb (tools/pack_meshes.mjs): the CAD STLs simplified to
// 0.05 mm and Draco-compressed, under 1 MB for the five robots instead of about 100 MB of STL. A part is
// found by its STL path inside the robot folder. The scene hands over its GLB loader (Draco, claude.ai bundles).
let loadGLB = null;
export const setPackLoader = fn => { loadGLB = fn; };
const packs = new Map();
function partGeometry(url) {
    const [, dir, file] = url.match(/^(.*\/robots\/[^/]+)\/(.+)$/);
    if (!packs.has(dir)) packs.set(dir, loadGLB(`${dir}/meshes.glb`).then(gltf => {
        const byFile = new Map();
        gltf.scene.traverse(o => { if (o.isMesh) byFile.set(o.userData.file, o.geometry); });
        return byFile;
    }));
    return packs.get(dir).then(byFile => {
        if (!byFile.has(file)) throw new Error(`${file} is not in ${dir}/meshes.glb`);
        return byFile.get(file).toNonIndexed(); // triangle soup, as the STL was: creaseNormals expects it
    });
}
// Load each part once; callers clone.
const cache = new Map();
function loadMesh(url) {
    if (!cache.has(url)) cache.set(url, partGeometry(url).then(creaseNormals));
    return cache.get(url).then(geo => new THREE.Mesh(geo));
}

// recolor(materialName) -> null | { color, roughness?, metalness? } overrides a CAD colour.
// keep: mesh names left as their own object (looked up by name later); every other part of a body is
// merged with the body's other parts of the same look, so a robot is a few draws instead of one per part.
export async function loadMJCF(url, { meshDir, joints = {}, recolor = () => null, keep = [] } = {}) {
    const doc = new DOMParser().parseFromString(await getText(url), 'application/xml');
    const meshes = Object.fromEntries([...doc.querySelectorAll('asset > mesh')].map(m => {
        const file = m.getAttribute('file');
        return [m.getAttribute('name') ?? file.replace(/\.[^.]+$/, ''), file];
    }));
    const mats = Object.fromEntries([...doc.querySelectorAll('asset > material')].map(m => [m.getAttribute('name'), nums(m.getAttribute('rgba'), [0.8, 0.8, 0.8, 1])]));
    const jobs = [];
    const place = (obj, el) => {
        obj.position.fromArray(nums(el.getAttribute('pos'), [0, 0, 0]));
        const [w, x, y, z] = nums(el.getAttribute('quat'), [1, 0, 0, 0]);
        obj.quaternion.set(x, y, z, w);
    };
    const build = el => {
        const g = new THREE.Group(); g.name = el.getAttribute('name') ?? '';
        bodies[g.name] = g;
        place(g, el);
        const inner = new THREE.Group(); // joint rotation, applied after the body frame
        g.add(inner);
        for (const j of kids(el, 'joint')) {
            const a = joints[j.getAttribute('name')];
            if (a) inner.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(...nums(j.getAttribute('axis'), [0, 0, 1])).normalize(), a));
        }
        for (const geom of kids(el, 'geom')) {
            if (geom.getAttribute('class') !== 'visual' || geom.getAttribute('type') !== 'mesh') continue;
            const holder = new THREE.Group();
            holder.name = geom.getAttribute('mesh');
            place(holder, geom);
            inner.add(holder);
            const name = geom.getAttribute('material');
            const [r, gr, b] = mats[name] ?? [0.8, 0.8, 0.8];
            const over = recolor(name);
            jobs.push(loadMesh(`${meshDir}/${meshes[geom.getAttribute('mesh')]}`).then(m => {
                // MJCF rgba from Onshape is sRGB; Color(r, g, b) would read it as linear and wash it out.
                m.material = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(r, gr, b, THREE.SRGBColorSpace), roughness: 0.55, metalness: 0.05, ...over });
                m.castShadow = m.receiveShadow = true;
                holder.add(m);
            }));
        }
        for (const b of kids(el, 'body')) inner.add(build(b));
        return g;
    };
    const bodies = {};
    const root = new THREE.Group();
    for (const b of kids(doc.querySelector('worldbody'), 'body')) root.add(build(b));
    await Promise.all(jobs);
    const looks = new Map(); // one material per colour / finish, shared by the merged meshes
    for (const g of Object.values(bodies)) {
        const inner = g.children[0], byLook = new Map();
        for (const h of [...inner.children]) {
            const m = h.children[0];
            if (!m?.isMesh || keep.includes(h.name)) continue; // child bodies are skipped: no mesh child
            const mat = m.material, key = [mat.color.getHex(), mat.roughness, mat.metalness, mat.envMapIntensity].join();
            if (!looks.has(key)) looks.set(key, mat);
            h.updateMatrix();
            (byLook.get(key) ?? byLook.set(key, []).get(key)).push(m.geometry.clone().applyMatrix4(h.matrix));
            inner.remove(h);
        }
        for (const [key, geos] of byLook) {
            const mesh = new THREE.Mesh(mergeGeometries(geos), looks.get(key));
            mesh.castShadow = mesh.receiveShadow = true;
            inner.add(mesh);
        }
    }
    const out = new THREE.Group();
    root.rotation.x = -Math.PI / 2;
    out.add(root);
    out.userData.bodies = bodies; // body name -> group (its first child carries the joint rotation)
    return out;
}

// Play a clip baked by shot-tools/mjcf_clip.py on a loadMJCF() model: each body's pose
// relative to its parent at `t` seconds (linear between samples), the joint groups reset.
export function poseMJCF(model, clip, t) {
    const n = clip.bodies[Object.keys(clip.bodies)[0]].pos.length;
    const x = Math.max(0, Math.min(n - 1, t * clip.fps)), i = Math.floor(x), k = x - i, j = Math.min(n - 1, i + 1);
    for (const [name, b] of Object.entries(clip.bodies)) {
        const g = model.userData.bodies[name];
        if (!g) continue;
        g.position.set(...b.pos[i]).lerp(_v.set(...b.pos[j]), k);
        g.quaternion.set(...b.quat[i]).slerp(_q.set(...b.quat[j]), k);
        g.children[0].quaternion.identity();
    }
}
const _v = new THREE.Vector3(), _q = new THREE.Quaternion();
