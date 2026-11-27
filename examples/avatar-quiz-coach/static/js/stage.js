// @ts-check
/**
 * A small three.js stage for coaches drawn with plain three.js (the robots):
 * transparent WebGL2 canvas over the page's backdrop, a camera that keeps
 * the character framed at any aspect (optionally from above), soft studio
 * lights, a procedural environment for reflections and, for robots that
 * stand on something, a wooden tabletop with soft shadows (all generated:
 * no files to download).
 */

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export class Stage {
  /**
   * @param {HTMLElement} container
   * @param {{ focusY?: number, frameH?: number, frameW?: number, fov?: number, tiltDeg?: number, maxPixelRatio?: number }} [opt]
   *   frameH/frameW: the height and width to fit, around (0, focusY, 0);
   *   tiltDeg: how far the camera looks down at it.
   */
  constructor(container, opt = {}) {
    this.container = container;
    this.opt = { focusY: -0.25, frameH: 3.1, frameW: 2.6, fov: 26, tiltDeg: 0, maxPixelRatio: 2, ...opt };
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.95;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.opt.maxPixelRatio));
    container.appendChild(renderer.domElement);
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.35;
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(this.opt.fov, 1, 0.1, 60);

    // Key light warm from the upper left, crisp studio rim from behind to
    // separate the silhouette cleanly from the daylight stone backdrop.
    const hemi = new THREE.HemisphereLight(0xf6f9fc, 0x7c8a98, 0.95);
    const key = new THREE.DirectionalLight(0xfff6ea, 1.9);
    key.position.set(-2.2, 3, 4);
    const rim = new THREE.DirectionalLight(0xdce8f8, 1.6);
    rim.position.set(2.8, 2.2, -3);
    const rim2 = new THREE.DirectionalLight(0xffe0bc, 0.9);
    rim2.position.set(-3, 0.5, -2.5);
    const fill = new THREE.DirectionalLight(0xeef3f8, 0.55);
    fill.position.set(2, -0.5, 3);
    this.scene.add(hemi, key, rim, rim2, fill);
    this.lights = { hemi, key, rim, rim2, fill };

    this.resize = this.resize.bind(this);
    new ResizeObserver(this.resize).observe(container);
    this.resize();

    /** @type {((dt: number, t: number) => void)[]} */
    this.updaters = [];
    this.running = true;
    this.fps = 0;
    this.frameTimes = /** @type {number[]} */ ([]);
    this.last = performance.now();
    renderer.setAnimationLoop(() => this.tick());
    document.addEventListener("visibilitychange", () => {
      this.running = document.visibilityState === "visible";
      renderer.setAnimationLoop(this.running ? () => this.tick() : null);
      this.last = performance.now();
    });
  }

  resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    const cam = this.camera;
    cam.aspect = w / h;
    // Distance that fits frameH vertically and frameW horizontally.
    const vfov = (cam.fov * Math.PI) / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * cam.aspect);
    const dist = Math.max(this.opt.frameH / 2 / Math.tan(vfov / 2), this.opt.frameW / 2 / Math.tan(hfov / 2));
    const tilt = Math.max((this.opt.tiltDeg * Math.PI) / 180, 0.035);
    cam.position.set(0, this.opt.focusY + dist * Math.sin(tilt), dist * Math.cos(tilt));
    cam.lookAt(0, this.opt.focusY, 0);
    cam.updateProjectionMatrix();
  }

  tick() {
    const t = performance.now();
    const dt = Math.min(100, t - this.last);
    this.last = t;
    this.frameTimes.push(dt);
    if (this.frameTimes.length > 120) this.frameTimes.shift();
    for (const u of this.updaters) u(dt, t);
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * A wooden tabletop at height `y` that fades into the page around the
   * character, lit like Pollen's reachy-mini-3d-voice Space: a cool key from
   * behind and above casts a soft shadow toward the viewer, a warm front fill
   * keeps the face readable, and the sky and the wood light it from above
   * and below.
   * @param {THREE.Object3D} subject casts the shadow
   * @param {{ y: number, radius: number }} opt
   */
  addTable(subject, { y, radius }) {
    const r = this.renderer;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    // Small parts (screws, eye internals) don't change the shadow; leaving
    // them out keeps the shadow pass to a fraction of the draws.
    subject.updateWorldMatrix(true, true);
    const box = new THREE.Box3();
    const size = new THREE.Vector3();
    subject.traverse((/** @type {any} */ o) => {
      if (o.isMesh && o.visible) o.castShadow = o.isSkinnedMesh || box.setFromObject(o).getSize(size).length() > radius * 0.06;
    });
    const mat = new THREE.MeshStandardMaterial({ map: woodTexture(), roughness: 0.62, metalness: 0, transparent: true, alphaMap: fadeTexture(), depthWrite: false });
    const table = new THREE.Mesh(new THREE.CircleGeometry(radius, 96), mat);
    table.rotation.x = -Math.PI / 2;
    table.position.y = y;
    table.receiveShadow = true;
    table.renderOrder = -1;
    this.scene.add(table);

    const L = this.lights;
    L.rim.intensity = 0;
    L.rim2.intensity = 0;
    L.fill.intensity = 0;
    L.key.color.set(0xfff4e8);
    L.key.intensity = 0.9;
    L.key.position.set(-1.6, 1.5, 2.2);
    L.hemi.color.set(0xdbe7ee);
    L.hemi.groundColor.set(0xb0864f);
    L.hemi.intensity = 0.55;
    const sun = new THREE.DirectionalLight(0xeaf2f6, 2.2);
    sun.position.set(-1.2, 4.5, -3.2).multiplyScalar(radius / 3);
    sun.target.position.set(0.2, y, 0.6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    const e = radius * 0.9;
    Object.assign(sun.shadow.camera, { left: -e, right: e, top: e, bottom: -e, near: 0.1, far: radius * 6 });
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    sun.shadow.radius = 6;
    const lamp = new THREE.PointLight(0xfff0dd, 1.2 * radius, radius * 3, 2);
    lamp.position.set(-radius * 0.6, y + radius * 0.8, radius * 0.5);
    this.scene.add(sun, sun.target, lamp);
    Object.assign(this.lights, { sun, lamp });
    return table;
  }

  /** Average frames per second over the last ~2 s. */
  measuredFps() {
    const n = this.frameTimes.length;
    if (!n) return 0;
    return 1000 / (this.frameTimes.reduce((a, b) => a + b, 0) / n);
  }
}

/** Oak-like grain drawn on a canvas: long wavy fibres over a warm base. */
function woodTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 1024;
  const g = /** @type {CanvasRenderingContext2D} */ (c.getContext("2d"));
  g.fillStyle = "#8a5d3b";
  g.fillRect(0, 0, 1024, 1024);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 520; i++) {
    const y0 = rnd() * 1024;
    const dark = rnd() < 0.55;
    g.strokeStyle = dark ? `rgba(70, 40, 22, ${0.05 + rnd() * 0.18})` : `rgba(196, 146, 96, ${0.04 + rnd() * 0.12})`;
    g.lineWidth = 0.6 + rnd() * (dark ? 2.6 : 1.6);
    const amp = 2 + rnd() * 9;
    const f = 0.002 + rnd() * 0.006;
    const ph = rnd() * 6.3;
    g.beginPath();
    for (let x = -8; x <= 1032; x += 8) {
      const yy = y0 + amp * Math.sin(x * f + ph) + 3 * Math.sin(x * f * 3.7 + ph * 2);
      if (x < 0) g.moveTo(x, yy);
      else g.lineTo(x, yy);
    }
    g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(1.6, 1.6);
  t.anisotropy = 8;
  return t;
}

/** Opaque in the middle, fading out to the edge (for alphaMap: green channel). */
function fadeTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = /** @type {CanvasRenderingContext2D} */ (c.getContext("2d"));
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, "#fff");
  grad.addColorStop(0.45, "#eee");
  grad.addColorStop(0.8, "#555");
  grad.addColorStop(1, "#000");
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}
