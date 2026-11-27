// @ts-check
/**
 * A small three.js stage for coaches drawn with plain three.js (Reachy):
 * transparent WebGL2 canvas over the page's backdrop, a camera that keeps
 * the character framed at any aspect, soft studio lights and a procedural
 * environment for reflections (no files to download).
 */

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export class Stage {
  /**
   * @param {HTMLElement} container
   * @param {{ focusY?: number, frameH?: number, frameW?: number, fov?: number, maxPixelRatio?: number }} [opt]
   */
  constructor(container, opt = {}) {
    this.container = container;
    this.opt = { focusY: -0.25, frameH: 3.1, frameW: 2.6, fov: 26, maxPixelRatio: 2, ...opt };
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

    // Key light warm from the upper left, a lilac rim from behind to lift the
    // silhouette off the dark backdrop, and a soft sky/ground fill.
    const hemi = new THREE.HemisphereLight(0xfff4e8, 0x3a2f6a, 0.75);
    const key = new THREE.DirectionalLight(0xfff0dc, 2.1);
    key.position.set(-2.2, 3, 4);
    const rim = new THREE.DirectionalLight(0xb9a8ff, 2.6);
    rim.position.set(2.8, 2.2, -3);
    const rim2 = new THREE.DirectionalLight(0xffc48a, 1.4);
    rim2.position.set(-3, 0.5, -2.5);
    const fill = new THREE.DirectionalLight(0xffe4cc, 0.5);
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
    cam.position.set(0, this.opt.focusY + 0.12, dist);
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

  /** Average frames per second over the last ~2 s. */
  measuredFps() {
    const n = this.frameTimes.length;
    if (!n) return 0;
    return 1000 / (this.frameTimes.reduce((a, b) => a + b, 0) / n);
  }
}
