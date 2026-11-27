// @ts-check
/**
 * Avatar lab (static/lab.html): drive a coach without a voice session.
 * `?coach=<key>` picks the coach. Speaks the test clips in static/_test
 * through beginResponse/feed/endResponse in 40 ms chunks, like the backend,
 * with HeadAudio or with fake server viseme cues, and fires the reactions.
 * `window.lab` exposes the same calls for scripted checks.
 */

import { COACHES } from "./coaches.js";
import { Quacker } from "./quack.js";

const $ = (/** @type {string} */ s) => /** @type {HTMLElement} */ (document.querySelector(s));
const params = new URLSearchParams(location.search);
const key = params.get("coach") || "max";
const coach = /** @type {any} */ (COACHES)[key] || COACHES.max;

const t0 = performance.now();
/** @type {any} */
let avatar;
if (coach.renderer === "splat") avatar = new (await import("./splat-avatar.js")).SplatAvatar($("#avatar"), coach);
else if (coach.renderer === "talkinghead") avatar = new (await import("./avatar.js")).Avatar($("#avatar"), coach);
else {
  const mod = await import(`./${coach.renderer}-avatar.js`);
  const Cls = Object.values(mod).find((v) => typeof v === "function" && /Avatar$/.test(v.name));
  avatar = new Cls($("#avatar"), coach);
}
await avatar.init((/** @type {any} */ ev) => {
  if (ev.lengthComputable) $("#avatar-pct").textContent = `${Math.round((ev.loaded / ev.total) * 100)}%`;
});
const loadMs = performance.now() - t0;
$("#avatar-loading").classList.add("done");

/** @param {string} url */
async function loadPcm(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  let off = 12;
  while (off < buf.byteLength) {
    const id = String.fromCharCode(...new Uint8Array(buf, off, 4));
    const len = dv.getUint32(off + 4, true);
    if (id === "data") return new Int16Array(buf.slice(off + 8, off + 8 + len));
    off += 8 + len;
  }
  throw new Error("no data chunk");
}

/**
 * Rough stand-in for the backend's phoneme cues: an open vowel on each
 * loud 120 ms stretch, a lip closure on each dip (Microsoft viseme IDs).
 * @param {Int16Array} pcm
 */
function fakeCues(pcm) {
  const cues = [];
  const hop = 1200; // 50 ms
  const vowels = [2, 4, 1, 6, 3, 7, 4, 2];
  let k = 0;
  let prevLoud = false;
  for (let i = 0; i + hop <= pcm.length; i += hop) {
    let e = 0;
    for (let j = i; j < i + hop; j++) e += (pcm[j] / 32768) ** 2;
    const loud = Math.sqrt(e / hop) > 0.03;
    const s = i / 24000;
    if (loud) cues.push({ viseme: vowels[k++ % vowels.length], start_s: s, end_s: s + 0.05 });
    else if (prevLoud) cues.push({ viseme: 21, start_s: s, end_s: s + 0.05 });
    prevLoud = loud;
  }
  return cues;
}

let feeding = 0;
/**
 * Speak a clip as one response.
 * @param {string} name @param {{ cues?: boolean }} [opt]
 */
async function speak(name, opt = {}) {
  await avatar.resume();
  const pcm = await loadPcm(`static/_test/${name}.wav`);
  const id = "lab-" + Math.random().toString(36).slice(2);
  avatar.beginResponse(id);
  if (opt.cues ?? /** @type {HTMLInputElement} */ ($("#cues")).checked) avatar.addVisemes(id, fakeCues(pcm));
  const chunk = 960;
  // A coach that speaks duck quacks over the clip instead.
  const quacker = /** @type {any} */ (coach).quack ? new Quacker() : null;
  let i = 0;
  clearInterval(feeding);
  return new Promise((resolve) => {
    feeding = window.setInterval(() => {
      for (let k = 0; k < 2 && i < pcm.length; k++, i += chunk) {
        const part = pcm.subarray(i, i + chunk);
        avatar.feed(id, quacker ? quacker.process(part) : part);
      }
      if (i >= pcm.length) {
        clearInterval(feeding);
        avatar.endResponse();
        resolve(id);
      }
    }, 40);
  });
}

for (const [k, c] of Object.entries(COACHES)) {
  const b = document.createElement("button");
  b.textContent = /** @type {any} */ (c).name;
  b.classList.toggle("on", k === key);
  b.onclick = () => (location.search = `?coach=${k}`);
  $("#coaches").append(b);
}
for (const clip of ["c0", "c1", "c2", "c3", "hint", "t1", "t4"]) {
  const b = document.createElement("button");
  b.textContent = clip;
  b.onclick = () => speak(clip);
  $("#speak").append(b);
}
/** @type {[string, () => void][]} */
const actions = [
  ["correct", () => avatar.react("correct", { streak: 1 })],
  ["streak", () => avatar.react("correct", { streak: 3 })],
  ["partial", () => avatar.react("partial")],
  ["incorrect", () => avatar.react("incorrect")],
  ["clue", () => avatar.react("clue")],
  ["skipped", () => avatar.react("skipped")],
  ["think", () => avatar.think()],
  ["listen", () => avatar.listen()],
  ["celebrate", () => avatar.celebrate()],
  ["happy", () => avatar.setMood("happy")],
  ["neutral", () => avatar.setMood("neutral")],
  ["interrupt", () => avatar.interrupt()],
];
for (const [name, fn] of actions) {
  const b = document.createElement("button");
  b.textContent = name;
  b.onclick = fn;
  $("#react").append(b);
}

// FPS from requestAnimationFrame, whatever the renderer.
const times = /** @type {number[]} */ ([]);
let last = performance.now();
(function loop() {
  const t = performance.now();
  times.push(t - last);
  last = t;
  if (times.length > 120) times.shift();
  requestAnimationFrame(loop);
})();
setInterval(() => {
  const fps = 1000 / (times.reduce((a, b) => a + b, 0) / Math.max(1, times.length));
  const gl = /** @type {HTMLCanvasElement | null} */ ($("#avatar").querySelector("canvas"));
  $("#stats").textContent = [
    `coach    ${key} (${coach.renderer})`,
    `load     ${Math.round(loadMs)} ms`,
    `fps      ${fps.toFixed(0)}`,
    `canvas   ${gl ? `${gl.width}x${gl.height}` : "-"}`,
    `lipsync  ${avatar.lipsyncSource}`,
    `playing  ${avatar.playing}`,
  ].join("\n");
}, 500);

Object.assign(window, { lab: { avatar, speak, fakeCues, loadMs, fps: () => 1000 / (times.reduce((a, b) => a + b, 0) / Math.max(1, times.length)) } });
