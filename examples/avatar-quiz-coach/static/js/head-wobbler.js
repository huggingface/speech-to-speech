// @ts-check
/**
 * Reachy Mini's speech head wobble, ported from the official SDK
 * (pollen-robotics/reachy_mini, src/reachy_mini/motion/speech_tapper.py and
 * head_wobbler.py, Apache-2.0), as the real robot does it when it talks.
 *
 * The voice is analysed in 50 ms hops (RMS of the last 20 ms in dBFS); a
 * hysteretic VAD with attack and release gates an envelope, and loudness
 * scales six slow, out-of-phase sines (pitch, yaw, roll and x, y, z). Like
 * the SDK, each hop is scheduled for when its audio is heard: feed() takes
 * the PCM with its playback time on the audio clock, and at(t) returns the
 * offsets for time t, interpolated between hops.
 */

const FRAME_MS = 20;
const HOP_MS = 50;
const SWAY_MASTER = 1.5;
const SENS_DB_OFFSET = 4;
const VAD_DB_ON = -35;
const VAD_DB_OFF = -45;
const ATTACK_FR = Math.max(1, Math.floor(40 / HOP_MS));
const RELEASE_FR = Math.max(1, Math.floor(250 / HOP_MS));
const ENV_FOLLOW_GAIN = 0.65;
const SWAY_DB_LOW = -46;
const SWAY_DB_HIGH = -18;
const LOUDNESS_GAMMA = 0.9;
const SWAY_ATTACK_FR = Math.max(1, Math.floor(50 / HOP_MS));
const SWAY_RELEASE_FR = Math.max(1, Math.floor(250 / HOP_MS));
const DEG = Math.PI / 180;
// [frequency Hz, amplitude] per axis: degrees for rotations, metres for translations.
const SWAY = {
  pitch: [2.2, 4.5 * DEG],
  yaw: [0.6, 7.5 * DEG],
  roll: [1.3, 2.25 * DEG],
  x: [0.35, 0.0045],
  y: [0.45, 0.00375],
  z: [0.25, 0.00225],
};
// The SDK's oscillator phases: np.random.default_rng(seed=7), drawn in
// pitch, yaw, roll, x, y, z order.
const PHASE = { pitch: 3.927590651355011, yaw: 5.637360571650786, roll: 4.873776931938056, x: 1.4150185072200883, y: 1.8860003910648933, z: 5.488698173149897 };

/** @typedef {{ pitch: number, yaw: number, roll: number, x: number, y: number, z: number }} Offsets */
const ZERO = () => /** @type {Offsets} */ ({ pitch: 0, yaw: 0, roll: 0, x: 0, y: 0, z: 0 });

export class HeadWobbler {
  /** @param {number} sampleRate of the PCM fed in */
  constructor(sampleRate) {
    this.rate = sampleRate;
    this.frame = Math.round((sampleRate * FRAME_MS) / 1000);
    this.hop = Math.round((sampleRate * HOP_MS) / 1000);
    /** Samples left over from the last chunk, and when the first of them plays (ms). */
    this.carry = new Float32Array(0);
    this.carryAt = 0;
    /** Scheduled hops, by playback time (ms on the audio clock). @type {{ t: number, o: Offsets }[]} */
    this.hops = [];
    this.last = { t: 0, o: ZERO() };
    this.reset();
  }

  reset() {
    this.carry = new Float32Array(0);
    this.vadOn = false;
    this.vadAbove = 0;
    this.vadBelow = 0;
    this.env = 0;
    this.swayUp = 0;
    this.swayDown = 0;
    this.t = 0;
    this.hops = [];
  }

  /**
   * @param {Int16Array} pcm
   * @param {number} startMs when its first sample plays, on the audio clock
   */
  feed(pcm, startMs) {
    const msPerSample = 1000 / this.rate;
    // A gap in playback (the player ran dry): start afresh from this chunk.
    if (this.carry.length && Math.abs(this.carryAt + this.carry.length * msPerSample - startMs) > 30) this.carry = new Float32Array(0);
    const all = new Float32Array(this.carry.length + pcm.length);
    all.set(this.carry);
    for (let i = 0; i < pcm.length; i++) all[this.carry.length + i] = pcm[i] / 32768;
    const at0 = this.carry.length ? this.carryAt : startMs;
    let i = 0;
    for (; i + this.hop <= all.length; i += this.hop) {
      const o = this.step(all.subarray(i + this.hop - this.frame, i + this.hop));
      // The analysed frame is the end of the hop: schedule it as it plays.
      this.hops.push({ t: at0 + (i + this.hop - this.frame / 2) * msPerSample, o });
    }
    this.carry = all.slice(i);
    this.carryAt = at0 + i * msPerSample;
  }

  /** One hop of speech_tapper.SwayRollRT.feed. @param {Float32Array} frame */
  step(frame) {
    let e = 0;
    for (let k = 0; k < frame.length; k++) e += frame[k] * frame[k];
    const db = 20 * Math.log10(Math.sqrt(e / frame.length + 1e-12) + 1e-12);
    if (db >= VAD_DB_ON) {
      this.vadAbove++;
      this.vadBelow = 0;
      if (!this.vadOn && this.vadAbove >= ATTACK_FR) this.vadOn = true;
    } else if (db <= VAD_DB_OFF) {
      this.vadBelow++;
      this.vadAbove = 0;
      if (this.vadOn && this.vadBelow >= RELEASE_FR) this.vadOn = false;
    }
    if (this.vadOn) {
      this.swayUp = Math.min(SWAY_ATTACK_FR, this.swayUp + 1);
      this.swayDown = 0;
    } else {
      this.swayDown = Math.min(SWAY_RELEASE_FR, this.swayDown + 1);
      this.swayUp = 0;
    }
    const target = this.vadOn ? this.swayUp / SWAY_ATTACK_FR : 1 - this.swayDown / SWAY_RELEASE_FR;
    this.env = Math.max(0, Math.min(1, this.env + ENV_FOLLOW_GAIN * (target - this.env)));
    const norm = Math.max(0, Math.min(1, (db + SENS_DB_OFFSET - SWAY_DB_LOW) / (SWAY_DB_HIGH - SWAY_DB_LOW)));
    const k = norm ** LOUDNESS_GAMMA * SWAY_MASTER * this.env;
    this.t += HOP_MS / 1000;
    const o = ZERO();
    for (const axis of /** @type {(keyof Offsets)[]} */ (Object.keys(SWAY))) {
      const [f, a] = SWAY[axis];
      o[axis] = a * k * Math.sin(2 * Math.PI * f * this.t + PHASE[axis]);
    }
    return o;
  }

  /**
   * Offsets for time `t` (ms on the audio clock), linear between hops; zero
   * once the scheduled audio has played out.
   * @param {number} t @param {Offsets} out
   */
  at(t, out) {
    while (this.hops.length && this.hops[0].t <= t) this.last = /** @type {{ t: number, o: Offsets }} */ (this.hops.shift());
    let next = this.hops[0];
    let a = this.last;
    // A new line after a pause: hold neutral until one hop before it.
    if (next && next.t - a.t > 2 * HOP_MS) {
      if (t < next.t - HOP_MS) next = undefined;
      else a = { t: next.t - HOP_MS, o: ZERO() };
    }
    // Past the last hop (the line ended or was cut off): ease back to neutral.
    const b = next ?? { t: a.t + 6 * HOP_MS, o: ZERO() };
    const u = Math.max(0, Math.min(1, (t - a.t) / Math.max(1, b.t - a.t)));
    for (const axis of /** @type {(keyof Offsets)[]} */ (Object.keys(SWAY))) out[axis] = a.o[axis] + (b.o[axis] - a.o[axis]) * u;
    return out;
  }
}
