// @ts-check
/**
 * Microduck talks duck: its voice is swapped for the robot's own quacks,
 * following the speech's rhythm, while the words show as subtitles.
 *
 * The quacks are Microduck's: a port of the `chirp` recipe (its mouth-trigger
 * sound, 12 variants) from pollen-robotics/microduck `sounds/` (Apache-2.0),
 * itself ported from apirrone/microduck_sounds. A seed derives the duck's
 * personality (pitch, harmonic tilt, nasality, vibrato, quackiness...) with
 * the same random generator, so a seed sounds like the same duck. Rendered at
 * 24 kHz here instead of the robot's 48 kHz.
 *
 * Quacker turns the voice's PCM into a stream of the same length with a quack
 * on each burst of syllables, so lip sync, subtitles, pauses and barge-in all
 * keep their timing.
 */

const SR = 24000;
const TUNED_SR = 22050;
const TAU = 2 * Math.PI;
const M64 = (1n << 64n) - 1n;
const rotl = (/** @type {bigint} */ x, /** @type {bigint} */ k) => ((x << k) | (x >> (64n - k))) & M64;

/** xoshiro256++ seeded with splitmix64, as in microduck's `sounds/src/rng.rs`. */
class Rng {
  /** @param {number} seed u32 */
  constructor(seed) {
    let x = BigInt(seed >>> 0);
    const next = () => {
      x = (x + 0x9e3779b97f4a7c15n) & M64;
      let z = x;
      z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M64;
      z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M64;
      return z ^ (z >> 31n);
    };
    this.s = [next(), next(), next(), next()];
    /** @type {number | null} */
    this.spare = null;
  }

  nextU64() {
    const s = this.s;
    const result = (rotl((s[0] + s[3]) & M64, 23n) + s[0]) & M64;
    const t = (s[1] << 17n) & M64;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45n);
    return result;
  }

  random() {
    return Number(this.nextU64() >> 11n) / 2 ** 53;
  }

  /** @param {number} lo @param {number} hi */
  uniform(lo, hi) {
    return lo + (hi - lo) * this.random();
  }

  /** @param {number} lo @param {number} hi */
  integers(lo, hi) {
    return lo + Math.floor(this.random() * (hi - lo));
  }

  /** @param {number[]} choices */
  choice(choices) {
    return choices[this.integers(0, choices.length)];
  }

  standardNormal() {
    if (this.spare !== null) {
      const z = this.spare;
      this.spare = null;
      return z;
    }
    const r = 1 - this.random();
    const theta = TAU * this.random();
    const mag = Math.sqrt(-2 * Math.log(r));
    this.spare = mag * Math.sin(theta);
    return mag * Math.cos(theta);
  }
}

/** @param {string} text */
function crc32(text) {
  let crc = 0xffffffff;
  for (const byte of new TextEncoder().encode(text)) {
    crc ^= byte;
    for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return ~crc >>> 0;
}

/** A duck's voice, derived from a seed (microduck `personality.rs`). */
export class Personality {
  /** @param {number} seed */
  constructor(seed) {
    const rng = new Rng(seed);
    this.seed = seed >>> 0;
    const register = rng.choice([-1, 0, 0, 1]) + rng.uniform(-0.4, 0.4);
    const base = rng.uniform(160, 380);
    this.pitch = Math.min(620, Math.max(110, base * 2 ** (register * 0.45)));
    this.register = register;
    this.pitchSpread = rng.uniform(0.4, 1.2);
    this.glideBias = rng.uniform(-1, 1);
    this.brightness = rng.uniform(0.05, 0.55);
    this.tilt = rng.uniform(1.4, 2.8);
    this.nasal = rng.uniform(0.1, 1.0);
    this.harmonicSkew = rng.uniform(-1, 1);
    this.formantN = rng.integers(1, 6);
    this.formantGain = rng.uniform(0, 1.4);
    this.vibratoRate = rng.uniform(3.5, 9.5);
    this.vibratoDepth = rng.uniform(0, 0.7);
    this.jitterDepth = rng.uniform(0.03, 0.35);
    this.breath = rng.uniform(0, 0.3);
    this.quackiness = rng.uniform(0.2, 1.0);
    this.amRate = rng.uniform(18, 55);
    this.amDepth = rng.uniform(0.15, 0.7);
    this.warbleHz = rng.uniform(7, 18);
    this.warbleDepth = rng.uniform(0, 1.4);
    this.attackSharpness = rng.uniform(0, 1);
    this.speed = rng.uniform(0.82, 1.22);
  }

  /** @param {string} tag @param {number} variant */
  variantRng(tag, variant) {
    const h = ((BigInt(this.seed) * 1000003n) & M64) ^ BigInt(crc32(tag)) ^ ((BigInt(variant) * 2654435761n) & M64);
    return new Rng(Number(h & 0xffffffffn));
  }

  harmonics() {
    const N = 7;
    const w = [];
    for (let n = 1; n <= N; n++) {
      const base = 1 / n ** this.tilt;
      const highLift = this.brightness * (n / N) ** 1.5;
      const nasal = this.nasal * (n === 2 || n === 3 ? 0.6 : 0);
      const skew =
        this.harmonicSkew >= 0 ? this.harmonicSkew * (n % 2 === 0 ? 0.4 : -0.2) : -this.harmonicSkew * (n % 2 === 0 ? -0.3 : 0.4);
      const formant = n === this.formantN ? this.formantGain : 0;
      w.push(Math.max(0, base + highLift + nasal + skew + formant * base * 1.5));
    }
    w[0] = Math.max(w[0], 0.7);
    return w;
  }
}

// ── The synth (microduck `synth.rs`) ─────────────────────────────────────────

/** @param {number} dur */
const tAxis = (dur) => Float32Array.from({ length: Math.max(1, Math.round(dur * SR)) }, (_, i) => i / SR);

/** @param {Float32Array} t @param {[number, number][]} pts */
function lerp(t, pts) {
  return t.map((x) => {
    if (x <= pts[0][0]) return pts[0][1];
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      if (x <= x1) return x1 <= x0 ? y1 : y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
    }
    return pts[pts.length - 1][1];
  });
}

/** @param {Float32Array} t @param {number} attack @param {number} decay */
function expdecay(t, attack, decay) {
  const a = Math.max(attack, 1e-4);
  const d = Math.max(decay, 1e-4);
  return t.map((x) => Math.min(1, Math.max(0, x / a)) * Math.exp(-Math.max(0, x - a) / d));
}

/** @param {Float32Array} t @param {number} rate @param {number} depth semitones @param {number} phase */
function vibrato(t, rate, depth, phase) {
  if (rate <= 0 || depth <= 0) return new Float32Array(t.length).fill(1);
  return t.map((x) => 2 ** ((depth * Math.sin(TAU * rate * x + phase)) / 12));
}

/** @param {Float32Array} t @param {number} depth @param {Rng} rng */
function jitter(t, depth, rng) {
  if (depth <= 0) return new Float32Array(t.length).fill(1);
  const raw = Float32Array.from({ length: t.length }, () => rng.standardNormal());
  const k = Math.max(1, Math.round((64 * SR) / TUNED_SR));
  // Centred moving average ("same" mode, zero padded), as numpy's convolve.
  const out = new Float32Array(t.length);
  const off = Math.floor((k - 1) / 2);
  let acc = 0;
  let hi = 0;
  for (let i = 0; i < t.length; i++) {
    const wantHi = Math.min(i + off + 1, t.length);
    while (hi < wantHi) acc += raw[hi++];
    const lo = Math.max(0, i + off + 1 - k);
    if (i > 0) for (let j = Math.max(0, i + off - k); j < lo; j++) acc -= raw[j];
    out[i] = 2 ** ((depth * (acc / k)) / 12);
  }
  return out;
}

/** @param {number} n @param {Rng} rng */
function pinkNoise(n, rng) {
  const a = 0.985 ** (TUNED_SR / SR);
  let leak = 0;
  const pink = Float32Array.from({ length: n }, () => (leak = a * leak + rng.standardNormal()));
  let peak = 1e-9;
  for (const v of pink) peak = Math.max(peak, Math.abs(v));
  return pink.map((v) => v / peak);
}

/** Attack time, sharper on snappy ducks. @param {Personality} p @param {number} dur @param {number} snappy */
function attackTime(p, dur, snappy) {
  const soft = 0.04 * dur;
  const sharp = 0.003 * dur;
  return Math.max(0.001, soft + (sharp - soft) * p.attackSharpness * snappy);
}

/**
 * Harmonic oscillator with vibrato, jitter, AM buzz and breath (microduck `voices.rs`).
 * @param {Personality} p @param {Float32Array} t @param {Float32Array} freq @param {Rng} rng
 * @param {number} amScale @param {number} breathScale
 */
function voice(p, t, freq, rng, amScale, breathScale) {
  const vib = vibrato(t, p.vibratoRate, p.vibratoDepth, rng.uniform(0, TAU));
  const jit = jitter(t, p.jitterDepth, rng);
  const weights = p.harmonics();
  const body = new Float32Array(t.length);
  let acc = 0;
  for (let i = 0; i < t.length; i++) {
    acc += freq[i] * vib[i] * jit[i];
    const phase = (TAU * acc) / SR;
    let v = 0;
    for (let n = 0; n < weights.length; n++) if (weights[n]) v += weights[n] * Math.sin((n + 1) * phase);
    body[i] = v;
  }
  const amD = p.amDepth * amScale * p.quackiness;
  if (amD > 0.01) for (let i = 0; i < t.length; i++) body[i] *= 1 - amD * (0.5 + 0.5 * Math.sin(TAU * p.amRate * t[i]));
  const breath = p.breath * breathScale;
  if (breath > 0) {
    const noise = pinkNoise(t.length, rng);
    for (let i = 0; i < t.length; i++) body[i] += breath * noise[i];
  }
  return body;
}

/** @param {Personality} p @param {Rng} rng @param {number} f0 @param {number} dur @param {[number, number][]} contour */
function chirpSyllable(p, rng, f0, dur, contour) {
  const t = tAxis(dur);
  const warble = vibrato(t, p.warbleHz, p.warbleDepth, rng.uniform(0, TAU));
  const shape = lerp(t, contour);
  const freq = shape.map((c, i) => f0 * c * warble[i]);
  const env = expdecay(t, attackTime(p, dur, 0.7), dur * 0.55);
  // Softer than greet: gut quackiness and brightness.
  const soft = Object.assign(Object.create(Personality.prototype), p, {
    quackiness: p.quackiness * 0.2,
    brightness: p.brightness * 0.5,
    formantGain: p.formantGain * 0.5,
  });
  return voice(soft, t, freq, rng, 1.0, 0.4).map((v, i) => v * env[i]);
}

/**
 * Microduck's mouth-trigger quack. Variants cycle four shapes: a rising blip,
 * a falling blip, a trill and a double "wek-wek".
 * @param {Personality} p @param {number} variant
 */
export function chirp(p, variant) {
  const rng = p.variantRng("chirp", variant);
  const shape = variant % 4;
  const f0 = p.pitch * (0.95 + 0.75 * rng.random());
  /** @type {Float32Array} */
  let sig;
  if (shape === 0) {
    const dur = (0.1 + 0.1 * rng.random()) / p.speed;
    const peak = 1.12 + 0.1 * rng.random();
    sig = chirpSyllable(p, rng, f0, dur, [[0, 0.88], [0.5 * dur, peak], [dur, 1.05]]);
  } else if (shape === 1) {
    const dur = (0.1 + 0.1 * rng.random()) / p.speed;
    const start = 1.12 + 0.1 * rng.random();
    sig = chirpSyllable(p, rng, f0, dur, [[0, start], [0.3 * dur, 1.0], [dur, 0.78]]);
  } else if (shape === 2) {
    const dur = (0.22 + 0.14 * rng.random()) / p.speed;
    const trill = Object.assign(Object.create(Personality.prototype), p, { warbleDepth: Math.max(p.warbleDepth, 0.7) * 1.6 });
    sig = chirpSyllable(trill, rng, f0, dur, [[0, 1.0], [0.5 * dur, 1.06], [dur, 0.94]]);
  } else {
    const dur = (0.08 + 0.05 * rng.random()) / p.speed;
    const a = chirpSyllable(p, rng, f0, dur, [[0, 0.95], [0.4 * dur, 1.1], [dur, 0.88]]);
    const gap = Math.floor(((0.03 + 0.03 * rng.random()) / p.speed) * SR);
    const durB = dur * 0.9;
    const f0B = f0 * (0.92 + 0.1 * rng.random());
    const b = chirpSyllable(p, rng, f0B, durB, [[0, 0.95], [0.4 * durB, 1.1], [durB, 0.88]]);
    sig = new Float32Array(a.length + gap + b.length);
    sig.set(a);
    sig.set(b, a.length + gap);
  }
  // Normalise to -6 dBFS.
  let peak = 1e-9;
  for (const v of sig) peak = Math.max(peak, Math.abs(v));
  const gain = 10 ** (-6 / 20) / peak;
  return sig.map((v) => v * gain);
}

/** @param {Personality} p @param {Rng} rng @param {number} durScale @param {number} f0Scale */
function greetSyllable(p, rng, durScale, f0Scale) {
  const dur = ((0.32 + 0.25 * rng.random()) * durScale) / p.speed;
  const t = tAxis(dur);
  const f0 = p.pitch * (0.9 + 0.15 * rng.random()) * f0Scale;
  // glide_bias flips the contour: positive ducks bend up, negative down.
  const bias = p.glideBias;
  const bend = 0.1 + 0.15 * p.pitchSpread;
  const start = f0 * (1 - bias * bend * 0.5);
  const mid = f0 * (1 + bias * bend);
  const end = f0 * (1 - bias * bend * 0.3) * (0.92 + 0.08 * rng.random());
  const freq = lerp(t, [[0, start], [0.18 * dur, mid], [dur, end]]);
  const env = expdecay(t, attackTime(p, dur, 0.5), dur * 0.7);
  return voice(p, t, freq, rng, 1.0, 1.0).map((v, i) => v * env[i]);
}

/**
 * Microduck's wake-up call: a quack, sometimes a double "wak-wak".
 * @param {Personality} p @param {number} variant
 */
export function greet(p, variant) {
  const rng = p.variantRng("greet", variant);
  let sig = greetSyllable(p, rng, 1.0, 1.0);
  if (rng.random() < 0.4) {
    const gap = Math.floor(((0.05 + 0.06 * rng.random()) / p.speed) * SR);
    const f0Scale = 0.95 + 0.06 * rng.random();
    const b = greetSyllable(p, rng, 0.8, f0Scale);
    const both = new Float32Array(sig.length + gap + b.length);
    both.set(sig);
    both.set(b, sig.length + gap);
    sig = both;
  }
  // Normalise to -3 dBFS.
  let peak = 1e-9;
  for (const v of sig) peak = Math.max(peak, Math.abs(v));
  const gain = 10 ** (-3 / 20) / peak;
  return sig.map((v) => v * gain);
}

/** The duck the coach speaks with. Any seed is a real Microduck voice. */
export const DUCK_SEED = 4;
const CHIRPS = 12;
/** @type {Float32Array[] | null} */
let bank = null;
const quacks = () => (bank ??= Array.from({ length: CHIRPS }, (_, v) => chirp(new Personality(DUCK_SEED), v)));
/** @type {Float32Array[] | null} */
let greets = null;

/** One of Microduck's wake-up calls, 24 kHz mono. */
export function duckGreet() {
  greets ??= Array.from({ length: 6 }, (_, v) => greet(new Personality(DUCK_SEED), v));
  return greets[Math.floor(Math.random() * greets.length)];
}

export const QUACK_RATE = SR;

// ── Speech in, quacks out ────────────────────────────────────────────────────

const FRAME = 240; // 10 ms

/** Turns a voice stream into quacks, chunk by chunk, keeping its length. */
export class Quacker {
  constructor() {
    this.reset();
  }

  reset() {
    /** Quack audio still to play into the next chunks. @type {Float32Array} */
    this.tail = new Float32Array(0);
    this.peak = 0.05;
    this.quietFrames = 99;
    this.sinceQuack = 99;
    this.busyFrames = 0;
    this.minEnv = 1;
    this.next = Math.floor(Math.random() * CHIRPS);
    /** Samples of an unfinished 10 ms frame carried to the next chunk. @type {number[]} */
    this.partial = [];
  }

  /**
   * @param {Int16Array} pcm 24 kHz mono voice
   * @returns {Int16Array} quacks, same length
   */
  process(pcm) {
    const out = new Float32Array(pcm.length);
    // Quack left over from the previous chunk.
    const carry = Math.min(this.tail.length, out.length);
    out.set(this.tail.subarray(0, carry));
    this.tail = this.tail.subarray(carry);
    // Frame energies, with the frame that straddles chunks completed first.
    const sample = (/** @type {number} */ k) => (k < 0 ? this.partial[this.partial.length + k] : pcm[k]);
    let i = -this.partial.length;
    for (; i + FRAME <= pcm.length; i += FRAME) {
      let sum = 0;
      for (let k = i; k < i + FRAME; k++) sum += (sample(k) / 32768) ** 2;
      this.frame(Math.sqrt(sum / FRAME), Math.max(0, i), out);
    }
    const rest = [];
    for (let k = i; k < pcm.length; k++) rest.push(sample(k));
    this.partial = rest;
    const res = new Int16Array(pcm.length);
    for (let k = 0; k < out.length; k++) res[k] = Math.max(-32767, Math.min(32767, Math.round(out[k] * 32767)));
    return res;
  }

  /** @param {number} env @param {number} at sample offset in this chunk @param {Float32Array} out */
  frame(env, at, out) {
    this.peak = Math.max(env, this.peak * 0.995);
    const voiced = env > Math.max(0.012, this.peak * 0.18);
    this.sinceQuack++;
    this.busyFrames--;
    if (!voiced) {
      this.quietFrames++;
      this.minEnv = Math.min(this.minEnv, env);
      return;
    }
    // A quack per burst of syllables: after a pause, on a clear rise in
    // loudness, or every third of a second through a long stretch; never
    // over the one still playing.
    const onset = this.quietFrames >= 6 || env > 1.6 * this.minEnv || this.sinceQuack >= 34;
    this.quietFrames = 0;
    this.minEnv = Math.min(this.minEnv, env);
    if (!onset || this.sinceQuack < 22 || this.busyFrames > 2) return;
    const bankNow = quacks();
    const quack = bankNow[this.next];
    this.next = (this.next + 1 + Math.floor(Math.random() * (CHIRPS - 1))) % CHIRPS;
    this.sinceQuack = 0;
    this.busyFrames = Math.ceil(quack.length / FRAME);
    this.minEnv = env;
    const gain = 0.7 + 0.3 * Math.min(1, env / this.peak);
    const room = Math.min(quack.length, out.length - at);
    for (let k = 0; k < room; k++) out[at + k] += quack[k] * gain;
    if (room < quack.length) {
      const rest = quack.subarray(room);
      const tail = new Float32Array(Math.max(this.tail.length, rest.length));
      tail.set(this.tail);
      for (let k = 0; k < rest.length; k++) tail[k] += rest[k] * gain;
      this.tail = tail;
    }
  }
}

/**
 * A whole clip in duck (the fillers).
 * @param {Int16Array} pcm
 */
export function quackify(pcm) {
  return new Quacker().process(pcm);
}
