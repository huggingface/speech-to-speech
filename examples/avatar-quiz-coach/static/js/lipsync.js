// @ts-check
/**
 * Lip-sync pieces shared by both avatar renderers.
 *
 * Mouth shapes are Oculus visemes. They come either from the speech backend
 * (`--enable_visemes`: phoneme cues in Microsoft viseme IDs, timed in seconds
 * from the start of the response audio) or from HeadAudio, which classifies
 * the outgoing audio in the browser.
 */

export const VOWELS = new Set(["aa", "E", "I", "O", "U"]);

// Peak strength per mouth shape. Real speakers fully form only lip closures
// (p, b, m, f, v) and open vowels; tongue sounds barely show on the lips, so
// driving every phoneme at full strength made the mouth flap.
export const VISEME_LEVEL = /** @type {Record<string, number>} */ ({
  PP: 0.9, FF: 0.8, aa: 0.7, O: 0.7, U: 0.65, E: 0.6, I: 0.55,
  CH: 0.5, SS: 0.4, TH: 0.4, RR: 0.35, DD: 0.3, nn: 0.3, kk: 0.3,
});

// Vowels also open the jaw; the visemes alone barely part the lips on some rigs.
export const JAW_LEVEL = /** @type {Record<string, number>} */ ({ aa: 0.3, O: 0.22, E: 0.15, U: 0.12, I: 0.1, CH: 0.05 });

// Microsoft viseme IDs (as emitted by the backend's STV stage) to Oculus names.
export const MS_TO_OCULUS = [
  "sil", "aa", "aa", "O", "E", "RR", "I", "U", "O", "aa", "O",
  "aa", "kk", "RR", "nn", "SS", "CH", "TH", "FF", "DD", "kk", "PP",
];

// Oculus visemes as ARKit blendshape mixes, from TalkingHead's
// blender/build-visemes-from-arkit.py (MIT), for avatars that only have ARKit.
export const VISEME_ARKIT = /** @type {Record<string, Record<string, number>>} */ ({
  aa: { jawOpen: 0.6 },
  E: { mouthPressLeft: 0.8, mouthPressRight: 0.8, mouthDimpleLeft: 1, mouthDimpleRight: 1, jawOpen: 0.3 },
  I: { mouthPressLeft: 0.6, mouthPressRight: 0.6, mouthDimpleLeft: 0.6, mouthDimpleRight: 0.6, jawOpen: 0.2 },
  O: { mouthPucker: 1, jawForward: 0.6, jawOpen: 0.2 },
  U: { mouthFunnel: 1 },
  PP: { mouthRollLower: 0.8, mouthRollUpper: 0.8, mouthUpperUpLeft: 0.3, mouthUpperUpRight: 0.3 },
  FF: {
    mouthPucker: 1, mouthShrugUpper: 1, mouthLowerDownLeft: 0.2, mouthLowerDownRight: 0.2,
    mouthDimpleLeft: 1, mouthDimpleRight: 1, mouthRollLower: 1,
  },
  DD: { mouthPressLeft: 0.8, mouthPressRight: 0.8, mouthFunnel: 0.5, jawOpen: 0.2 },
  SS: { mouthPressLeft: 0.8, mouthPressRight: 0.8, mouthLowerDownLeft: 0.5, mouthLowerDownRight: 0.5, jawOpen: 0.1 },
  TH: { mouthRollUpper: 0.6, jawOpen: 0.2, tongueOut: 0.4 },
  CH: { mouthPucker: 0.5, jawOpen: 0.2 },
  RR: { mouthPucker: 0.5, jawOpen: 0.2 },
  kk: {
    mouthLowerDownLeft: 0.4, mouthLowerDownRight: 0.4, mouthDimpleLeft: 0.3, mouthDimpleRight: 0.3,
    mouthFunnel: 0.3, mouthPucker: 0.3, jawOpen: 0.15,
  },
  nn: {
    mouthLowerDownLeft: 0.4, mouthLowerDownRight: 0.4, mouthDimpleLeft: 0.3, mouthDimpleRight: 0.3,
    mouthFunnel: 0.3, mouthPucker: 0.3, jawOpen: 0.15, tongueOut: 0.2,
  },
});

// `?lipsync=headaudio` ignores server visemes (for comparing the two sources).
/** Silence between two coach lines that would otherwise run together, e.g. the verdict and the next question (ms). */
export const TURN_GAP_MS = 500;

/**
 * Silence to play before a new line so it starts TURN_GAP_MS after the queued
 * audio ends; none when the voice has already been quiet that long.
 * @param {number} start when the line would start (audio clock, ms)
 * @param {number} cursor when the queued audio runs out (same clock; 0 when nothing has been queued)
 */
export function turnGap(start, cursor) {
  return cursor ? Math.max(0, cursor + TURN_GAP_MS - start) : 0;
}

export const FORCE_HEADAUDIO = new URLSearchParams(location.search).get("lipsync") === "headaudio";

/**
 * The phoneme recognizer emits each phoneme as a 20-40 ms blip with a
 * silence cue between neighbours, so played as-is the mouth snaps shut
 * between sounds of the same word. Drop the short in-word silences and hold
 * each shape until the next one starts (at most 300 ms); real pauses of
 * 150 ms or more still close the mouth.
 * @param {{ viseme: number, start_s: number, end_s: number }[]} cues
 */
export function coarticulate(cues) {
  const kept = [...cues]
    .sort((a, b) => a.start_s - b.start_s)
    .filter((c) => c.viseme !== 0 || c.end_s - c.start_s >= 0.15)
    .map((c) => ({ ...c }));
  for (let i = 0; i < kept.length; i++) {
    const c = kept[i];
    if (c.viseme === 0) continue;
    const next = kept[i + 1];
    const until = next ? next.start_s : c.start_s + 0.12;
    c.end_s = Math.max(c.end_s, Math.min(until, c.start_s + 0.3));
  }
  return kept.filter((c) => c.viseme !== 0);
}

/**
 * Peak level and jaw opening for one cue, with the envelope TalkingHead uses
 * (rise from 2/3 of a duration before, peak mid-cue, fall by 1.5 durations).
 * @param {{ viseme: number, start_s: number, end_s: number }} cue
 * @param {{ lipGain: number, jawGain: number }} coach
 */
export function cueShape(cue, coach) {
  const v = MS_TO_OCULUS[cue.viseme];
  if (!v || v === "sil") return null;
  const d = Math.max(40, (cue.end_s - cue.start_s) * 1000);
  // Short vowels open the jaw less than long ones.
  const length = VOWELS.has(v) ? Math.min(1, Math.max(0.6, d / 120)) : 1;
  return {
    v,
    d,
    level: Math.min(1, (VISEME_LEVEL[v] ?? 0.4) * coach.lipGain * length),
    jaw: (JAW_LEVEL[v] ?? 0) * coach.jawGain * length,
  };
}

/** Linear 24 kHz -> AudioContext-rate resampler that keeps state across chunks. */
export class Upsampler {
  /** @param {number} inRate @param {number} outRate */
  constructor(inRate, outRate) {
    this.step = inRate / outRate;
    this.reset();
  }
  reset() {
    this.pos = 0;
    this.prev = 0;
  }
  /** @param {Int16Array} input */
  process(input) {
    const n = input.length;
    if (!n) return new Int16Array(0);
    const out = new Int16Array(Math.ceil((n - this.pos) / this.step) + 1);
    let p = this.pos;
    let k = 0;
    while (p < n) {
      const i = Math.floor(p);
      const a = i === 0 ? this.prev : input[i - 1];
      out[k++] = a + (input[i] - a) * (p - i);
      p += this.step;
    }
    this.pos = p - n;
    this.prev = input[n - 1];
    return out.subarray(0, k);
  }
}
