// @ts-check

const DUCK_GAIN = 0.12;
const ATTACK_SECONDS = 0.015;
const RELEASE_SECONDS = 0.08;

/**
 * Reversible output-gain state machine for pre-confirmation barge-in.
 *
 * A speech candidate only attenuates playback. The standard confirmed
 * speech_started event may hard-mute it, and speech_stopped restores normal
 * gain. Candidate ids keep a late reject from releasing a newer candidate.
 */
export class ReversibleAudioDucker {
  /** @param {GainNode} gainNode @param {AudioContext} audioContext */
  constructor(gainNode, audioContext) {
    this._gainNode = gainNode;
    this._ctx = audioContext;
    this._candidateId = "";
    this._confirmed = false;
  }

  /** @param {string} candidateId @param {boolean} [interruptResponse] */
  candidateStarted(candidateId, interruptResponse = true) {
    if (!candidateId || !interruptResponse) return;
    this._candidateId = candidateId;
    this._confirmed = false;
    this._ramp(DUCK_GAIN, ATTACK_SECONDS);
  }

  /** @param {string} candidateId */
  candidateRejected(candidateId) {
    if (!candidateId || candidateId !== this._candidateId || this._confirmed) return;
    this._candidateId = "";
    this._ramp(1, RELEASE_SECONDS);
  }

  /** @param {boolean} [interruptResponse] */
  speechStarted(interruptResponse = true) {
    this._candidateId = "";
    this._confirmed = interruptResponse;
    this._ramp(interruptResponse ? 0 : 1, interruptResponse ? ATTACK_SECONDS : RELEASE_SECONDS);
  }

  speechStopped() {
    this._candidateId = "";
    this._confirmed = false;
    this._ramp(1, RELEASE_SECONDS);
  }

  reset() {
    this._candidateId = "";
    this._confirmed = false;
    this._ramp(1, 0);
  }

  /** @param {number} target @param {number} durationSeconds */
  _ramp(target, durationSeconds) {
    const param = this._gainNode.gain;
    const now = this._ctx.currentTime;
    param.cancelScheduledValues(now);
    param.setValueAtTime(param.value, now);
    param.linearRampToValueAtTime(target, now + durationSeconds);
  }
}
