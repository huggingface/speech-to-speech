import assert from "node:assert/strict";
import test from "node:test";
import { LocalVad, SpeechCandidate } from "../ws/local-vad.js";
import { S2sRealtimeClient } from "../s2s-realtime-client.js";

test("speech candidates react to speech, ignore noise, and release after 64 ms of silence", () => {
  const changes = [];
  const candidate = new SpeechCandidate((active) => changes.push(active));
  candidate.update(0.1, 0);
  candidate.update(0.49, 32);
  assert.deepEqual(changes, []);
  candidate.update(0.9, 64);
  candidate.update(0.8, 96);
  candidate.update(0.4, 128);
  candidate.update(0.1, 160);
  assert.deepEqual(changes, [true]);
  candidate.update(0.1, 192);
  assert.deepEqual(changes, [true, false]);
  candidate.update(0.9, 224);
  candidate.checkStale(723);
  assert.equal(candidate.active, true);
  candidate.checkStale(724);
  assert.equal(candidate.active, false);
  candidate.update(0.9, 800);
  candidate.update(NaN, 832);
  assert.equal(candidate.active, false);
});

for (const transport of ["websocket", "webrtc"]) {
  test(`${transport}: local speech ducks reversibly without canceling or dropping buffered audio`, () => {
    const gains = [];
    const messages = [];
    const client = new S2sRealtimeClient({ transport });
    client._ctx = { currentTime: 1 };
    client._outputGain = { gain: {
      cancelScheduledValues() {},
      setTargetAtTime(value) { gains.push(value); },
    } };
    client._transport = { sendEvent(event) { messages.push(event); } };
    client._session = { interrupt() { assert.fail("local candidates must not cancel the response"); }, mute() {} };
    client._playbackNode = { port: { postMessage(event) { messages.push(event); } } };
    client._resetPlaybackBuffer("reply");
    client._pendingPlayback.push({ samples: new Float32Array(2400), key: "item" });
    client._pendingPlaybackSamples = 2400;
    client._setLocalSpeech(true);
    assert.equal(gains.at(-1), 0.25);
    assert.equal(client._pendingPlaybackSamples, 2400);
    assert.deepEqual(messages, []);
    client._setLocalSpeech(false);
    assert.equal(gains.at(-1), 1);
    client._setLocalSpeech(true);
    client.setMuted(true);
    assert.equal(gains.at(-1), 1);
    client._setLocalSpeech(true);
    assert.equal(gains.at(-1), 1, "muted mic must not lower playback");
    client.setMuted(false);
    client._setLocalSpeech(true);
    client._setStatus("error");
    assert.equal(gains.at(-1), 1);
  });
}

test("VAD uses the conversation's stream/context, respects mute, and cleans up", async () => {
  let options;
  let destroyed = 0;
  let enabled = true;
  const stream = { getTracks() { assert.fail("VAD must not stop the mic"); } };
  const audioContext = {};
  const changes = [];
  const vad = new LocalVad({
    stream, audioContext,
    isEnabled: () => enabled,
    onChange: (active) => changes.push(active),
    load: async () => ({ async new(opts) {
      options = opts;
      return { async destroy() { destroyed++; } };
    } }),
  });
  await vad.start();
  assert.equal(options.audioContext, audioContext);
  assert.equal(await options.getStream(), stream);
  await options.pauseStream(stream);
  assert.equal(await options.resumeStream(stream), stream);
  const ort = { env: { wasm: {} } };
  options.ortConfig(ort);
  assert.equal(ort.env.wasm.numThreads, 1);
  options.onFrameProcessed({ isSpeech: 0.9 });
  enabled = false;
  options.onFrameProcessed({ isSpeech: 0.9 });
  enabled = true;
  options.onFrameProcessed({ isSpeech: 0.9 });
  await vad.close();
  options.onFrameProcessed({ isSpeech: 0.9 });
  assert.deepEqual(changes, [true, false, true, false]);
  assert.equal(destroyed, 1);
});

test("closing during model loading destroys the late instance and cannot duck playback", async () => {
  let finish;
  let options;
  let destroyed = false;
  const changes = [];
  const vad = new LocalVad({
    audioContext: {}, stream: {}, isEnabled: () => true,
    onChange: (active) => changes.push(active),
    load: async () => ({ new(opts) {
      options = opts;
      return new Promise((resolve) => { finish = resolve; });
    } }),
  });
  const starting = vad.start();
  await Promise.resolve();
  await vad.close();
  options.onFrameProcessed({ isSpeech: 0.9 });
  finish({ async destroy() { destroyed = true; } });
  await starting;
  assert.equal(destroyed, true);
  assert.deepEqual(changes, []);
});

test("failed model loading leaves server VAD available", async (t) => {
  const warnings = [];
  t.mock.method(console, "warn", (...args) => warnings.push(args));
  const vad = new LocalVad({
    audioContext: {}, stream: {}, isEnabled: () => true, onChange() {},
    load: async () => { throw new Error("model unavailable"); },
  });
  await vad.start();
  assert.equal(vad.instance, null);
  assert.equal(warnings.length, 1);
  await vad.close();
});
