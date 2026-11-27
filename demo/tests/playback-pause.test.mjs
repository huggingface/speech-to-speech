import assert from "node:assert/strict";
import test from "node:test";

import { S2sRealtimeClient } from "../s2s-realtime-client.js";

import { worklet } from "./playback-helpers.mjs";

const RATE = 48_000;
const ms = (n) => Math.round((n * RATE) / 1000);

/** Input at 48 kHz so one input sample is one output sample. */
function player() {
  const output = worklet(() => {}, RATE);
  output.send({ kind: "config", inputRate: RATE });
  return output;
}

const tone = (count, value) => new Float32Array(count).fill(value);

test("pause holds the queue and resume plays on from the same sample", () => {
  const output = player();
  const ramp = Float32Array.from({ length: ms(500) }, (_, i) => 0.1 + (0.5 * i) / ms(500));
  output.send({ kind: "audio", samples: ramp });
  output.render(ms(100));
  output.send({ kind: "pause" });
  const paused = output.render(ms(200));
  assert.ok(paused.slice(32).every((v) => v === 0), "silent while paused");
  output.send({ kind: "resume" });
  const resumed = output.render(ms(500));
  // A continuous sound has no gap between words: nothing is replayed, and
  // playback goes on from where the 32-sample fade-out left it (the slack is
  // the fade-in and the underrun ramp-out at the end).
  const played = ms(100) + 32;
  assert.equal(resumed[100], ramp[played + 100]);
  const audible = resumed.filter((v) => Math.abs(v) > 0.05).length;
  assert.ok(Math.abs(audible - (ramp.length - played)) <= 128, `rest of the ramp, once (${audible} samples)`);
});

test("resume replays from the last gap between words", () => {
  const output = player();
  output.send({ kind: "audio", samples: tone(ms(300), 0.4) });
  output.send({ kind: "audio", samples: tone(ms(80), 0) });
  output.send({ kind: "audio", samples: tone(ms(400), -0.4) });
  output.render(ms(300 + 80 + 200));
  output.send({ kind: "pause" });
  output.render(ms(100));
  output.send({ kind: "resume" });
  const resumed = output.render(ms(700));
  // 10 to 20 ms of the gap (windows do not line up with its edges), then
  // the second word from its start.
  const word = resumed.filter((v) => v === Math.fround(-0.4)).length;
  assert.ok(Math.abs(word - ms(400)) <= 64, `second word replayed whole (${word} samples)`);
  const onset = resumed.findIndex((v) => v === Math.fround(-0.4));
  assert.ok(onset >= ms(10) && onset <= ms(20) + 32, `replay starts in the gap (${onset})`);
});

test("a clear ends the pause and a new reply plays", () => {
  const output = player();
  output.send({ kind: "audio", samples: tone(ms(300), 0.4) });
  output.render(ms(50));
  output.send({ kind: "pause" });
  output.render(ms(50));
  output.send({ kind: "clear" });
  output.send({ kind: "audio", samples: tone(ms(100), -0.2) });
  const next = output.render(ms(100));
  assert.ok(next.slice(64).every((v) => v === Math.fround(-0.2)));
});

test("duck lowers the volume and resume restores it", () => {
  const output = player();
  output.send({ kind: "audio", samples: tone(ms(300), 0.4) });
  output.render(ms(50));
  output.send({ kind: "gain", value: 0.25 });
  assert.ok(output.render(ms(50)).every((v) => Math.abs(v - 0.1) < 1e-6));
  output.send({ kind: "resume" });
  assert.ok(output.render(ms(50)).every((v) => Math.abs(v - 0.4) < 1e-6));
});

test("the client pauses on a pending interruption and resumes only when it was not one", () => {
  const messages = [];
  const client = new S2sRealtimeClient({ transport: "websocket", directUrl: "ws://unused" });
  client._playbackNode = { port: { postMessage(message) { messages.push(message); } } };
  const pending = (playback) => client._onTransportEvent({
    type: "speech_to_speech.interruption.pending", event_id: "e", audio_start_ms: 0, playback,
  });
  const resolved = (interrupted) => client._onTransportEvent({
    type: "speech_to_speech.interruption.resolved", event_id: "e", interrupted, transcript: "",
  });
  pending("pause");
  resolved(false);
  pending("duck");
  resolved(true);
  pending("continue");
  assert.deepEqual(messages, [{ kind: "pause" }, { kind: "resume" }, { kind: "gain", value: 0.25 }]);
});

test("the client opts in to interruption events", () => {
  const client = new S2sRealtimeClient({ transport: "websocket", directUrl: "ws://unused" });
  assert.ok(client._sessionConfig().providerData.extensions.includes("speech_to_speech.interruption"));
});
