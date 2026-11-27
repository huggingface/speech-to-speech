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
  const played = ms(100) + 32;
  assert.equal(resumed[100], ramp[played + 100]);
  const audible = resumed.filter((v) => Math.abs(v) > 0.05).length;
  assert.ok(Math.abs(audible - (ramp.length - played)) <= 128, `rest of the ramp, once (${audible} samples)`);
});

test("resume replays from the last gap between words and survives a second pause mid-replay", () => {
  const output = player();
  output.send({ kind: "audio", samples: tone(ms(300), 0.4) });
  output.send({ kind: "audio", samples: tone(ms(80), 0) });
  output.send({ kind: "audio", samples: tone(ms(400), -0.4) });
  output.render(ms(300 + 80 + 200));
  output.send({ kind: "pause" });
  output.render(ms(100));
  output.send({ kind: "resume" });
  // Pause again halfway through replaying the first 200 ms of the second word.
  output.render(ms(100));
  output.send({ kind: "pause" });
  output.render(ms(50));
  output.send({ kind: "resume" });
  const resumed = output.render(ms(700));
  const word = resumed.filter((v) => v === Math.fround(-0.4)).length;
  assert.ok(Math.abs(word - ms(400)) <= 96, `second word replayed whole without skipping (${word} samples)`);
  const onset = resumed.findIndex((v) => v === Math.fround(-0.4));
  assert.ok(onset >= ms(10) && onset <= ms(20) + 32, `replay starts in the gap (${onset})`);
});

test("resume finds a word gap at the start of the 600 ms history window", () => {
  const output = player();
  output.send({ kind: "audio", samples: tone(ms(100), 0) });
  output.send({ kind: "audio", samples: tone(ms(500), -0.4) });
  output.send({ kind: "audio", samples: tone(ms(100), 0.2) });
  output.render(ms(600));
  output.send({ kind: "pause" });
  output.render(ms(50));
  output.send({ kind: "resume" });
  const resumed = output.render(ms(700));
  const replayed = resumed.filter((v) => v === Math.fround(-0.4)).length;
  assert.ok(Math.abs(replayed - ms(500)) <= 64, `full 500 ms word replayed (${replayed} samples)`);
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

test("the client opts in, pauses on pending, and resumes only when not interrupted", () => {
  const messages = [];
  const client = new S2sRealtimeClient({ transport: "websocket", directUrl: "ws://unused" });
  assert.ok(client._sessionConfig().providerData.extensions.includes("speech_to_speech.interruption"));
  client._playbackNode = { port: { postMessage(message) { messages.push(message); } } };
  client._onTransportEvent({ type: "speech_to_speech.interruption.pending", event_id: "e", audio_start_ms: 0 });
  client._onTransportEvent({ type: "speech_to_speech.interruption.resolved", event_id: "e", interrupted: false, transcript: "" });
  client._onTransportEvent({ type: "speech_to_speech.interruption.pending", event_id: "e", audio_start_ms: 0 });
  client._onTransportEvent({ type: "speech_to_speech.interruption.resolved", event_id: "e", interrupted: true, transcript: "Stop." });
  assert.deepEqual(messages, [{ kind: "pause" }, { kind: "resume" }, { kind: "pause" }]);
});
