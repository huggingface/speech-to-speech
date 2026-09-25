import assert from "node:assert/strict";
import test from "node:test";
import * as realtime from "@openai/agents/realtime";
import { S2sRealtimeClient } from "../s2s-realtime-client.js";
import { worklet } from "./playback-helpers.mjs";

// Exercise the pinned SDK's parsing, event ordering, session, and transport.
// Only the socket and audio device are deterministic in-process stand-ins.
async function fixture(bufferMs = 1200, sampleRate = 48_000) {
  const sent = [];
  const socket = new class extends EventTarget {
    send(data) { sent.push(JSON.parse(data)); }
    close() { this.dispatchEvent(new Event("close")); }
    receive(data) { this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify(data) })); }
  }();
  globalThis.OpenAIAgentsRealtime = {
    ...realtime,
    OpenAIRealtimeWebSocket: class extends realtime.OpenAIRealtimeWebSocket {
      constructor(options) {
        super({ ...options, createWebSocket: () => socket, skipOpenEventListeners: true });
      }
    },
  };
  const client = new S2sRealtimeClient({
    transport: "websocket", directUrl: "ws://test", playbackBufferMs: bufferMs,
    micStream: { getAudioTracks: () => [{}] }, voice: "coral", instructions: "Test.",
  });
  const acknowledgements = [];
  const output = worklet((message) => acknowledgements.push(message), sampleRate);
  client._setupAudio = async () => {
    client._playbackNode = { port: { postMessage(message) { output.send(message); } } };
  };
  await client.connect();
  const receive = (event) => socket.receive({ event_id: "event-test", ...event });
  receive({ type: "session.updated", session: {
    id: "session-test", object: "realtime.session", model: "s2s-local",
    audio: { input: { turn_detection: { type: "server_vad", interrupt_response: true } },
      output: { format: { type: "audio/pcm", rate: 24000 } } },
  } });
  return {
    client, output, sent, receive,
    start(id = "a") { receive({ type: "response.created", response: { id, status: "in_progress", output: [] } }); },
    audio(count, id = "a", itemId = "item-a", contentIndex = 2) {
      receive({ type: "response.output_audio.delta", response_id: id, item_id: itemId,
        output_index: 0, content_index: contentIndex, delta: Buffer.alloc(count * 2, 32).toString("base64") });
    },
    audioDone(id = "a", itemId = "item-a", contentIndex = 2) {
      receive({ type: "response.output_audio.done", response_id: id, item_id: itemId,
        output_index: 0, content_index: contentIndex });
    },
    done(id = "a", status = "completed") {
      receive({ type: "response.done", response: { id, status, output: [] } });
    },
    interrupt(manual = false) {
      if (manual) client._session.interrupt();
      else receive({ type: "input_audio_buffer.speech_started", item_id: "user", audio_start_ms: 0 });
    },
    acknowledge() {
      for (const message of acknowledgements.splice(0)) client._onPlaybackMessage(message);
    },
    truncations() { return sent.filter((event) => event.type === "conversation.item.truncate"); },
  };
}

for (const manual of [false, true]) {
  for (const audioDone of [false, true]) {
    test(`SDK truncates pending audio to zero (manual=${manual}, audioDone=${audioDone})`, async (t) => {
      const f = await fixture();
      t.after(() => f.client.close());
      f.start();
      f.audio(24000); // 1000 ms: below the 1200 ms startup gate.
      if (audioDone) f.audioDone();
      const interruptedAt = Date.now() + 800;
      t.mock.method(Date, "now", () => interruptedAt);
      f.output.render(38400); // 800 ms passes with no rendered response audio.
      f.interrupt(manual);
      f.acknowledge();
      assert.deepEqual(f.truncations(), [{
        type: "conversation.item.truncate", item_id: "item-a", content_index: 2, audio_end_ms: 0,
      }]);
      assert.equal(f.sent.filter((e) => e.type === "response.cancel").length, manual ? 1 : 0);
    });
  }
}

for (const sampleRate of [48000, 44100, 16000]) {
  test(`SDK truncates a completed response tail at rendered time (${sampleRate} Hz)`, async (t) => {
    const f = await fixture(1200, sampleRate);
    t.after(() => f.client.close());
    f.start();
    f.audio(14400);
    f.audio(14400);
    f.output.render(sampleRate / 4); // 250 ms after startup release.
    f.audioDone();
    f.done();
    f.interrupt();
    f.acknowledge();
    const [event] = f.truncations();
    assert.equal(f.truncations().length, 1);
    assert.equal(event.item_id, "item-a");
    assert.equal(event.content_index, 2);
    assert.ok(Math.abs(event.audio_end_ms - 250) <= 1, JSON.stringify(event));
  });
}

test("SDK excludes underrun silence from the rendered position", async (t) => {
  const f = await fixture(0);
  t.after(() => f.client.close());
  f.start();
  f.audio(2400);
  f.output.render(48000); // Only 100 ms of PCM, followed by silence.
  f.audio(2400);
  f.interrupt();
  f.acknowledge();
  assert.equal(f.truncations()[0].audio_end_ms, 100);
});

test("SDK keeps identities across queued responses and delayed clear acknowledgements", async (t) => {
  const f = await fixture(100);
  t.after(() => f.client.close());
  f.start();
  f.audio(4800);
  f.done();
  f.output.render(2400); // 50 ms of A.
  f.start("b");
  f.audio(1000, "b", "item-b", 3);
  f.interrupt();
  f.start("c");
  f.audio(2400, "c", "item-c", 1);
  f.acknowledge();
  assert.deepEqual(f.truncations().map((e) => [e.item_id, e.content_index, e.audio_end_ms]), [
    ["item-a", 2, 50], ["item-b", 3, 0],
  ]);
  assert.ok(f.output.render(128).some((v) => v > 0));
});

test("fully rendered completed audio is not truncated again", async (t) => {
  const f = await fixture(0);
  t.after(() => f.client.close());
  f.start();
  f.audio(2400);
  f.audioDone();
  f.done();
  f.output.render(4800);
  f.interrupt();
  f.acknowledge();
  assert.deepEqual(f.truncations(), []);
});

test("short response flushed on done still truncates to zero before rendering", async (t) => {
  const f = await fixture();
  t.after(() => f.client.close());
  f.start();
  f.audio(2400);
  f.audioDone();
  f.done();
  f.interrupt();
  f.acknowledge();
  assert.equal(f.truncations().length, 1);
  assert.equal(f.truncations()[0].audio_end_ms, 0);
});

test("cancellation before the speech event retains the render position", async (t) => {
  const f = await fixture(0);
  t.after(() => f.client.close());
  f.start();
  f.audio(4800);
  f.output.render(2400);
  f.done("a", "cancelled");
  f.interrupt();
  f.acknowledge();
  assert.equal(f.truncations().length, 1);
  assert.equal(f.truncations()[0].audio_end_ms, 50);
});

for (const bufferMs of [0, 100]) {
  for (const status of ["failed", "cancelled"]) {
    test(`released ${status} response preserves earlier playback (${bufferMs} ms buffer)`, async (t) => {
      const f = await fixture(bufferMs);
      t.after(() => f.client.close());
      f.start("a");
      f.audio(24000, "a", "item-a", 0);
      f.audioDone("a", "item-a", 0);
      f.done("a");
      f.output.render(4800); // A has rendered 100 ms, with 900 ms still queued.

      f.start("b");
      f.audio(2400, "b", "item-b", 0); // B crosses the gate behind A.
      f.done("b", status);
      f.audio(2400, "b", "item-b", 0); // Late audio must remain ignored.
      f.acknowledge();
      assert.deepEqual(f.truncations(), [], "terminal status is not a user interruption");
      assert.ok(f.output.render(43200).every((v) => v > 0), "A's entire tail must survive");

      // A real interruption still clears B, retaining its identity after done.
      f.interrupt();
      f.acknowledge();
      assert.deepEqual(f.truncations(), [{
        type: "conversation.item.truncate", item_id: "item-b", content_index: 0, audio_end_ms: 0,
      }]);
      assert.ok(f.output.render(128).slice(64).every((v) => v === 0));
    });
  }
}

test("SDK preserves terminal server timings through its WebSocket parser", async (t) => {
  const f = await fixture();
  t.after(() => f.client.close());
  const timing = {
    version: 1, turn_id: "turn_1", turn_revision: 0, response_key: "key-1", status: "completed",
    stt_s: 0.181284, llm_s: 1.24, tts_ttfa_s: 0.12, e2e_s: 1.61, mlx_lock_wait_s: 0,
  };
  const finished = [];
  f.client.addEventListener("response-finished", (event) => finished.push(event.detail));
  f.start();
  f.receive({ type: "response.done", response: {
    id: "a", status: "completed", output: [],
    metadata: { "speech_to_speech.turn_latency": JSON.stringify(timing) },
  } });
  assert.equal(finished.length, 1);
  assert.deepEqual(finished[0].latency, timing);
});
