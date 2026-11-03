import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import { modelReadiness } from "../ws/model-readiness.js";
import { S2sRealtimeClient } from "../s2s-realtime-client.js";

globalThis.localStorage = { getItem: () => null };

function fixture(connect = async () => {}, waitForModels = true) {
  const calls = [];
  let session;
  class Session extends EventEmitter {
    constructor(agent, options) {
      super(); this.transport = options.transport; session = this;
    }
    async connect() { await connect(this.transport); }
    sendMessage(message) { calls.push(["greeting", message]); }
    sendAudio(buffer) { calls.push(["audio", buffer]); }
    mute() {}
    close() { calls.push(["closed"]); }
  }
  globalThis.OpenAIAgentsRealtime = {
    OpenAIRealtimeWebSocket: EventEmitter,
    RealtimeSession: Session,
  };
  const client = new S2sRealtimeClient({
    waitForModels, transport: "websocket", directUrl: "ws://unused", startupGreeting: "Hello",
    micStream: { getAudioTracks: () => [{}] }, localVad: false,
  });
  client._setupAudio = async () => {};
  client._buildAgent = () => ({});
  client._sessionConfig = () => ({});
  client._clearPlayback = () => {};
  const statuses = [];
  client.addEventListener("status", event => statuses.push(event.detail.status));
  return { client, calls, statuses, event: event => session.transport.emit("*", event) };
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const ready = { type: "speech_to_speech.models.ready" };

test("SDK connection alone cannot send microphone audio or the greeting", async () => {
  const f = fixture();
  const connecting = f.client.connect();
  await tick();
  assert.equal(f.client.status, "preparing");
  f.client._onMicChunk(new ArrayBuffer(16));
  assert.deepEqual(f.calls, []);
  f.event({ type: "speech_to_speech.models.preparing" });
  assert.equal(f.client.status, "preparing");
  f.event(ready);
  await connecting;
  assert.equal(f.client.status, "connected");
  assert.deepEqual(f.calls, [["greeting", "Hello"]]);
  f.client._onMicChunk(new ArrayBuffer(16));
  assert.equal(f.calls[1][0], "audio");
  await f.client.close();
});

test("ready arriving before SDK connect resolves is retained", async () => {
  const f = fixture(async transport => { transport.emit("*", ready); await tick(); });
  await f.client.connect();
  assert.equal(f.client.status, "connected");
  assert.deepEqual(f.calls, [["greeting", "Hello"]]);
  await f.client.close();
});

test("cancelling preparation rejects the pending start without a greeting", async () => {
  const f = fixture();
  const connecting = f.client.connect();
  const rejected = assert.rejects(connecting, { code: "aborted" });
  await tick();
  await f.client.close();
  await rejected;
  assert.equal(f.client.status, "closed");
  assert.ok(f.calls.every(call => call[0] === "closed"));
  f.client._onMicChunk(new ArrayBuffer(16));
  assert.ok(f.calls.every(call => call[0] === "closed"));
});

test("preparation failure closes the client and preserves the useful error", async () => {
  const f = fixture();
  const connecting = f.client.connect();
  const rejected = assert.rejects(connecting, /GPU warmup failed/);
  await tick();
  f.event({ type: "speech_to_speech.models.error", error: { message: "GPU warmup failed" } });
  await rejected;
  assert.equal(f.client.status, "closed");
  assert.ok(f.calls.every(call => call[0] === "closed"));
});

test("readiness timeout and late events cannot restart a failed gate", async () => {
  const gate = modelReadiness({ timeoutMs: 5 });
  await assert.rejects(gate.promise, /timed out/);
  gate.event(ready);
  await assert.rejects(gate.promise, /timed out/);
});


test("ordinary demos connect and greet without any custom readiness events", async () => {
  const f = fixture(undefined, false);
  await f.client.connect();
  assert.equal(f.client.status, "connected");
  assert.ok(!f.statuses.includes("preparing"));
  assert.deepEqual(f.calls, [["greeting", "Hello"]]);
  await f.client.close();
});
