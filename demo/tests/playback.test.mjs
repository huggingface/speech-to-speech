import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import test from "node:test";

import { S2sRealtimeClient, DEFAULT_PLAYBACK_BUFFER_MS, normalizePlaybackBufferMs } from "../s2s-realtime-client.js";

import { worklet } from "./playback-helpers.mjs";

function fixture(playbackBufferMs = 100, transport = "websocket") {
  const output = worklet();
  const messages = [];
  const client = new S2sRealtimeClient({ transport, playbackBufferMs, directUrl: "ws://unused" });
  client._playbackNode = { port: { postMessage(message, transfer = []) {
    const copy = structuredClone(message, { transfer });
    messages.push(copy);
    output.send(copy);
  } } };
  const start = (id) => client._onTransportEvent({ type: "response.created", response: { id } });
  const done = (id, status = "completed") => client._onTransportEvent({
    type: "response.done", response: { id, status },
  });
  const audio = (id, count, value = 8192) => {
    const data = new ArrayBuffer(count * 2);
    const view = new DataView(data);
    for (let i = 0; i < count; i++) view.setInt16(i * 2, value, true);
    client._onAudio({ data, responseId: id });
  };
  return { client, output, messages, start, done, audio,
    interrupt() { client._interruptPlayback(); },
    released() { return messages.filter((m) => m.kind === "audio").flatMap((m) => [...m.samples]); },
  };
}

test("clear followed immediately by new audio keeps the new response playing", () => {
  const output = worklet();
  output.send({ kind: "audio", samples: new Float32Array(2400).fill(0.25) });
  output.render(128);
  output.send({ kind: "clear" });
  output.send({ kind: "audio", samples: new Float32Array(2400).fill(-0.25) });
  const rendered = output.render(256);
  assert.ok(rendered.slice(64).every((value) => value === -0.25));
});

test("unfinished first chunk stays silent and empty chunks do not open the gate", () => {
  const f = fixture();
  f.start("a");
  f.audio("a", 2399);
  f.audio("a", 0);
  assert.ok(f.output.render(4800).every((v) => v === 0));
  assert.equal(f.released().length, 0);
});

test("exact sample threshold releases chunks in order and later chunks pass through", () => {
  const f = fixture();
  f.start("a");
  f.audio("a", 2399, -16384);
  assert.equal(f.released().length, 0);
  f.audio("a", 1, 32767);
  assert.deepEqual(f.released(), [...Array(2399).fill(-0.5), 1]);
  f.audio("a", 3, 0);
  assert.deepEqual(f.released().slice(-4), [1, 0, 0, 0]);
});

test("short successful completion releases every sample and reports audible", () => {
  const f = fixture();
  const finished = [];
  f.client.addEventListener("response-finished", (e) => finished.push(e.detail));
  f.start("a");
  f.audio("a", 100);
  assert.equal(f.released().length, 0);
  f.done("a");
  assert.equal(f.released().length, 100);
  assert.equal(finished[0].audible, true);
  assert.ok(f.output.render(128).some((v) => v > 0));
});

test("interruption discards pending audio including late audio and completion", () => {
  const f = fixture();
  f.start("a");
  f.audio("a", 100);
  f.interrupt();
  f.audio("a", 2400);
  f.done("a");
  assert.equal(f.released().length, 0);
  assert.ok(f.output.render(128).every((v) => v === 0));
});

test("cancelled and failed responses discard while incomplete responses release partial audio", () => {
  for (const status of ["cancelled", "failed", "incomplete"]) {
    const f = fixture();
    const finished = [];
    f.client.addEventListener("response-finished", (e) => finished.push(e.detail));
    f.start("a");
    f.audio("a", 100);
    f.done("a", status);
    assert.equal(f.released().length, status === "incomplete" ? 100 : 0);
    assert.equal(finished[0].audible, status === "incomplete");
    f.audio("a", 2400);
    assert.equal(f.released().length, status === "incomplete" ? 100 : 0);
  }
});

test("new responses use a fresh gate without clearing released tails", () => {
  const f = fixture();
  f.start("a");
  f.audio("a", 2400);
  f.done("a");
  f.start("b");
  f.audio("b", 2399);
  assert.equal(f.released().length, 2400);
  assert.ok(f.output.render(128).some((v) => v > 0));
  assert.equal(f.messages.some((m) => m.kind === "clear"), false);
  f.audio("b", 1);
  assert.equal(f.released().length, 4800);
});

test("late old completion and audio cannot reset or release a newer response", () => {
  for (const status of ["completed", "cancelled"]) {
    const f = fixture();
    const finished = [];
    f.client.addEventListener("response-finished", (e) => finished.push(e.detail));
    f.start("a");
    f.audio("a", 100);
    f.interrupt();
    f.start("b");
    f.audio("b", 2399, -16384);
    f.audio("a", 2400);
    f.done("a", status);
    assert.equal(f.released().length, 0);
    f.audio("b", 1, -16384);
    assert.deepEqual(f.released(), Array(2400).fill(-0.5));
    f.done("a", status);
    assert.equal(finished[0].responseId, "a", "old transcript still receives completion");
    f.audio("b", 1, -16384);
    assert.equal(f.released().length, 2401);
  }
});

test("interruption during playback clears old audio and next response works", () => {
  const f = fixture();
  f.start("a");
  f.audio("a", 2400);
  f.output.render(128);
  f.interrupt();
  f.start("b");
  f.audio("b", 2400, -16384);
  assert.ok(f.output.render(256).slice(64).every((v) => v === -0.5));
});

test("zero buffer preserves immediate playback", () => {
  const f = fixture(0);
  f.start("a");
  f.audio("a", 100);
  assert.equal(f.released().length, 100);
  assert.ok(f.output.render(128).some((v) => v > 0));
});

test("close discards pending audio and stale events cannot restart playback", async () => {
  const f = fixture();
  f.start("a");
  f.audio("a", 100);
  await f.client.close();
  f.start("b");
  f.audio("b", 2400);
  f.done("a");
  assert.equal(f.released().length, 0);
});

test("WebRTC audio does not enter the worklet buffer", () => {
  const f = fixture(100, "webrtc");
  f.start("a");
  f.audio("a", 2400);
  f.done("a");
  assert.deepEqual(f.messages, []);
});

test("startup reserve absorbs a bounded producer gap that immediate playback cannot", () => {
  for (const threshold of [0, 200]) {
    const f = fixture(threshold);
    f.start("a");
    f.audio("a", 2400); // 100 ms at t=0
    f.output.render(7200); // next chunk at t=150 ms
    f.audio("a", 2400);
    f.output.render(7200); // next chunk at t=300 ms
    f.audio("a", 2400);
    const sustained = f.output.render(4800);
    const underruns = f.output.reports.filter((m) => m.kind === "underrun").length;
    if (threshold === 0) assert.ok(underruns > 0);
    else {
      assert.equal(underruns, 0);
      assert.ok(sustained.every((v) => v > 0));
    }
  }
});

test("fractional milliseconds round up to a whole PCM sample", () => {
  const f = fixture(0.05); // 1.2 samples: require two
  f.start("a");
  f.audio("a", 1);
  assert.equal(f.released().length, 0);
  f.audio("a", 1);
  assert.equal(f.released().length, 2);
});

test("a chunk crossing beyond the threshold is released whole", () => {
  const f = fixture();
  f.start("a");
  f.audio("a", 1000, -16384);
  f.audio("a", 2000, 32767);
  assert.deepEqual(f.released(), [...Array(1000).fill(-0.5), ...Array(2000).fill(1)]);
});

test("disconnect discards pending audio even before the UI closes the client", async () => {
  class Transport {
    listeners = new Map();
    on(name, callback) { this.listeners.set(name, callback); }
  }
  class Session {
    on() {}
    async connect() {}
    close() {}
  }
  globalThis.OpenAIAgentsRealtime = {
    OpenAIRealtimeWebSocket: Transport,
    RealtimeSession: Session,
    RealtimeAgent: class {},
  };
  const f = fixture();
  f.client.options.micStream = { getAudioTracks: () => [{}] };
  f.client._setupAudio = async () => {};
  await f.client.connect();
  f.start("a");
  f.audio("a", 100);
  f.client._transport.listeners.get("connection_change")("disconnected");
  f.start("b");
  f.audio("b", 2400);
  f.done("a");
  assert.equal(f.released().length, 0);
  assert.ok(f.output.render(128).every((v) => v === 0));
  await f.client.close();
  delete globalThis.OpenAIAgentsRealtime;
});

test("SDK interruption signal drops pending audio", async () => {
  const listeners = new Map();
  globalThis.OpenAIAgentsRealtime = {
    OpenAIRealtimeWebSocket: class { on() {} },
    RealtimeSession: class {
      on(name, callback) { listeners.set(name, callback); }
      async connect() {}
      close() {}
    },
    RealtimeAgent: class {},
  };
  const f = fixture();
  f.client.options.micStream = { getAudioTracks: () => [{}] };
  f.client._setupAudio = async () => {};
  await f.client.connect();
  f.start("a");
  listeners.get("audio")({ data: new ArrayBuffer(200), responseId: "a" });
  listeners.get("audio_interrupted")();
  f.done("a");
  assert.equal(f.released().length, 0);
  await f.client.close();
  delete globalThis.OpenAIAgentsRealtime;
});

test("settings persist the reserve, normalize invalid values, and hide it for WebRTC", () => {
  const source = fs.readFileSync(new URL("../main.js", import.meta.url), "utf8");
  // Execute the existing settings functions, without booting camera/account UI.
  const prefix = source.slice(0, source.indexOf("function loadTools()"))
    .replace(/^import .*;$/gm, "");
  const functions = ["readSettingsFromForm", "readGateThreshold", "transportSelectable", "effectiveTransport", "syncTransportUi"]
    .map((name) => source.match(new RegExp(`^function ${name}\\([^]*?^}`, "m"))[0]).join("\n");
  const stored = new Map();
  const field = () => ({ value: "", hidden: false });
  const context = vm.createContext({
    normalizePlaybackBufferMs,
    localStorage: { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value) },
    allowDirect: true, pinnedUrl: "http://server", rtcAvailable: true,
    transportField: field(), inputTransport: field(), transportHint: field(), gateField: field(),
    playbackBufferField: field(), inputPlaybackBuffer: field(), inputLbUrl: field(),
    inputVoice: field(), inputInstructions: field(), inputNoiseGate: { value: "-50" },
    inputAudioInput: field(), inputAudioOutput: field(),
  });
  vm.runInContext(`${prefix}\n${functions}\nglobalThis.api = {loadSettings, saveSettings, readSettingsFromForm, syncTransportUi};`, context);
  const { api } = context;
  assert.equal(api.loadSettings().playbackBufferMs, DEFAULT_PLAYBACK_BUFFER_MS);
  for (const raw of ["", "bad", "NaN", "Infinity", "-1", "   "]) {
    stored.set("s2s.ws.playbackBufferMs", raw);
    assert.equal(api.loadSettings().playbackBufferMs, DEFAULT_PLAYBACK_BUFFER_MS);
  }
  for (const raw of [undefined, null, NaN, Infinity, -1, {}, true]) {
    assert.equal(normalizePlaybackBufferMs(raw), DEFAULT_PLAYBACK_BUFFER_MS);
  }
  context.settings = api.loadSettings();
  context.inputTransport.value = "ws";
  for (const ms of [1200, 0, 0.05]) {
    context.inputPlaybackBuffer.value = String(ms);
    api.saveSettings(api.readSettingsFromForm());
    assert.equal(api.loadSettings().playbackBufferMs, ms);
    assert.equal(stored.get("s2s.ws.playbackBufferMs"), String(ms));
  }
  context.settings = { ...api.loadSettings(), transport: "webrtc", playbackBufferMs: 1200 };
  api.syncTransportUi();
  assert.equal(context.playbackBufferField.hidden, true);
  context.settings.transport = "ws";
  api.syncTransportUi();
  assert.equal(context.playbackBufferField.hidden, false);
  const current = fixture(context.settings.playbackBufferMs);
  context.inputPlaybackBuffer.value = "0";
  api.saveSettings(api.readSettingsFromForm());
  current.start("a");
  current.audio("a", 2400);
  assert.equal(current.released().length, 0, "saved changes must not change an existing conversation");
  const next = fixture(api.loadSettings().playbackBufferMs);
  next.start("b");
  next.audio("b", 1);
  assert.equal(next.released().length, 1, "new conversation uses saved zero");
});

for (const status of ["cancelled", "failed"]) {
  test(`pending ${status} response preserves the completed response tail`, () => {
    const f = fixture();
    f.start("a");
    f.audio("a", 2400);
    f.done("a");
    f.start("b");
    f.audio("b", 100, -16384);
    f.done("b", status);
    assert.equal(f.messages.some((m) => m.kind === "clear"), false);
    assert.ok(f.output.render(4800).slice(32).every((v) => v > 0));
    assert.equal(f.released().length, 2400);
  });
}
