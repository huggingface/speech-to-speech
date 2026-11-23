import test from "node:test";
import assert from "node:assert/strict";

import { S2sRealtimeClient } from "../s2s-realtime-client.js";

test("selected language reaches transcription, the agent, and live updates", async () => {
  globalThis.OpenAIAgentsRealtime = {
    RealtimeAgent: class {
      constructor(options) { Object.assign(this, options); }
    },
    tool: (definition) => definition,
  };
  const events = [];
  const client = new S2sRealtimeClient({
    transport: "websocket", voice: "Aiden", instructions: "Be helpful.", language: "es",
  });
  client._autoLanguageSupported = true;
  client._transport = { sendEvent: (event) => events.push(event) };
  client._session = { options: { config: client._sessionConfig() }, updateAgent: async () => {} };

  assert.equal(client._sessionConfig().audio.input.transcription.language, "es");
  assert.match(client._buildAgent().instructions, /Respond in Spanish\./);

  client.updateSession({ language: "de" });
  await Promise.resolve();
  assert.equal(events.at(-1).session.audio.input.transcription.language, "de");
  assert.match(client._buildAgent().instructions, /Respond in German\./);

  client.updateSession({ language: "auto" });
  await Promise.resolve();
  assert.equal(events.at(-1).session.audio.input.transcription.language, "auto");
  assert.equal(client._sessionConfig().audio.input.transcription.language, "auto");
  assert.equal(client._buildAgent().instructions, "Be helpful.");
  delete globalThis.OpenAIAgentsRealtime;
});

test("Auto is sent explicitly at session initialization", () => {
  const client = new S2sRealtimeClient({
    transport: "websocket", voice: "Aiden", instructions: "Be helpful.", language: "auto",
  });
  client._autoLanguageSupported = true;
  assert.equal(client._sessionConfig().audio.input.transcription.language, "auto");
  const events = [];
  client._autoLanguageSupported = true;
  client._transport = { sendEvent: (event) => events.push(event) };
  client._sendLanguageUpdate(client.options.language);
  assert.equal(events[0].session.audio.input.transcription.language, "auto");
});

for (const [initial, selected] of [["auto", "fr"], ["es", "de"], ["fr", "auto"]]) {
  test(`live language ${initial} → ${selected} and reasoning survive agent and tool updates`, async (t) => {
    const realtime = await import("@openai/agents/realtime");
    globalThis.OpenAIAgentsRealtime = realtime;
    t.after(() => { delete globalThis.OpenAIAgentsRealtime; });
    const events = [];
    const transport = new realtime.OpenAIRealtimeWebSocket();
    transport.sendEvent = (event) => events.push(event);
    const client = new S2sRealtimeClient({
      transport: "websocket", voice: "Aiden", instructions: "Be helpful.", language: initial, reasoningEffort: "low",
    });
    client._autoLanguageSupported = true;
    client._transport = transport;
    client._session = new realtime.RealtimeSession(client._buildAgent(), {
      transport, config: client._sessionConfig(), tracingDisabled: true,
    });
    await client._session.getInitialSessionConfig();
    client.setReasoningEffort("high");
    let expectedReasoning = { effort: "high" };
    const flush = () => new Promise((resolve) => setImmediate(resolve));
    const assertLanguage = () => {
      const updates = events.filter((event) => event.session?.audio?.input?.transcription);
      assert.ok(updates.length > 0);
      for (const update of updates) {
        assert.equal(update.session.audio.input.transcription.language, selected);
      }
      const reasoningUpdates = events.filter(event => Object.hasOwn(event.session || {}, "reasoning"));
      assert.ok(reasoningUpdates.length > 0);
      for (const update of reasoningUpdates) {
        assert.deepEqual(update.session.reasoning, expectedReasoning);
      }
      events.length = 0;
    };

    client.updateSession({ language: selected });
    await flush();
    assertLanguage();
    client.setTools([{ type: "function", name: "web_search", description: "Search", parameters: {
      type: "object", properties: {}, required: [], additionalProperties: false,
    } }]);
    await flush();
    assertLanguage();
    client.setTools([]);
    await flush();
    assertLanguage();
    client.setReasoningEffort(null);
    expectedReasoning = null;
    assert.equal(client._session.options.config.audio.input.transcription.language, selected);
    client.updateSession({ voice: "Vivian", instructions: "Be concise." });
    await flush();
    assertLanguage();
  });
}

for (const supported of [true, false, null]) {
  test(`capability discovery gates Auto in initial and subsequent config: ${supported}`, async (t) => {
    const client = new S2sRealtimeClient({ transport: "websocket", language: "auto" });
    t.mock.method(globalThis, "fetch", async (url) => {
      assert.equal(url.pathname, "/v1/realtime/capabilities");
      assert.equal(url.protocol, "https:");
      return { ok: true, json: async () => ({ pipelines: [{ conversation: { session_auto_supported: supported } }] }) };
    });
    await client._loadLanguageCapabilities("wss://backend.test/v1/realtime");
    assert.deepEqual(client._sessionConfig().audio.input.transcription,
      supported === true ? { model: "whisper-1", language: "auto" } : { model: "whisper-1" });
    const events = [];
    client._transport = { sendEvent: event => events.push(event) };
    client._sendLanguageUpdate("auto");
    assert.equal(events.length, supported === true ? 1 : 0);
    assert.deepEqual(client._languageConfig("es"), { language: "es" });
  });
}

test("unavailable discovery leaves backend defaults intact", async (t) => {
  t.mock.method(globalThis, "fetch", async () => { throw new Error("unavailable"); });
  const client = new S2sRealtimeClient({ transport: "websocket", language: "auto" });
  await client._loadLanguageCapabilities("ws://backend.test/v1/realtime");
  assert.equal(Object.hasOwn(client._sessionConfig().audio.input.transcription, "language"), false);
});

test("queued language choices update options before the SDK session exists", async () => {
  const { readFile } = await import("node:fs/promises");
  const vm = await import("node:vm");
  const source = await readFile(new URL("../main.js", import.meta.url), "utf8");
  const handler = source.slice(source.indexOf('mainLanguage.addEventListener("change"'), source.indexOf('settingsForm.addEventListener("submit"'));
  for (const currentState of ["queued", "your-turn", "listening"]) {
    const client = new S2sRealtimeClient({ transport: "websocket", language: "auto" });
    const settings = { language: "auto" };
    let change;
    const mainLanguage = { value: "es", addEventListener: (_, callback) => { change = callback; } };
    vm.runInNewContext(handler, { mainLanguage, inputLanguage: {}, settings, saveSettings() {}, client,
      currentState, LIVE_STATES: new Set(["listening"]) });
    change();
    assert.equal(client.options.language, "es");
    assert.equal(client._sessionConfig().audio.input.transcription.language, "es");
  }
});
