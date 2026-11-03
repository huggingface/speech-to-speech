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
  assert.equal(client._sessionConfig().audio.input.transcription.language, "auto");
  const events = [];
  client._transport = { sendEvent: (event) => events.push(event) };
  client._sendLanguageUpdate(client.options.language);
  assert.equal(events[0].session.audio.input.transcription.language, "auto");
});

for (const [initial, selected] of [["auto", "fr"], ["es", "de"], ["fr", "auto"]]) {
  test(`live language ${initial} → ${selected} survives agent and tool updates`, async (t) => {
    const realtime = await import("@openai/agents/realtime");
    globalThis.OpenAIAgentsRealtime = realtime;
    t.after(() => { delete globalThis.OpenAIAgentsRealtime; });
    const events = [];
    const transport = new realtime.OpenAIRealtimeWebSocket();
    transport.sendEvent = (event) => events.push(event);
    const client = new S2sRealtimeClient({
      transport: "websocket", voice: "Aiden", instructions: "Be helpful.", language: initial,
    });
    client._transport = transport;
    client._session = new realtime.RealtimeSession(client._buildAgent(), {
      transport, config: client._sessionConfig(), tracingDisabled: true,
    });
    await client._session.getInitialSessionConfig();
    const flush = () => new Promise((resolve) => setImmediate(resolve));
    const assertLanguage = () => {
      const updates = events.filter((event) => event.session?.audio?.input?.transcription);
      assert.ok(updates.length > 0);
      for (const update of updates) {
        assert.equal(update.session.audio.input.transcription.language, selected);
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
    client.updateSession({ voice: "Vivian", instructions: "Be concise." });
    await flush();
    assertLanguage();
  });
}
