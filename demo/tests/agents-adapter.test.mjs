import assert from "node:assert/strict";
import test from "node:test";

import * as realtime from "@openai/agents/realtime";

import { S2sRealtimeClient } from "../s2s-realtime-client.js";
import { waitFor } from "./helpers.mjs";

test("the pinned SDK changes the live voice and explicitly clears all tools", async () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;

  const client = new S2sRealtimeClient({
    transport: "websocket",
    directUrl: "ws://unused",
    voice: "Aiden",
    instructions: "Initial instructions.",
    tools: [{
      type: "function",
      name: "lookup",
      description: "Look up a value.",
      parameters: { type: "object", properties: { query: { type: "string" } } },
    }],
    async executeTool() { return { output: "found" }; },
  });
  const sent = [];
  const transport = new realtime.OpenAIRealtimeWebSocket({ useInsecureApiKey: true });
  transport.sendEvent = (event) => sent.push(event);
  client._transport = transport;
  client._agent = client._buildAgent();
  client._session = new realtime.RealtimeSession(client._agent, {
    transport,
    model: "s2s-local",
    config: client._sessionConfig(),
    tracingDisabled: true,
  });
  transport.updateSessionConfig(await client._session.getInitialSessionConfig());

  client.updateSession({ voice: "Coral", instructions: "Updated instructions." });
  await waitFor(() => sent.length >= 2);
  const voiceUpdate = sent.at(-1)?.session;
  assert.equal(voiceUpdate?.audio?.output?.voice, "Coral");
  assert.equal(voiceUpdate?.instructions, "Updated instructions.");

  const beforeClear = sent.length;
  client.setTools([]);
  await waitFor(() => sent.length >= beforeClear + 2);
  assert.deepEqual(sent.at(-1), {
    type: "session.update",
    session: { type: "realtime", tools: [], tool_choice: "none" },
  });
});
