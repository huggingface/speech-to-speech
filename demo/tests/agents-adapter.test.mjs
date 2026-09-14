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

test("a current transcription failure clears its item and resumes listening", () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;

  const client = new S2sRealtimeClient({
    transport: "websocket",
    directUrl: "ws://unused",
  });
  client._status = "processing";
  client._currentUserItemId = "item-current";
  client._userTranscriptByItem.set("item-current", "partial words");
  const statuses = [];
  const errors = [];
  client.addEventListener("status", (event) => statuses.push(event.detail.status));
  client.addEventListener("server-error", (event) => errors.push(event.detail.error.message));

  client._onTransportEvent({
    type: "conversation.item.input_audio_transcription.failed",
    item_id: "item-current",
    content_index: 0,
    error: { message: "transcription request timed out" },
  });

  assert.equal(client.status, "connected");
  assert.equal(client._currentUserItemId, "");
  assert.equal(client._userTranscriptByItem.has("item-current"), false);
  assert.deepEqual(statuses, ["connected"]);
  assert.deepEqual(errors, ["transcription request timed out"]);
});

test("an older transcription failure does not reset the current item", () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;

  const client = new S2sRealtimeClient({
    transport: "websocket",
    directUrl: "ws://unused",
  });
  client._status = "processing";
  client._currentUserItemId = "item-current";
  client._userTranscriptByItem.set("item-old", "old partial");
  client._userTranscriptByItem.set("item-current", "current partial");
  const errors = [];
  client.addEventListener("server-error", (event) => errors.push(event.detail.error.message));

  client._onTransportEvent({
    type: "conversation.item.input_audio_transcription.failed",
    item_id: "item-old",
    content_index: 0,
    error: { message: "old transcription failed" },
  });

  assert.equal(client.status, "processing");
  assert.equal(client._currentUserItemId, "item-current");
  assert.equal(client._userTranscriptByItem.has("item-old"), false);
  assert.equal(client._userTranscriptByItem.get("item-current"), "current partial");
  assert.deepEqual(errors, ["old transcription failed"]);
});
