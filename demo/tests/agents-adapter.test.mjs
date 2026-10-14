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

test("speculative transcript snapshots update display and prune on completion", () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;

  const client = new S2sRealtimeClient({
    transport: "websocket",
    directUrl: "ws://unused",
  });

  const config = client._sessionConfig();
  assert.deepEqual(config.providerData?.extensions, ["speech_to_speech.input_audio_transcription.snapshot"]);

  const transcripts = [];
  client.addEventListener("transcript", (event) => transcripts.push(event.detail));

  client._onTransportEvent({
    type: "speech_to_speech.input_audio_transcription.snapshot",
    item_id: "item-1",
    content_index: 0,
    transcript: "hello brave",
  });

  assert.equal(client._userSnapshotByItem.get("item-1"), "hello brave");
  assert.deepEqual(transcripts, [
    { role: "user", text: "hello brave", partial: true, itemId: "item-1" },
  ]);

  client._onTransportEvent({
    type: "conversation.item.input_audio_transcription.delta",
    item_id: "item-1",
    content_index: 0,
    delta: "hello",
  });

  assert.equal(client._userTranscriptByItem.get("item-1"), "hello");
  assert.equal(transcripts.at(-1)?.text, "hello brave");

  client._onTransportEvent({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "item-1",
    content_index: 0,
    transcript: "hello brave new world",
  });

  assert.equal(client._userSnapshotByItem.has("item-1"), false);
  assert.equal(client._userTranscriptByItem.has("item-1"), false);
  assert.deepEqual(transcripts.at(-1), {
    role: "user",
    text: "hello brave new world",
    partial: false,
    itemId: "item-1",
  });
});

test("the client negotiates speculative snapshot extensions on the wire", async () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;

  const client = new S2sRealtimeClient({
    transport: "websocket",
    directUrl: "ws://unused",
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

  assert.ok(sent.length >= 1);
  const initialSession = sent[0]?.session;
  assert.deepEqual(initialSession?.extensions, [
    "speech_to_speech.input_audio_transcription.snapshot",
  ]);

  client.updateSession({ voice: "Coral" });
  await waitFor(() => sent.length >= 2);
  const updatedSession = sent.at(-1)?.session;
  assert.deepEqual(updatedSession?.extensions, [
    "speech_to_speech.input_audio_transcription.snapshot",
  ]);
});
