import assert from "node:assert/strict";
import test from "node:test";

import * as realtime from "@openai/agents/realtime";

import { S2sRealtimeClient } from "../s2s-realtime-client.js";
import { waitFor } from "./helpers.mjs";
import { REASONING_LEVELS } from "../ui/reasoning.js";

for (const transport of ["websocket", "webrtc"]) {
  test(`${transport} reports server reasoning acknowledgements without fabricating them from selections`, () => {
    globalThis.localStorage = { getItem() { return null; } };
    const client = new S2sRealtimeClient({ transport });
    const updates = [];
    client.addEventListener("reasoning-updated", (event) => updates.push(event.detail));
    client.setReasoningEffort("high");
    client._onTransportEvent({ type: "session.created", session: { reasoning: { effort: "high" } } });
    assert.deepEqual(updates, []);
    for (const reasoning of [{ effort: "high" }, { effort: "low" }, null, undefined]) {
      client._onTransportEvent({ type: "session.updated", session: { reasoning } });
      assert.deepEqual(updates.at(-1), { reasoning });
    }
  });
}

test("WebSocket preserves live reasoning choices through voice and tool updates", async () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;
  const client = new S2sRealtimeClient({ transport: "websocket", voice: "Aiden", instructions: "Be concise." });
  const sent = [];
  const transport = new realtime.OpenAIRealtimeWebSocket({ useInsecureApiKey: true });
  transport.sendEvent = (event) => sent.push(event);
  Object.defineProperty(transport, "status", { value: "connected" });
  client._transport = transport;
  client._agent = client._buildAgent();
  client._session = new realtime.RealtimeSession(client._agent, {
    transport, model: "s2s-local", config: client._sessionConfig(), tracingDisabled: true,
  });
  transport.updateSessionConfig(await client._session.getInitialSessionConfig());
  assert.equal("reasoning" in sent.at(-1).session, false, "first visit must keep the server default");

  for (const [effort] of REASONING_LEVELS.slice(1).concat([REASONING_LEVELS[0]])) {
    const expected = effort === "default" ? null : { effort };
    client.setReasoningEffort(effort === "default" ? null : effort);
    assert.deepEqual(sent.at(-1), {
      type: "session.update", session: { type: "realtime", reasoning: expected },
    });
    const before = sent.length;
    client.updateSession({ voice: "Coral" });
    await waitFor(() => sent.length > before);
    assert.deepEqual(sent.at(-1).session.reasoning, expected);
    const beforeTools = sent.length;
    client.setTools([]);
    await waitFor(() => sent.length >= beforeTools + 2);
    assert.deepEqual(sent.at(-2).session.reasoning, expected);
  }
});

test("saved reasoning levels are included in the initial SDK session payload", async () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;
  for (const [effort] of REASONING_LEVELS.slice(1)) {
    const client = new S2sRealtimeClient({ transport: "websocket", reasoningEffort: effort });
    const transport = new realtime.OpenAIRealtimeWebSocket({ useInsecureApiKey: true });
    const session = new realtime.RealtimeSession(client._buildAgent(), {
      transport, config: client._sessionConfig(), tracingDisabled: true,
    });
    assert.deepEqual(transport.buildSessionPayload(await session.getInitialSessionConfig()).reasoning, { effort });
  }
});

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
  assert.deepEqual(config.providerData?.extensions, ["speech_to_speech.input_audio_transcription.snapshot", "speech_to_speech.reasoning.delta"]);

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
    "speech_to_speech.reasoning.delta",
  ]);

  client.updateSession({ voice: "Coral" });
  await waitFor(() => sent.length >= 2);
  const updatedSession = sent.at(-1)?.session;
  assert.deepEqual(updatedSession?.extensions, [
    "speech_to_speech.input_audio_transcription.snapshot",
    "speech_to_speech.reasoning.delta",
  ]);
});

test("over WebRTC each barge-in asks for the heard text of the last two replies", () => {
  globalThis.localStorage = { getItem() { return null; } };
  globalThis.OpenAIAgentsRealtime = realtime;
  const client = new S2sRealtimeClient({ transport: "webrtc" });
  const sent = [];
  client._transport = { sendEvent(event) { sent.push(event); } };
  const heard = [];
  client.addEventListener("assistant-heard", (event) => heard.push(event.detail));
  const reply = (responseId, itemId, text) => {
    const ids = { response_id: responseId, item_id: itemId, output_index: 0, content_index: 0 };
    client._onTransportEvent({ type: "response.output_audio_transcript.delta", ...ids, delta: text });
    client._onTransportEvent({ type: "response.output_audio_transcript.done", ...ids, transcript: text });
  };
  reply("r0", "item-0", "Hello.");
  reply("r1", "item-1", "Let me check.");
  reply("r2", "item-2", "It is sunny today.");
  reply("r2", "item-3", "Anything else?");
  const retrieves = () => sent.map((event) => event.item_id);

  client._onTransportEvent({ type: "input_audio_buffer.speech_started", item_id: "user-1" });
  assert.deepEqual(retrieves(), ["item-1", "item-2", "item-3"]);
  assert.equal(sent[0].event_id, "heard_text_item-1");
  // A backchannel that did not interrupt must not stop a later barge-in's check.
  client._onTransportEvent({ type: "input_audio_buffer.speech_started", item_id: "user-2" });
  assert.equal(sent.length, 6);

  const retrieved = (id, status, transcript) => client._onTransportEvent({ type: "conversation.item.retrieved", item: {
    id, type: "message", role: "assistant", status, content: [{ type: "output_audio", transcript }],
  } });
  retrieved("item-1", "completed", "Let me check.");
  // A reply still generating returns its pending text; it was not cut.
  retrieved("item-2", "in_progress", "It is sunny today. And warm.");
  retrieved("item-2", "incomplete", "It is");
  assert.deepEqual(heard, [{ responseId: "r2", text: "It is Anything else?" }]);
});
