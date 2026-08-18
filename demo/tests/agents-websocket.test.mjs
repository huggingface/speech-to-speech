import assert from "node:assert/strict";
import test from "node:test";

import { chromium } from "@playwright/test";
import {
  OpenAIRealtimeWebSocket,
  RealtimeAgent,
  RealtimeSession,
  tool,
} from "@openai/agents/realtime";

import { post, SDK_CONFIG, startTestServer, state, waitFor } from "./helpers.mjs";

test("browser RealtimeSession negotiates the stock WebSocket subprotocol", async (t) => {
  const server = await startTestServer();
  t.after(() => server.close());
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.goto(`${server.http}/test/sdk-page`);

  const result = await page.evaluate(async ({ config, ws }) => {
    const { OpenAIRealtimeWebSocket, RealtimeAgent, RealtimeSession } = window.OpenAIAgentsRealtime;
    const rawEvents = [];
    const errors = [];
    const transport = new OpenAIRealtimeWebSocket({ useInsecureApiKey: true });
    const session = new RealtimeSession(
      new RealtimeAgent({ name: "browser-websocket-test", instructions: "Be concise." }),
      { transport, model: "s2s-local", config, tracingDisabled: true },
    );
    transport.on("*", (event) => rawEvents.push(event));
    session.on("error", (event) => errors.push(String(event?.error?.message || event?.error || event)));
    const waitForEvent = async (eventType) => {
      const deadline = Date.now() + 10_000;
      while (Date.now() < deadline) {
        if (rawEvents.some((event) => event.type === eventType)) return;
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      throw new Error(`Timed out waiting for ${eventType}; events=${JSON.stringify(rawEvents)}`);
    };
    try {
      await session.connect({ apiKey: "test-key", url: ws });
      await waitForEvent("session.updated");
      return {
        errors,
        protocol: transport.connectionState.websocket?.protocol,
      };
    } finally {
      session.close();
    }
  }, { config: SDK_CONFIG, ws: server.ws });

  assert.equal(result.protocol, "realtime");
  assert.deepEqual(result.errors, []);
});

test("RealtimeSession covers the core GA flow over the stock WebSocket transport", async (t) => {
  const server = await startTestServer();
  t.after(() => server.close());

  let toolExecutions = 0;
  const lookup = tool({
    name: "lookup",
    description: "Look up a value for the compatibility test.",
    parameters: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
      additionalProperties: false,
    },
    strict: false,
    execute: async ({ query }) => {
      toolExecutions += 1;
      return `result:${query}`;
    },
  });
  const agent = new RealtimeAgent({
    name: "compatibility-test",
    instructions: "Keep replies concise.",
    voice: "coral",
    tools: [lookup],
  });
  const transport = new OpenAIRealtimeWebSocket({ useInsecureApiKey: true });
  const session = new RealtimeSession(agent, {
    transport,
    model: "s2s-local",
    config: SDK_CONFIG,
    tracingDisabled: true,
  });
  const rawEvents = [];
  const errors = [];
  const audio = [];
  transport.on("*", (event) => rawEvents.push(event));
  session.on("error", (event) => errors.push(event));
  session.on("audio", (event) => audio.push(event));

  await session.connect({ apiKey: "test-key", url: server.ws });
  t.after(() => session.close());
  await waitFor(() => rawEvents.some((event) => event.type === "session.updated"));

  const configured = await state(server.http);
  assert.equal(configured.instructions, "Keep replies concise.");
  assert.equal(configured.voice, "coral");
  assert.equal(configured.turnDetection, "server_vad");
  assert.equal(configured.inputRate, 24_000);
  assert.equal(configured.outputRate, 24_000);
  assert.deepEqual(configured.tools, ["lookup"]);

  session.sendAudio(new ArrayBuffer(4_096));
  await waitFor(async () => (await state(server.http)).inputChunks > 0);

  await post(server.http, "/test/voice");
  await waitFor(() => rawEvents.some((event) => event.type === "conversation.item.input_audio_transcription.completed"));
  await waitFor(() => rawEvents.some((event) => event.type === "response.output_audio_transcript.done"));
  await waitFor(() => rawEvents.some((event) => event.type === "response.output_audio.delta"));
  await waitFor(() => audio.length > 0);
  await waitFor(() => rawEvents.some((event) => event.type === "response.done" && event.response?.status === "completed"));

  const completedBeforeInterrupt = rawEvents.filter((event) => event.type === "response.done").length;
  await post(server.http, "/test/start-audio");
  await waitFor(() => audio.length > 1);
  session.interrupt();
  await waitFor(() => rawEvents.filter((event) => event.type === "response.done")
    .slice(completedBeforeInterrupt).some(
    (event) => event.type === "response.done" && event.response?.status === "cancelled",
  ));
  assert.equal(errors.length, 0, JSON.stringify(errors));
  await post(server.http, "/test/settle-audio");
  await new Promise((resolve) => setTimeout(resolve, 100));

  const doneBeforeBargeIn = rawEvents.filter((event) => event.type === "response.done").length;
  const speechStartsBeforeBargeIn = rawEvents.filter(
    (event) => event.type === "input_audio_buffer.speech_started",
  ).length;
  const audioBeforeBargeIn = audio.length;
  await post(server.http, "/test/start-audio");
  await waitFor(() => audio.length > audioBeforeBargeIn);
  await post(server.http, "/test/barge-in");
  await waitFor(() => rawEvents.filter(
    (event) => event.type === "input_audio_buffer.speech_started",
  ).length > speechStartsBeforeBargeIn);
  await waitFor(() => rawEvents.filter((event) => event.type === "response.done")
    .slice(doneBeforeBargeIn).some(
      (event) => event.response?.status === "cancelled"
        && event.response?.status_details?.reason === "turn_detected",
  ));
  assert.equal(errors.length, 0, JSON.stringify(errors));
  await post(server.http, "/test/settle-audio");

  await new Promise((resolve) => setTimeout(resolve, 100));
  await post(server.http, "/test/tool");
  await waitFor(
    () => toolExecutions === 1,
    { timeout: 10_000, interval: 25 },
  ).catch(async (error) => {
    const current = await state(server.http);
    const eventTypes = rawEvents.slice(-20).map((event) => event.type);
    throw new Error(`${error.message}; events=${JSON.stringify(eventTypes)}; state=${JSON.stringify(current)}; errors=${JSON.stringify(errors)}`);
  });
  await waitFor(async () => {
    const current = await state(server.http);
    return current.responseRequests > 0 && current.inResponse;
  });
  await post(server.http, "/test/finish-response");
  const toolState = await waitFor(async () => {
    const current = await state(server.http);
    return current.toolOutputs.length === 1 ? current : false;
  }).catch(async (error) => {
    const current = await state(server.http);
    const serverErrors = rawEvents.filter((event) => event.type === "error");
    throw new Error(`${error.message}; state=${JSON.stringify(current)}; errors=${JSON.stringify(serverErrors)}`);
  });
  assert.deepEqual(toolState.toolOutputs, ["result:sdk"]);
});
