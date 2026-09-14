import assert from "node:assert/strict";
import test from "node:test";

import { chromium } from "@playwright/test";

import { startTestServer } from "./helpers.mjs";

test("RealtimeSession covers the core GA flow over the stock WebRTC transport", async (t) => {
  const server = await startTestServer();
  t.after(() => server.close());
  const browser = await chromium.launch({
    headless: true,
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
  });
  t.after(() => browser.close());
  const context = await browser.newContext({ permissions: ["microphone"] });
  const page = await context.newPage();
  await page.goto(`${server.http}/test/sdk-page`);

  const result = await page.evaluate(async ({ http }) => {
    const { OpenAIRealtimeWebRTC, RealtimeAgent, RealtimeSession, tool } = window.OpenAIAgentsRealtime;
    const rawEvents = [];
    const errors = [];
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
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioElement = new Audio();
    audioElement.autoplay = true;
    const transport = new OpenAIRealtimeWebRTC({
      mediaStream: mic,
      audioElement,
      useInsecureApiKey: true,
    });
    const session = new RealtimeSession(
      new RealtimeAgent({
        name: "compatibility-test",
        instructions: "Keep replies concise.",
        voice: "coral",
        tools: [lookup],
      }),
      {
        transport,
        model: "s2s-local",
        config: {
          outputModalities: ["audio"],
          audio: {
            input: {
              format: { type: "audio/pcm", rate: 24000 },
              transcription: { model: "whisper-1" },
              turnDetection: { type: "server_vad", interruptResponse: true },
              noiseReduction: null,
            },
            output: { format: { type: "audio/pcm", rate: 24000 }, voice: "coral", speed: 1 },
          },
        },
        tracingDisabled: true,
      },
    );
    transport.on("*", (event) => rawEvents.push(event));
    session.on("error", (event) => errors.push(String(event?.error?.message || event?.error || event)));
    const waitFor = async (predicate, timeout = 15_000) => {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        const value = await predicate();
        if (value) return value;
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      throw new Error(`condition timed out; recent events=${JSON.stringify(rawEvents.slice(-8))}`);
    };
    const post = async (path) => {
      const response = await fetch(`${http}${path}`, { method: "POST" });
      if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
    };
    const state = async () => (await fetch(`${http}/test/state`)).json();
    const receiver = () => transport.connectionState.peerConnection
      ?.getReceivers().find((item) => item.track?.kind === "audio");
    const assistantAudioEnergy = async () => {
      const current = receiver();
      if (!current?.track) return 0;
      const reports = await transport.connectionState.peerConnection.getStats(current.track);
      const inbound = [...reports.values()].find((report) => report.type === "inbound-rtp");
      return inbound?.totalAudioEnergy || 0;
    };
    try {
      await session.connect({ apiKey: "test-key", url: `${http}/v1/realtime/calls` });
      await waitFor(() => rawEvents.some((event) => event.type === "session.updated"));
      const configured = await state();
      await waitFor(async () => (await state()).inputChunks > 0);
      await waitFor(() => receiver()?.track?.readyState === "live" && audioElement.srcObject);
      const energyBeforeVoice = await assistantAudioEnergy();

      await post("/test/voice");
      await waitFor(() => rawEvents.some((event) => event.type === "conversation.item.input_audio_transcription.completed"));
      await waitFor(() => rawEvents.some((event) => event.type === "response.output_audio_transcript.done"));
      await waitFor(() => rawEvents.some((event) => event.type === "response.done" && event.response?.status === "completed"));
      await waitFor(async () => (await assistantAudioEnergy()) > energyBeforeVoice);

      const completedBeforeInterrupt = rawEvents.filter((event) => event.type === "response.done").length;
      const energyBeforeInterrupt = await assistantAudioEnergy();
      await post("/test/start-audio");
      await waitFor(() => rawEvents.filter((event) => event.type === "response.created").length > completedBeforeInterrupt);
      await waitFor(async () => (await assistantAudioEnergy()) > energyBeforeInterrupt);
      session.interrupt();
      await waitFor(() => rawEvents.filter((event) => event.type === "response.done")
        .slice(completedBeforeInterrupt).some(
        (event) => event.type === "response.done" && event.response?.status === "cancelled",
      ));
      await post("/test/settle-audio");
      await new Promise((resolve) => setTimeout(resolve, 100));

      const doneBeforeBargeIn = rawEvents.filter((event) => event.type === "response.done").length;
      const speechStartsBeforeBargeIn = rawEvents.filter(
        (event) => event.type === "input_audio_buffer.speech_started",
      ).length;
      const energyBeforeBargeIn = await assistantAudioEnergy();
      await post("/test/start-audio");
      await waitFor(async () => (await assistantAudioEnergy()) > energyBeforeBargeIn);
      await post("/test/barge-in");
      await waitFor(() => rawEvents.filter(
        (event) => event.type === "input_audio_buffer.speech_started",
      ).length > speechStartsBeforeBargeIn);
      await waitFor(() => rawEvents.filter((event) => event.type === "response.done")
        .slice(doneBeforeBargeIn).some(
          (event) => event.response?.status === "cancelled"
            && event.response?.status_details?.reason === "turn_detected",
        ));
      await post("/test/settle-audio");

      await new Promise((resolve) => setTimeout(resolve, 100));
      await post("/test/tool");
      await waitFor(() => toolExecutions === 1);
      const toolState = await waitFor(async () => {
        const current = await state();
        return current.toolOutputs.includes("result:sdk") && current.responseRequests > 0 ? current : false;
      });
      return { configured, errors, toolOutputs: toolState.toolOutputs };
    } finally {
      session.close();
      mic.getTracks().forEach((track) => track.stop());
    }
  }, { http: server.http });

  assert.equal(result.configured.instructions, "Keep replies concise.");
  assert.equal(result.configured.voice, "coral");
  assert.equal(result.configured.turnDetection, "server_vad");
  assert.equal(result.configured.inputRate, 24_000);
  assert.equal(result.configured.outputRate, 24_000);
  assert.deepEqual(result.configured.tools, ["lookup"]);
  assert.deepEqual(result.toolOutputs, ["result:sdk"]);
  assert.deepEqual(result.errors, []);
});
