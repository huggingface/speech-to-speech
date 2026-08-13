import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_node(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_adapter_uses_stock_sdk_transport_for_each_mode():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
globalThis.location = { href: "https://demo.test/" };
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) { super(type); this.detail = init.detail; }
};
globalThis.Audio = class Audio { constructor() { this.autoplay = false; this.srcObject = null; } };

const sessions = [];
class Transport {
  constructor(options) { this.options = options; this.listeners = new Map(); }
  on(name, callback) { this.listeners.set(name, callback); }
  requestResponse() {}
}
class WebSocketTransport extends Transport {}
class WebRtcTransport extends Transport {}
class RealtimeAgent { constructor(options) { Object.assign(this, options); } }
class RealtimeSession {
  constructor(agent, options) {
    this.agent = agent;
    this.options = options;
    this.listeners = new Map();
    this.messages = [];
    sessions.push(this);
  }
  on(name, callback) { this.listeners.set(name, callback); }
  async connect(options) { this.connectOptions = options; }
  sendMessage(message) { this.messages.push(message); }
  mute() {}
  close() {}
}
globalThis.OpenAIAgentsRealtime = {
  OpenAIRealtimeWebSocket: WebSocketTransport,
  OpenAIRealtimeWebRTC: WebRtcTransport,
  RealtimeAgent,
  RealtimeSession,
  tool: (options) => options,
};

const { S2sRealtimeClient } = await import("./demo/s2s-realtime-client.js");
const micStream = { getAudioTracks() { return [{}]; } };
for (const [transport, url] of [
  ["websocket", "ws://example.test/v1/realtime"],
  ["webrtc", "api/calls"],
]) {
  const client = new S2sRealtimeClient({
    transport,
    directUrl: transport === "websocket" ? url : "",
    callsUrl: transport === "webrtc" ? url : "",
    voice: "coral",
    instructions: "Be concise.",
    startupGreeting: "Say hello.",
    micStream,
  });
  client._setupAudio = async () => {};
  client._attachRtcOutput = () => {};
  await client.connect();
  const session = sessions.at(-1);
  const expectedUrl = transport === "webrtc" ? "https://demo.test/api/calls" : url;
  if (session.connectOptions.url !== expectedUrl) throw new Error(`wrong ${transport} URL`);
  if (session.messages.join("|") !== "Say hello.") throw new Error("startup greeting missing");
  if (session.options.config.audio.input.format.rate !== 24000) throw new Error("input is not GA PCM 24 kHz");
  if (session.options.config.audio.output.format.rate !== 24000) throw new Error("output is not GA PCM 24 kHz");
  if (Object.hasOwn(session.options.config.audio.output, "voice")) {
    throw new Error("nested voice would pin live updates to the initial value");
  }
  if (session.agent.voice !== "coral") throw new Error("initial agent voice missing");
  if (transport === "websocket" && !(session.options.transport instanceof WebSocketTransport)) {
    throw new Error("WebSocket did not use the stock SDK transport");
  }
  if (transport === "webrtc" && !(session.options.transport instanceof WebRtcTransport)) {
    throw new Error("WebRTC did not use the stock SDK transport");
  }
}
"""
    )


def test_adapter_delegates_tools_to_realtime_session():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
class RealtimeAgent { constructor(options) { Object.assign(this, options); } }
globalThis.OpenAIAgentsRealtime = {
  RealtimeAgent,
  tool: (options) => options,
};
const { S2sRealtimeClient } = await import("./demo/s2s-realtime-client.js");
const calls = [];
const images = [];
const client = new S2sRealtimeClient({
  transport: "websocket",
  directUrl: "ws://unused",
  voice: "coral",
  instructions: "Be concise.",
  tools: [{
    type: "function",
    name: "lookup",
    description: "Look up a value.",
    parameters: { type: "object", properties: { query: { type: "string" } } },
  }],
  async executeTool(call) {
    calls.push(call);
    return { output: "found", image: "data:image/jpeg;base64,AA==" };
  },
});
client._session = { addImage(image, options) { images.push([image, options]); } };
const agent = client._buildAgent();
const output = await agent.tools[0].execute(
  { query: "sdk" },
  undefined,
  { toolCall: { callId: "call_1" } },
);
if (output !== "found") throw new Error("tool output was not returned to RealtimeSession");
if (JSON.stringify(calls) !== JSON.stringify([{
  name: "lookup", arguments: '{"query":"sdk"}', callId: "call_1",
}])) throw new Error(`unexpected tool call: ${JSON.stringify(calls)}`);
if (images.length !== 1 || images[0][1].triggerResponse !== false) {
  throw new Error("tool image was not inserted before the SDK function output");
}
"""
    )


def test_webrtc_mic_opens_only_after_startup_greeting():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) { super(type); this.detail = init.detail; }
};
globalThis.Audio = class Audio { constructor() { this.autoplay = false; this.srcObject = null; } };

const timeline = [];
let enabled = true;
const track = {};
Object.defineProperty(track, "enabled", {
  get() { return enabled; },
  set(value) { enabled = value; timeline.push(`mic:${value}`); },
});
class RealtimeSession {
  on() {}
  async connect() { timeline.push("connect"); }
  sendMessage(message) { timeline.push(`greeting:${message}`); }
  mute(muted) { track.enabled = !muted; }
}
class Transport { on() {} }
globalThis.OpenAIAgentsRealtime = {
  OpenAIRealtimeWebRTC: Transport,
  RealtimeAgent: class RealtimeAgent {},
  RealtimeSession,
  tool: (options) => options,
};

const { S2sRealtimeClient } = await import("./demo/s2s-realtime-client.js");
const client = new S2sRealtimeClient({
  transport: "webrtc",
  callsUrl: "https://example.test/v1/realtime/calls",
  voice: "coral",
  instructions: "Be concise.",
  startupGreeting: "Say hello.",
  micStream: { getAudioTracks() { return [track]; } },
});
client._setupAudio = async () => {};
client._attachRtcOutput = () => {};
await client.connect();
const expected = ["mic:false", "connect", "greeting:Say hello.", "mic:true"];
if (JSON.stringify(timeline) !== JSON.stringify(expected)) {
  throw new Error(`unexpected startup ordering: ${JSON.stringify(timeline)}`);
}
"""
    )


def test_empty_transcript_does_not_clear_active_or_pending_response():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) { super(type); this.detail = init.detail; }
};
const { S2sRealtimeClient } = await import("./demo/s2s-realtime-client.js");
const client = new S2sRealtimeClient({
  transport: "websocket",
  directUrl: "ws://unused",
  voice: "coral",
  instructions: "Be concise.",
});
const emptyTranscript = {
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_empty",
  transcript: "",
};

client._status = "processing";
client._activeResponseId = "response_1";
client._onTransportEvent(emptyTranscript);
if (client.status !== "processing") throw new Error("active response was cleared");

client._activeResponseId = "";
client._responseRequested = true;
client._onTransportEvent(emptyTranscript);
if (client.status !== "processing") throw new Error("pending response was cleared");

client._responseRequested = false;
client._onTransportEvent(emptyTranscript);
if (client.status !== "connected") throw new Error("idle empty transcript did not settle");
"""
    )


def test_queue_errors_preserve_auth_and_limit_details():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
const { S2sRealtimeClient } = await import("./demo/s2s-realtime-client.js");
const client = new S2sRealtimeClient({
  transport: "websocket",
  sessionUrl: "api/session",
  voice: "coral",
  instructions: "Be concise.",
});
client._queueSleep = async () => {};

globalThis.fetch = async () => ({
  status: 402,
  async json() { return { tier: "free" }; },
});
let error;
try {
  await client._pollQueue({ queueId: "q1", position: 1, pollIntervalS: 1 });
} catch (caught) { error = caught; }
if (error?.code !== "limit" || error?.tier !== "free") {
  throw new Error(`lost limit detail: ${JSON.stringify(error)}`);
}

globalThis.fetch = async () => ({
  status: 401,
  async json() { return { loginUrl: "/login" }; },
});
error = undefined;
try {
  await client._pollQueue({ queueId: "q2", position: 1, pollIntervalS: 1 });
} catch (caught) { error = caught; }
if (error?.code !== "login-required" || error?.loginUrl !== "/login") {
  throw new Error(`lost login detail: ${JSON.stringify(error)}`);
}
"""
    )
