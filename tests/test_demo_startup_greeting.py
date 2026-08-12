import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("./demo/ws/s2s-ws-client.js", "S2sWsRealtimeClient"),
        ("./demo/rtc/s2s-rtc-client.js", "S2sRtcRealtimeClient"),
    ],
)
def test_startup_greeting_is_sent_once(module_path, class_name):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
globalThis.localStorage = {{ getItem() {{ return null; }} }};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
  startupGreeting: "  Say hello.  ",
}});
const sent = [];
client._send = (event) => sent.push(event);
client.requestResponse = () => sent.push({{ type: "response.create" }});
client._sendStartupGreeting();
client._sendStartupGreeting();
if (sent.length !== 2) throw new Error(`expected 2 events, got ${{sent.length}}`);
if (sent[0]?.item?.content?.[0]?.text !== "Say hello.") {{
  throw new Error(`unexpected greeting event: ${{JSON.stringify(sent[0])}}`);
}}
if (sent[1]?.type !== "response.create") {{
  throw new Error(`unexpected response event: ${{JSON.stringify(sent[1])}}`);
}}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("./demo/ws/s2s-ws-client.js", "S2sWsRealtimeClient"),
        ("./demo/rtc/s2s-rtc-client.js", "S2sRtcRealtimeClient"),
    ],
)
def test_empty_startup_greeting_is_disabled(module_path, class_name):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
globalThis.localStorage = {{ getItem() {{ return null; }} }};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
  startupGreeting: "   ",
}});
let sends = 0;
client._send = () => sends += 1;
client.requestResponse = () => sends += 1;
client._sendStartupGreeting();
if (sends !== 0) throw new Error(`expected no events, got ${{sends}}`);
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_path,class_name,message_handler",
    [
        ("./demo/ws/s2s-ws-client.js", "S2sWsRealtimeClient", "_onWsMessage"),
        ("./demo/rtc/s2s-rtc-client.js", "S2sRtcRealtimeClient", "_onDcMessage"),
    ],
)
def test_demo_clients_finalize_one_transcript_after_streamed_deltas(module_path, class_name, message_handler):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
globalThis.localStorage = {{ getItem() {{ return null; }} }};
globalThis.CustomEvent = class CustomEvent extends Event {{
  constructor(type, init = {{}}) {{
    super(type);
    this.detail = init.detail;
  }}
}};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const transcripts = [];
let finishedTranscript = null;
client.addEventListener("transcript", (event) => transcripts.push(event.detail));
client.addEventListener("response-finished", (event) => {{
  finishedTranscript = event.detail.transcript;
}});
const deliver = async (event) => {{
  await client[{json.dumps(message_handler)}](JSON.stringify(event));
}};

await deliver({{
  type: "response.output_audio_transcript.delta",
  response_id: "response_1",
  delta: "Hello",
}});
await deliver({{
  type: "response.output_audio_transcript.delta",
  response_id: "response_1",
  delta: " there.",
}});
await deliver({{
  type: "response.output_audio_transcript.done",
  response_id: "response_1",
  transcript: "Hello there.",
}});
await deliver({{
  type: "response.done",
  response: {{ id: "response_1", status: "completed", output: [] }},
}});

const expected = [
  {{ role: "assistant", text: "Hello", partial: true, responseId: "response_1" }},
  {{ role: "assistant", text: "Hello there.", partial: true, responseId: "response_1" }},
  {{ role: "assistant", text: "Hello there.", partial: false, responseId: "response_1" }},
];
if (JSON.stringify(transcripts) !== JSON.stringify(expected)) {{
  throw new Error(`unexpected transcript events: ${{JSON.stringify(transcripts)}}`);
}}
if (finishedTranscript !== "Hello there.") {{
  throw new Error(`unexpected finished transcript: ${{finishedTranscript}}`);
}}
if (client._asstTranscriptByResp.size !== 0 || client._asstFullByResp.size !== 0) {{
  throw new Error("response.done did not clear transcript state");
}}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_path,class_name,message_handler",
    [
        ("./demo/ws/s2s-ws-client.js", "S2sWsRealtimeClient", "_onWsMessage"),
        ("./demo/rtc/s2s-rtc-client.js", "S2sRtcRealtimeClient", "_onDcMessage"),
    ],
)
def test_demo_clients_accumulate_user_transcription_deltas_across_reopen(module_path, class_name, message_handler):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
globalThis.localStorage = {{ getItem() {{ return null; }} }};
globalThis.CustomEvent = class CustomEvent extends Event {{
  constructor(type, init = {{}}) {{
    super(type);
    this.detail = init.detail;
  }}
}};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const transcripts = [];
client.addEventListener("transcript", (event) => transcripts.push(event.detail));
const deliver = async (event) => {{
  await client[{json.dumps(message_handler)}](JSON.stringify(event));
}};

await deliver({{ type: "input_audio_buffer.speech_started", item_id: "item_1" }});
await deliver({{
  type: "conversation.item.input_audio_transcription.delta",
  item_id: "item_1",
  content_index: 0,
  delta: "hel",
}});
await deliver({{
  type: "conversation.item.input_audio_transcription.delta",
  item_id: "item_1",
  content_index: 0,
  delta: "lo",
}});
await deliver({{
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_1",
  content_index: 0,
  transcript: "hello",
}});
await deliver({{ type: "input_audio_buffer.speech_started", item_id: "item_1" }});
await deliver({{
  type: "conversation.item.input_audio_transcription.delta",
  item_id: "item_1",
  content_index: 0,
  delta: " again",
}});

const expected = [
  {{ role: "user", text: "hel", partial: true, itemId: "item_1" }},
  {{ role: "user", text: "hello", partial: true, itemId: "item_1" }},
  {{ role: "user", text: "hello", partial: false, itemId: "item_1" }},
  {{ role: "user", text: "hello again", partial: true, itemId: "item_1" }},
];
if (JSON.stringify(transcripts) !== JSON.stringify(expected)) {{
  throw new Error(`unexpected transcript events: ${{JSON.stringify(transcripts)}}`);
}}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_path,class_name,message_handler",
    [
        ("./demo/ws/s2s-ws-client.js", "S2sWsRealtimeClient", "_onWsMessage"),
        ("./demo/rtc/s2s-rtc-client.js", "S2sRtcRealtimeClient", "_onDcMessage"),
    ],
)
def test_demo_clients_leave_processing_after_empty_transcript(module_path, class_name, message_handler):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
globalThis.localStorage = {{ getItem() {{ return null; }} }};
globalThis.CustomEvent = class CustomEvent extends Event {{
  constructor(type, init = {{}}) {{
    super(type);
    this.detail = init.detail;
  }}
}};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
client._status = "connected";
const statuses = [];
const transcripts = [];
client.addEventListener("status", (event) => statuses.push(event.detail.status));
client.addEventListener("transcript", (event) => transcripts.push(event.detail));
const deliver = async (event) => {{
  await client[{json.dumps(message_handler)}](JSON.stringify(event));
}};

await deliver({{ type: "input_audio_buffer.speech_started", item_id: "item_empty" }});
await deliver({{ type: "input_audio_buffer.speech_stopped", item_id: "item_empty" }});
if (client.status !== "processing") throw new Error(`expected processing, got ${{client.status}}`);

await deliver({{
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_empty",
  transcript: "",
}});

const expectedStatuses = ["user-speaking", "processing", "connected"];
if (JSON.stringify(statuses) !== JSON.stringify(expectedStatuses)) {{
  throw new Error(`unexpected statuses: ${{JSON.stringify(statuses)}}`);
}}
if (transcripts.length !== 0) {{
  throw new Error(`empty final transcript was emitted: ${{JSON.stringify(transcripts)}}`);
}}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module_path,class_name,message_handler",
    [
        ("./demo/ws/s2s-ws-client.js", "S2sWsRealtimeClient", "_onWsMessage"),
        ("./demo/rtc/s2s-rtc-client.js", "S2sRtcRealtimeClient", "_onDcMessage"),
    ],
)
def test_empty_transcript_preserves_processing_for_pending_response(module_path, class_name, message_handler):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
globalThis.localStorage = {{ getItem() {{ return null; }} }};
globalThis.CustomEvent = class CustomEvent extends Event {{
  constructor(type, init = {{}}) {{
    super(type);
    this.detail = init.detail;
  }}
}};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const deliverEmptyTranscript = async () => {{
  client._status = "processing";
  await client[{json.dumps(message_handler)}](JSON.stringify({{
    type: "conversation.item.input_audio_transcription.completed",
    transcript: "",
  }}));
  if (client.status !== "processing") {{
    throw new Error(`pending response status was overwritten: ${{client.status}}`);
  }}
}};

client._activeResponseId = "response_1";
await deliverEmptyTranscript();
client._activeResponseId = "";
client._pendingCreateId = "create_1";
await deliverEmptyTranscript();
client._pendingCreateId = "";
client._createQueue.push({{}});
await deliverEmptyTranscript();
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_rtc_microphone_opens_after_startup_events_are_queued():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = """
globalThis.localStorage = { getItem() { return null; } };
const { S2sRtcRealtimeClient } = await import("./demo/rtc/s2s-rtc-client.js");
const timeline = [];
let enabled = true;
const track = {};
Object.defineProperty(track, "enabled", {
  get() { return enabled; },
  set(value) {
    enabled = value;
    timeline.push(`mic:${value}`);
  },
});
const client = new S2sRtcRealtimeClient({
  voice: "Aiden",
  instructions: "Be helpful.",
  callsUrl: "api/calls",
  startupGreeting: "Say hello.",
  micStream: { getAudioTracks() { return [track]; } },
});
client._sendSessionUpdate = () => timeline.push("session.update");
client._send = (event) => timeline.push(event.type);
client.requestResponse = () => timeline.push("response.create");

// This is the gate applied before addTrack() during connect().
client._syncMicTransmission();
// An unmute while connecting must not bypass the gate.
client.setMuted(false);
client._onDcMessage(JSON.stringify({ type: "session.created" }));

const expected = [
  "mic:false",
  "mic:false",
  "session.update",
  "conversation.item.create",
  "response.create",
  "mic:true",
];
if (JSON.stringify(timeline) !== JSON.stringify(expected)) {
  throw new Error(`unexpected ordering: ${JSON.stringify(timeline)}`);
}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_rtc_client_emits_user_turn_lifecycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = """
globalThis.localStorage = { getItem() { return null; } };
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) {
    super(type);
    this.detail = init.detail;
  }
};
const { S2sRtcRealtimeClient } = await import("./demo/rtc/s2s-rtc-client.js");
const client = new S2sRtcRealtimeClient({
  voice: "Aiden",
  instructions: "Be helpful.",
  callsUrl: "api/calls",
});
const events = [];
client.addEventListener("user-turn-started", (event) => {
  events.push(["started", event.detail.itemId]);
});
client.addEventListener("user-turn-stopped", (event) => {
  events.push(["stopped", event.detail.itemId]);
});
client._onDcMessage(JSON.stringify({
  type: "input_audio_buffer.speech_started",
  item_id: "item_voice",
}));
client._onDcMessage(JSON.stringify({
  type: "input_audio_buffer.speech_stopped",
  item_id: "item_voice",
}));
const expected = [
  ["started", "item_voice"],
  ["stopped", "item_voice"],
];
if (JSON.stringify(events) !== JSON.stringify(expected)) {
  throw new Error(`unexpected lifecycle: ${JSON.stringify(events)}`);
}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
