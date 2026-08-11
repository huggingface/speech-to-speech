import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("module_path", "class_name", "message_handler"),
    [
        ("./demo/ws/s2s-ws-client.js", "S2sWsRealtimeClient", "_onWsMessage"),
        ("./demo/rtc/s2s-rtc-client.js", "S2sRtcRealtimeClient", "_onDcMessage"),
    ],
)
def test_demo_clients_preserve_tool_call_response_id(module_path, class_name, message_handler):
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
let toolCall = null;
client.addEventListener("toolcall", (event) => {{ toolCall = event.detail; }});
await client[{json.dumps(message_handler)}](JSON.stringify({{
  type: "response.function_call_arguments.done",
  response_id: "response_1",
  call_id: "call_1",
  name: "web_search",
  arguments: "{{}}",
}}));
if (toolCall?.responseId !== "response_1") {{
  throw new Error(`response_id was not preserved: ${{JSON.stringify(toolCall)}}`);
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
    ("completion_order", "response_finishes_first"),
    [([0, 1], True), ([1, 0], False)],
)
def test_tool_calls_from_one_response_request_one_follow_up(completion_order, response_finishes_first):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
const {{ ToolCallBatcher }} = await import("./demo/tool-call-batcher.js");
const deferred = () => {{
  let resolve;
  const promise = new Promise((done) => {{ resolve = done; }});
  return {{ promise, resolve }};
}};
const pending = [deferred(), deferred()];
const executions = [
  pending[0].promise.then(() => ({{ callId: "call_1", output: "first" }})),
  pending[1].promise.then(() => ({{ callId: "call_2", output: "second" }})),
];
const timeline = [];
const batches = new ToolCallBatcher(
  (_responseId, result) => timeline.push(`output:${{result.callId}}:${{result.output}}`),
  () => timeline.push("response.create"),
);
batches.add("response_1", executions[0]);
batches.add("response_1", executions[1]);
if (JSON.stringify(timeline) !== JSON.stringify(["response.create"])) {{
  throw new Error(`follow-up was not queued immediately: ${{JSON.stringify(timeline)}}`);
}}

let flush = null;
if ({json.dumps(response_finishes_first)}) {{
  flush = batches.finish("response_1", "completed");
}}
for (const index of {json.dumps(completion_order)}) {{
  pending[index].resolve();
  await Promise.resolve();
}}
await Promise.all(executions);
if (!flush) flush = batches.finish("response_1", "completed");
if (!flush) throw new Error("response batch was not registered");
await flush;

const expected = ["response.create", "output:call_1:first", "output:call_2:second"];
if (JSON.stringify(timeline) !== JSON.stringify(expected)) {{
  throw new Error(`unexpected tool batch: ${{JSON.stringify(timeline)}}`);
}}
if (timeline.filter((item) => item === "response.create").length !== 1) {{
  throw new Error(`expected exactly one follow-up: ${{JSON.stringify(timeline)}}`);
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
    ("module_path", "class_name", "message_handler", "attach_transport"),
    [
        (
            "./demo/ws/s2s-ws-client.js",
            "S2sWsRealtimeClient",
            "_onWsMessage",
            "client._ws = { readyState: WebSocket.OPEN, send: record };",
        ),
        (
            "./demo/rtc/s2s-rtc-client.js",
            "S2sRtcRealtimeClient",
            "_onDcMessage",
            'client._dc = { readyState: "open", send: record };',
        ),
    ],
)
@pytest.mark.parametrize("completion_order", [[0, 1], [1, 0]])
def test_demo_clients_send_tool_follow_up_after_stale_response_created(
    module_path,
    class_name,
    message_handler,
    attach_transport,
    completion_order,
):
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
globalThis.WebSocket = {{ OPEN: 1 }};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const {{ ToolCallBatcher }} = await import("./demo/tool-call-batcher.js");
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const sent = [];
const record = (raw) => sent.push(JSON.parse(raw));
{attach_transport}

const deferred = () => {{
  let resolve;
  const promise = new Promise((done) => {{ resolve = done; }});
  return {{ promise, resolve }};
}};
const pending = [deferred(), deferred()];
let executionIndex = 0;
let flush = null;
const batches = new ToolCallBatcher(
  (_responseId, result) => client.sendToolOutput(result.callId, result.output),
  (responseId) => client.requestToolFollowUp(responseId),
);
client.addEventListener("toolcall", (event) => {{
  const index = executionIndex++;
  const detail = event.detail;
  batches.add(
    detail.responseId,
    pending[index].promise.then(() => ({{ callId: detail.callId, output: `output_${{index + 1}}` }})),
  );
}});
client.addEventListener("response-finished", (event) => {{
  flush = batches.finish(event.detail.responseId, event.detail.status);
}});
const deliver = (event) => client[{json.dumps(message_handler)}](JSON.stringify(event));

// A stale or duplicated response.created used to poison the counter and leave
// the completed tool response's follow-up queued forever.
await deliver({{ type: "response.created", response: {{ id: "response_stale" }} }});
await deliver({{ type: "response.created", response: {{ id: "response_1" }} }});
for (const [index, callId] of ["call_1", "call_2"].entries()) {{
  await deliver({{
    type: "response.function_call_arguments.done",
    response_id: "response_1",
    call_id: callId,
    name: "web_search",
    arguments: JSON.stringify({{ query: `query_${{index + 1}}` }}),
  }});
}}
await deliver({{ type: "response.done", response: {{ id: "response_1", status: "completed" }} }});
if (!flush) throw new Error("completed response did not register a tool batch flush");
for (const index of {json.dumps(completion_order)}) pending[index].resolve();
await flush;

const eventTypes = sent.map((event) => event.type);
const expected = ["response.create", "conversation.item.create", "conversation.item.create"];
if (JSON.stringify(eventTypes) !== JSON.stringify(expected)) {{
  throw new Error(`tool follow-up was not transmitted: ${{JSON.stringify(sent)}}`);
}}
const outputs = sent.slice(1).map((event) => event.item.call_id);
if (JSON.stringify(outputs) !== JSON.stringify(["call_1", "call_2"])) {{
  throw new Error(`tool outputs lost call order: ${{JSON.stringify(outputs)}}`);
}}
if (client._createQueue.length !== 0) {{
  throw new Error(`follow-up remained queued: ${{client._createQueue.length}}`);
}}
const followUpMetadata = sent[0].response.metadata;
await deliver({{
  type: "response.created",
  response: {{ id: "response_2", metadata: followUpMetadata }},
}});
if (client._queuedToolFollowUp !== null) {{
  throw new Error("follow-up response.created did not clear the queued marker");
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
    ("module_path", "class_name", "message_handler", "attach_transport"),
    [
        (
            "./demo/ws/s2s-ws-client.js",
            "S2sWsRealtimeClient",
            "_onWsMessage",
            "client._ws = { readyState: WebSocket.OPEN, send: record };",
        ),
        (
            "./demo/rtc/s2s-rtc-client.js",
            "S2sRtcRealtimeClient",
            "_onDcMessage",
            'client._dc = { readyState: "open", send: record };',
        ),
    ],
)
def test_demo_clients_unlock_after_invalid_tool_follow_up(
    module_path,
    class_name,
    message_handler,
    attach_transport,
):
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
globalThis.WebSocket = {{ OPEN: 1 }};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const sent = [];
const record = (raw) => sent.push(JSON.parse(raw));
{attach_transport}

client.requestToolFollowUp("response_1");
client.requestResponse();
await client[{json.dumps(message_handler)}](JSON.stringify({{
  type: "error",
  error: {{ type: "invalid_tool_follow_up", message: "stale origin" }},
}}));

if (client._queuedToolFollowUp !== null || client._createQueue.length !== 0) {{
  throw new Error("invalid tool follow-up left the client response-locked");
}}
if (!client._pendingCreateId || sent.length !== 2) {{
  throw new Error(`queued response did not resume: ${{JSON.stringify(sent)}}`);
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
    ("module_path", "class_name", "message_handler", "attach_transport"),
    [
        (
            "./demo/ws/s2s-ws-client.js",
            "S2sWsRealtimeClient",
            "_onWsMessage",
            "client._ws = { readyState: WebSocket.OPEN, send() {} };",
        ),
        (
            "./demo/rtc/s2s-rtc-client.js",
            "S2sRtcRealtimeClient",
            "_onDcMessage",
            'client._dc = { readyState: "open", send() {} };',
        ),
    ],
)
def test_demo_clients_preserve_tool_follow_up_when_interrupts_are_disabled(
    module_path,
    class_name,
    message_handler,
    attach_transport,
):
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
globalThis.WebSocket = {{ OPEN: 1 }};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
{attach_transport}
let turnStarted = null;
client.addEventListener("user-turn-started", (event) => {{ turnStarted = event.detail; }});
await client[{json.dumps(message_handler)}](JSON.stringify({{
  type: "session.created",
  session: {{
    audio: {{ input: {{ turn_detection: {{ type: "server_vad", interrupt_response: false }} }} }},
  }},
}}));
client.requestToolFollowUp("response_1");
await client[{json.dumps(message_handler)}](JSON.stringify({{
  type: "input_audio_buffer.speech_started",
  item_id: "item_1",
  audio_start_ms: 0,
}}));

if (!client._queuedToolFollowUp) {{
  throw new Error("non-interrupting speech discarded the queued tool follow-up");
}}
if (turnStarted?.interruptResponse !== false) {{
  throw new Error(`interrupt setting was not exposed: ${{JSON.stringify(turnStarted)}}`);
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
    ("module_path", "class_name", "attach_transport"),
    [
        (
            "./demo/ws/s2s-ws-client.js",
            "S2sWsRealtimeClient",
            "client._ws = { readyState: WebSocket.OPEN, send: record };",
        ),
        (
            "./demo/rtc/s2s-rtc-client.js",
            "S2sRtcRealtimeClient",
            'client._dc = { readyState: "open", send: record };',
        ),
    ],
)
def test_demo_clients_tag_tool_follow_up_images(module_path, class_name, attach_transport):
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
globalThis.WebSocket = {{ OPEN: 1 }};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const sent = [];
const record = (raw) => sent.push(JSON.parse(raw));
{attach_transport}
client.sendUserImage("data:image/jpeg;base64,aW1hZ2U=", "response_1");

const item = sent[0]?.item;
if (item?.s2s_tool_follow_up_for_response_id !== "response_1") {{
  throw new Error(`tool image origin was not tagged: ${{JSON.stringify(sent)}}`);
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
    ("module_path", "class_name", "message_handler", "attach_transport"),
    [
        (
            "./demo/ws/s2s-ws-client.js",
            "S2sWsRealtimeClient",
            "_onWsMessage",
            "client._ws = { readyState: WebSocket.OPEN, send: record };",
        ),
        (
            "./demo/rtc/s2s-rtc-client.js",
            "S2sRtcRealtimeClient",
            "_onDcMessage",
            'client._dc = { readyState: "open", send: record };',
        ),
    ],
)
@pytest.mark.parametrize(
    "lifecycle_order",
    [
        ["collision", "created", "done"],
        ["created", "collision", "done"],
        ["created", "done", "collision"],
        ["collision", "speech_started", "speech_stopped", "transcription_completed"],
    ],
)
def test_demo_clients_replay_create_after_automatic_response_collision(
    module_path,
    class_name,
    message_handler,
    attach_transport,
    lifecycle_order,
):
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
globalThis.WebSocket = {{ OPEN: 1 }};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const sent = [];
const record = (raw) => sent.push(JSON.parse(raw));
{attach_transport}
const deliver = (event) => client[{json.dumps(message_handler)}](JSON.stringify(event));

client.requestResponse({{ image: "data:image/png;base64,image_a" }});
const firstCreateId = sent[1]?.response?.metadata?.s2s_demo_create_id;
if (!firstCreateId) throw new Error(`initial create has no correlation id: ${{JSON.stringify(sent)}}`);
client.requestResponse({{ image: "data:image/png;base64,image_b" }});
const collision = {{
  type: "error",
  error: {{ type: "conversation_already_has_active_response" }},
}};
const automaticDone = {{
  type: "response.done",
  response: {{ id: "response_automatic", status: "completed" }},
}};
const automaticCreated = {{ type: "response.created", response: {{ id: "response_automatic" }} }};
const speechStarted = {{ type: "input_audio_buffer.speech_started", item_id: "item_1" }};
const speechStopped = {{ type: "input_audio_buffer.speech_stopped", item_id: "item_1" }};
const transcriptionCompleted = {{
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_1",
  transcript: "Next question",
}};
// Replay all server orderings around a pending response, including barge-in
// cancellation before that response emits any lifecycle events.
const lifecycle = {{
  collision,
  created: automaticCreated,
  done: automaticDone,
  speech_started: speechStarted,
  speech_stopped: speechStopped,
  transcription_completed: transcriptionCompleted,
}};
for (const name of {json.dumps(lifecycle_order)}) {{
  await deliver(lifecycle[name]);
  if (name === "collision") {{
    client.requestResponse({{ image: "data:image/png;base64,image_c" }});
  }}
  if (name === "created" && client._pendingCreateId && client._pendingCreateId !== firstCreateId) {{
    throw new Error("automatic response incorrectly acknowledged the explicit create");
  }}
}}
const retryCreateId = sent
  .filter((event) => event.type === "response.create")[1]
  ?.response?.metadata?.s2s_demo_create_id;
if (!retryCreateId || retryCreateId === firstCreateId) {{
  throw new Error(`retry create has an invalid correlation id: ${{JSON.stringify(sent)}}`);
}}
await deliver({{
  type: "response.created",
  response: {{
    id: "response_retry_a",
    metadata: {{ s2s_demo_create_id: retryCreateId }},
  }},
}});
await deliver({{
  type: "response.done",
  response: {{ id: "response_retry_a", status: "completed" }},
}});
const secondQueuedCreateId = sent
  .filter((event) => event.type === "response.create")[2]
  ?.response?.metadata?.s2s_demo_create_id;
if (!secondQueuedCreateId) throw new Error(`second queued create was not sent: ${{JSON.stringify(sent)}}`);
await deliver({{
  type: "response.created",
  response: {{
    id: "response_b",
    metadata: {{ s2s_demo_create_id: secondQueuedCreateId }},
  }},
}});
await deliver({{
  type: "response.done",
  response: {{ id: "response_b", status: "completed" }},
}});

const creates = sent.filter((event) => event.type === "response.create");
if (creates.length !== 4) {{
  throw new Error(`rejected and queued creates were not sent once each: ${{JSON.stringify(sent)}}`);
}}
const eventTypes = sent.map((event) => event.type);
const expectedTypes = [
  "conversation.item.create",
  "response.create",
  "response.create",
  "conversation.item.create",
  "response.create",
  "conversation.item.create",
  "response.create",
];
if (JSON.stringify(eventTypes) !== JSON.stringify(expectedTypes)) {{
  throw new Error(`rejected create did not retain queue priority: ${{JSON.stringify(sent)}}`);
}}
const images = sent
  .filter((event) => event.type === "conversation.item.create")
  .map((event) => event.item.content[0].image_url);
if (JSON.stringify(images) !== JSON.stringify([
  "data:image/png;base64,image_a",
  "data:image/png;base64,image_b",
  "data:image/png;base64,image_c",
])) {{
  throw new Error(`queued images were reordered or resent: ${{JSON.stringify(images)}}`);
}}
if (client._createQueue.length !== 0 || !client._pendingCreateId) {{
  throw new Error(`replayed create has invalid lock state: queue=${{client._createQueue.length}} pending=${{client._pendingCreateId}}`);
}}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("status", ["cancelled", "failed", "incomplete"])
def test_unsuccessful_response_discards_tool_batch(status):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = f"""
const {{ ToolCallBatcher }} = await import("./demo/tool-call-batcher.js");
let resolveExecution;
const execution = new Promise((resolve) => {{ resolveExecution = resolve; }});
const timeline = [];
const batches = new ToolCallBatcher(
  () => {{ timeline.push("output"); }},
  () => {{ timeline.push("response.create"); }},
);
batches.add("response_1", execution);
const flush = batches.finish("response_1", {json.dumps(status)});
if (flush !== null) throw new Error("unsuccessful response returned a flush");
resolveExecution({{ callId: "call_1", output: "unused" }});
await execution;
await Promise.resolve();
if (JSON.stringify(timeline) !== JSON.stringify(["response.create"])) {{
  throw new Error(`discarded response unexpectedly flushed: ${{JSON.stringify(timeline)}}`);
}}
if (batches.finish("response_1", "completed") !== null) {{
  throw new Error("discarded response remained registered");
}}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
