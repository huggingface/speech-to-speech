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
const batches = new ToolCallBatcher((results) => {{
  for (const result of results) timeline.push(`output:${{result.callId}}:${{result.output}}`);
  timeline.push("response.create");
}});
batches.add("response_1", executions[0]);
batches.add("response_1", executions[1]);

let flush = null;
if ({json.dumps(response_finishes_first)}) {{
  flush = batches.finish("response_1", "completed");
}}
for (const index of {json.dumps(completion_order)}) {{
  pending[index].resolve();
  await Promise.resolve();
  if (timeline.length !== 0) {{
    throw new Error(`batch flushed before both response and tools completed: ${{JSON.stringify(timeline)}}`);
  }}
}}
await Promise.all(executions);
if (!flush) flush = batches.finish("response_1", "completed");
if (!flush) throw new Error("response batch was not registered");
await flush;

const expected = ["output:call_1:first", "output:call_2:second", "response.create"];
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
const batches = new ToolCallBatcher(() => {{ timeline.push("response.create"); }});
batches.add("response_1", execution);
const flush = batches.finish("response_1", {json.dumps(status)});
if (flush !== null) throw new Error("unsuccessful response returned a flush");
resolveExecution({{ callId: "call_1", output: "unused" }});
await execution;
await Promise.resolve();
if (timeline.length !== 0) {{
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
