import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENTS = [
    ("./demo/s2s-realtime-client.js", "S2sRealtimeClient", "_onTransportEvent"),
]

CASES = [
    pytest.param(
        """
await deliver({ type: "input_audio_buffer.speech_started", item_id: "item_1" });
await deliver({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_1", delta: "hel" });
await deliver({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_1", delta: "lo" });
await deliver({
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_1",
  transcript: "hello",
});
await deliver({ type: "input_audio_buffer.speech_started", item_id: "item_2" });
await deliver({
  type: "conversation.item.input_audio_transcription.delta",
  item_id: "item_2",
  delta: "hello again",
});

assertEqual(transcripts, [
  { role: "user", text: "hel", partial: true, itemId: "item_1" },
  { role: "user", text: "hello", partial: true, itemId: "item_1" },
  { role: "user", text: "hello", partial: false, itemId: "item_1" },
  { role: "user", text: "hello again", partial: true, itemId: "item_2" },
], "new stream transcript events");
""",
        id="new-stream-after-completion",
    ),
    pytest.param(
        """
await deliver({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_1", delta: "hel" });
await deliver({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_2", delta: "wor" });
await deliver({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_1", delta: "lo" });
await deliver({
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_1",
  transcript: "hello",
});
await deliver({ type: "conversation.item.input_audio_transcription.delta", item_id: "item_2", delta: "ld" });

assertEqual(transcripts, [
  { role: "user", text: "hel", partial: true, itemId: "item_1" },
  { role: "user", text: "wor", partial: true, itemId: "item_2" },
  { role: "user", text: "hello", partial: true, itemId: "item_1" },
  { role: "user", text: "hello", partial: false, itemId: "item_1" },
  { role: "user", text: "world", partial: true, itemId: "item_2" },
], "overlapping transcript events");
if (client._userTranscriptByItem.has("item_1")) throw new Error("completed item was not pruned");
if (client._userTranscriptByItem.get("item_2") !== "world") throw new Error("active prefix was lost");
""",
        id="overlapping-items",
    ),
    pytest.param(
        """
for (let index = 0; index < 130; index += 1) {
  await deliver({ type: "input_audio_buffer.speech_started", item_id: `item_${index}` });
}
if (client._userTranscriptByItem.size !== 0) {
  throw new Error(`direct-audio turns retained ${client._userTranscriptByItem.size} entries`);
}
""",
        id="direct-audio-is-lazy",
    ),
    pytest.param(
        """
for (let index = 0; index < 130; index += 1) {
  await deliver({
    type: "conversation.item.input_audio_transcription.delta",
    item_id: `item_${index}`,
    delta: String(index),
  });
}
if (client._userTranscriptByItem.size !== 130) throw new Error("unterminated transcripts were evicted");
await deliver({
  type: "conversation.item.input_audio_transcription.delta",
  item_id: "item_0",
  delta: " more",
});
if (client._userTranscriptByItem.get("item_0") !== "0 more") throw new Error("late delta lost its prefix");
await deliver({
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_0",
  transcript: "",
});
const completed = transcripts.at(-1);
assertEqual(completed, { role: "user", text: "", partial: false, itemId: "item_0" }, "empty final");
if (client._userTranscriptByItem.has("item_0") || client._userTranscriptByItem.size !== 129) {
  throw new Error("completed transcript was not pruned");
}
""",
        id="retain-until-empty-completion",
    ),
    pytest.param(
        """
client._status = "connected";
await deliver({ type: "input_audio_buffer.speech_started", item_id: "item_empty" });
await deliver({ type: "input_audio_buffer.speech_stopped", item_id: "item_empty" });
await deliver({
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_empty",
  transcript: "",
});
assertEqual(statuses, ["user-speaking", "processing", "connected"], "empty-final statuses");
assertEqual(transcripts, [], "empty-final transcripts");

const assertPendingStatus = async () => {
  client._status = "processing";
  await deliver({ type: "conversation.item.input_audio_transcription.completed", transcript: "" });
  if (client.status !== "processing") throw new Error(`pending status was overwritten: ${client.status}`);
};
client._activeResponseId = "response_1";
await assertPendingStatus();
client._activeResponseId = "";
client._responseRequested = true;
await assertPendingStatus();
client._responseRequested = false;
""",
        id="empty-final-status",
    ),
    pytest.param(
        """
client._status = "connected";
await deliver({ type: "input_audio_buffer.speech_started", item_id: "item_old" });
await deliver({ type: "input_audio_buffer.speech_stopped", item_id: "item_old" });
await deliver({ type: "input_audio_buffer.speech_started", item_id: "item_current" });
await deliver({ type: "input_audio_buffer.speech_stopped", item_id: "item_current" });

await deliver({
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_old",
  transcript: "",
});
if (client.status !== "processing") {
  throw new Error(`late empty final overwrote current status: ${client.status}`);
}

await deliver({
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_current",
  transcript: "",
});
if (client.status !== "connected") {
  throw new Error(`current empty final did not restore status: ${client.status}`);
}
""",
        id="late-empty-final-keeps-current-status",
    ),
    pytest.param(
        """
await deliver({
  type: "conversation.item.input_audio_transcription.delta",
  item_id: "item_1",
  delta: "hallucinated",
});
await deliver({
  type: "conversation.item.input_audio_transcription.completed",
  item_id: "item_1",
  transcript: "",
});
assertEqual(transcripts, [
  { role: "user", text: "hallucinated", partial: true, itemId: "item_1" },
  { role: "user", text: "", partial: false, itemId: "item_1" },
], "partial cleared by empty final");
if (client._userTranscriptByItem.has("item_1")) throw new Error("empty final did not prune state");
""",
        id="empty-final-clears-partial",
    ),
]


@pytest.mark.parametrize("module_path,class_name,message_handler", CLIENTS)
@pytest.mark.parametrize("case_body", CASES)
def test_demo_input_transcription_cases(module_path, class_name, message_handler, case_body):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    harness = f"""
globalThis.localStorage = {{ getItem() {{ return null; }} }};
globalThis.CustomEvent = class CustomEvent extends Event {{
  constructor(type, init = {{}}) {{
    super(type);
    this.detail = init.detail;
  }}
}};
const {{ {class_name} }} = await import({json.dumps(module_path)});
const client = new {class_name}({{
  transport: "websocket",
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
  callsUrl: "api/calls",
}});
const transcripts = [];
const statuses = [];
client.addEventListener("transcript", (event) => transcripts.push(event.detail));
client.addEventListener("status", (event) => statuses.push(event.detail.status));
const deliver = async (event) => {{
  await client[{json.dumps(message_handler)}](event);
}};
const assertEqual = (actual, expected, label) => {{
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {{
    throw new Error(`${{label}}: expected ${{JSON.stringify(expected)}}, got ${{JSON.stringify(actual)}}`);
  }}
}};
"""
    subprocess.run(
        [node, "--input-type=module", "-e", harness + case_body],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
