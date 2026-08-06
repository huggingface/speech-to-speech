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


def test_ducker_reject_is_reversible_and_candidate_scoped():
    _run_node(
        """
const { ReversibleAudioDucker } = await import("./demo/audio-ducker.js");
const targets = [];
const gain = {
  value: 1,
  cancelScheduledValues() {},
  setValueAtTime(value) { this.value = value; },
  linearRampToValueAtTime(value) { this.value = value; targets.push(value); },
};
const ducker = new ReversibleAudioDucker({ gain }, { currentTime: 1 });

ducker.candidateStarted("candidate_1");
ducker.candidateStarted("candidate_2");
ducker.candidateRejected("candidate_1");
if (targets.at(-1) !== 0.12) throw new Error("stale reject released newer candidate");
ducker.candidateRejected("candidate_2");
if (targets.at(-1) !== 1) throw new Error("matching reject did not restore output");

ducker.candidateStarted("candidate_3");
ducker.speechStarted(true);
ducker.candidateRejected("candidate_3");
if (targets.at(-1) !== 0) throw new Error("confirmed speech was released by stale reject");
ducker.speechStopped();
if (targets.at(-1) !== 1) throw new Error("speech stop did not restore output");

ducker.candidateStarted("candidate_4");
ducker.speechStarted(false);
if (targets.at(-1) !== 1) throw new Error("non-interrupting speech should not mute output");

const beforeDisabledCandidate = targets.length;
ducker.candidateStarted("candidate_5", false);
if (targets.length !== beforeDisabledCandidate) throw new Error("disabled candidate changed output gain");

ducker.candidateStarted("candidate_6");
ducker.reset();
if (targets.at(-1) !== 1) throw new Error("reset did not restore output");
"""
    )


def test_websocket_client_maps_candidate_and_confirmation_events():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
globalThis.WebSocket = { OPEN: 1 };
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) { super(type); this.detail = init.detail; }
};
const { S2sWsRealtimeClient } = await import("./demo/ws/s2s-ws-client.js");
const client = new S2sWsRealtimeClient({ voice: "Aiden", instructions: "Be helpful." });
const calls = [];
client._audioDucker = {
  candidateStarted(id, interrupt) { calls.push(["candidate", id, interrupt]); },
  candidateRejected(id) { calls.push(["reject", id]); },
  speechStarted(interrupt) { calls.push(["start", interrupt]); },
  speechStopped() { calls.push(["stop"]); },
};
const playback = [];
client._playbackNode = { port: { postMessage(msg) { playback.push(msg.kind); } } };

await client._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_candidate_started",
  candidate_id: "candidate_1",
  interrupt_response: true,
}));
await client._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_candidate_rejected",
  candidate_id: "candidate_1",
}));
await client._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_started",
  item_id: "item_1",
  audio_start_ms: 0,
  interrupt_response: false,
}));

if (JSON.stringify(calls) !== JSON.stringify([
  ["candidate", "candidate_1", true],
  ["reject", "candidate_1"],
  ["start", false],
])) throw new Error(`unexpected ducker calls: ${JSON.stringify(calls)}`);
if (playback.includes("clear")) throw new Error("non-interrupting speech cleared playback");

await client._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_started",
  item_id: "item_2",
  audio_start_ms: 0,
  interrupt_response: true,
}));
if (playback.at(-1) !== "clear") throw new Error("confirmed interruption did not clear playback");

await client._onWsMessage(JSON.stringify({
  type: "response.done",
  response: { id: "response_1", status: "cancelled", output: [] },
}));
if (JSON.stringify(calls.at(-1)) !== JSON.stringify(["start", true])) {
  throw new Error("response.done released confirmed speech before speech_stopped");
}
"""
    )


def test_webrtc_client_maps_candidate_confirm_and_stop_events():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) { super(type); this.detail = init.detail; }
};
const { S2sRtcRealtimeClient } = await import("./demo/rtc/s2s-rtc-client.js");
const client = new S2sRtcRealtimeClient({
  callsUrl: "/api/calls",
  voice: "Aiden",
  instructions: "Be helpful.",
});
const calls = [];
client._audioDucker = {
  candidateStarted(id, interrupt) { calls.push(["candidate", id, interrupt]); },
  candidateRejected(id) { calls.push(["reject", id]); },
  speechStarted(interrupt) { calls.push(["start", interrupt]); },
  speechStopped() { calls.push(["stop"]); },
};

client._onDcMessage(JSON.stringify({
  type: "input_audio_buffer.speech_candidate_started",
  candidate_id: "candidate_1",
  interrupt_response: true,
}));
client._onDcMessage(JSON.stringify({
  type: "input_audio_buffer.speech_started",
  item_id: "item_1",
  interrupt_response: true,
}));
client._onDcMessage(JSON.stringify({
  type: "input_audio_buffer.speech_stopped",
  item_id: "item_1",
}));

if (JSON.stringify(calls) !== JSON.stringify([
  ["candidate", "candidate_1", true],
  ["start", true],
  ["stop"],
])) throw new Error(`unexpected ducker calls: ${JSON.stringify(calls)}`);
"""
    )
