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


def test_sent_audio_recorder_uses_backend_vad_boundaries():
    _run_node(
        """
const { SentAudioRecorder } = await import("./demo/ws/user-audio-recorder.js");
const recorder = new SentAudioRecorder({
  sampleRate: 16000,
  preRollMs: 1000,
  maxBufferMs: 10000,
});

// 300 ms of recognizable PCM, delivered in the same 40 ms frames as the demo.
const samples = new Int16Array(4800);
for (let i = 0; i < samples.length; i++) samples[i] = i - 2400;
for (let offset = 0; offset < samples.length; offset += 640) {
  recorder.append(samples.slice(offset, Math.min(offset + 640, samples.length)).buffer);
}

recorder.speechStarted({ itemId: "item_1", audioStartMs: 50 });
const recording = recorder.speechStopped({ itemId: "item_1", audioEndMs: 250 });
if (!recording) throw new Error("expected a recording");
if (Math.abs(recording.durationMs - 200) > 0.001) {
  throw new Error(`unexpected duration: ${recording.durationMs}`);
}
if (recording.truncated) throw new Error("recording should include its full onset");

const wav = new DataView(await recording.audio.arrayBuffer());
const ascii = (offset, length) =>
  String.fromCharCode(...new Uint8Array(wav.buffer, offset, length));
if (ascii(0, 4) !== "RIFF" || ascii(8, 4) !== "WAVE") {
  throw new Error("invalid WAV header");
}
if (wav.getUint32(24, true) !== 16000) throw new Error("wrong sample rate");
if (wav.getUint32(40, true) !== 6400) throw new Error("wrong PCM payload length");
// 50 ms * 16 samples/ms = sample 800.
if (wav.getInt16(44, true) !== samples[800]) {
  throw new Error(`wrong first sample: ${wav.getInt16(44, true)}`);
}
"""
    )


def test_reopened_item_replaces_recording_with_accumulated_audio():
    _run_node(
        """
const { SentAudioRecorder } = await import("./demo/ws/user-audio-recorder.js");
const recorder = new SentAudioRecorder({ sampleRate: 16000 });
const frame = (value, samples) => {
  const pcm = new Int16Array(samples);
  pcm.fill(value);
  return pcm.buffer;
};

recorder.append(frame(100, 1600));
recorder.speechStarted({ itemId: "item_same", audioStartMs: 0 });
const first = recorder.speechStopped({ itemId: "item_same", audioEndMs: 100 });
recorder.append(frame(200, 1600));
recorder.speechStarted({ itemId: "item_same", audioStartMs: 100 });
const reopened = recorder.speechStopped({ itemId: "item_same", audioEndMs: 200 });

if (!first || !reopened) throw new Error("expected both recordings");
if (first.durationMs !== 100 || reopened.durationMs !== 200) {
  throw new Error(`unexpected accumulated durations: ${first.durationMs}, ${reopened.durationMs}`);
}
const wav = new DataView(await reopened.audio.arrayBuffer());
if (wav.getInt16(44, true) !== 100) throw new Error("first segment missing");
if (wav.getInt16(44 + 1600 * 2, true) !== 200) throw new Error("reopened segment missing");
"""
    )


def test_websocket_client_emits_audio_only_user_turn():
    _run_node(
        """
globalThis.localStorage = { getItem() { return null; } };
globalThis.WebSocket = { OPEN: 1 };
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) {
    super(type);
    this.detail = init.detail;
  }
};
const { S2sWsRealtimeClient } = await import("./demo/ws/s2s-ws-client.js");
const client = new S2sWsRealtimeClient({
  voice: "Aiden",
  instructions: "Be helpful.",
  directUrl: "ws://unused",
});
client._ws = { readyState: 1, send() {} };
client._sessionConfigured = true;

let recording = null;
const turnEvents = [];
client.addEventListener("user-audio", (event) => { recording = event.detail; });
client.addEventListener("user-turn-started", (event) => {
  turnEvents.push(["started", event.detail.itemId]);
});
client.addEventListener("user-turn-stopped", (event) => {
  turnEvents.push(["stopped", event.detail.itemId]);
});
const frame = new Int16Array(640);
frame.fill(123);
for (let i = 0; i < 5; i++) client._onMicChunk(frame.buffer);
await client._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_started",
  item_id: "item_audio_only",
  audio_start_ms: 40,
}));
for (let i = 0; i < 3; i++) client._onMicChunk(frame.buffer);
await client._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_stopped",
  item_id: "item_audio_only",
  audio_end_ms: 280,
}));

if (!recording) throw new Error("user-audio event was not emitted");
if (recording.itemId !== "item_audio_only") throw new Error("wrong item association");
if (Math.abs(recording.durationMs - 240) > 0.001) {
  throw new Error(`unexpected emitted duration: ${recording.durationMs}`);
}
if (recording.audio.type !== "audio/wav") throw new Error("recording is not WAV");
const expectedTurns = [
  ["started", "item_audio_only"],
  ["stopped", "item_audio_only"],
];
if (JSON.stringify(turnEvents) !== JSON.stringify(expectedTurns)) {
  throw new Error(`unexpected turn events: ${JSON.stringify(turnEvents)}`);
}
"""
    )


def test_voice_bubble_reaper_reschedules_shortened_deadline():
    _run_node(
        """
const { ChatView } = await import("./demo/ui/chat.js");
const bubble = {};
const view = Object.create(ChatView.prototype);
view._bubbleExpiry = new WeakMap();
view._reaperHandle = 0;
view._bubbleStack = {
  querySelector() { return bubble; },
};
let reapCount = 0;
view._reapBubbles = () => {
  view._reaperHandle = 0;
  reapCount += 1;
};

// Reproduce the listening -> sending transition: the second, shorter deadline
// must replace the already scheduled long fail-safe.
view._bumpDismiss(bubble, 1000);
await new Promise((resolve) => setTimeout(resolve, 10));
view._bumpDismiss(bubble, 40);
await new Promise((resolve) => setTimeout(resolve, 100));

if (reapCount !== 1) {
  throw new Error(`shortened deadline did not reschedule reaper: ${reapCount}`);
}
"""
    )


def test_assistant_activity_dismisses_pending_voice_bubble():
    _run_node(
        """
const { ChatView } = await import("./demo/ui/chat.js");
const view = Object.create(ChatView.prototype);
const bubble = {
  isConnected: true,
  classList: {
    contains(name) { return name === "voice"; },
  },
};
view._activeUserBubble = bubble;
view._activeUserItemId = "item_voice";
view._assistantDismissedUserItemId = "";
view._bubbleExpiry = new WeakMap();
view._reaperHandle = 0;
view._bubbleStack = {
  querySelector() { return null; },
};
let dismissed = null;
view._dismissBubble = (element) => { dismissed = element; };

view.onAssistantActivity();
if (dismissed !== bubble) throw new Error("pending voice bubble was not dismissed");
if (view._activeUserBubble !== null) throw new Error("active voice bubble was not released");
if (view._assistantDismissedUserItemId !== "item_voice") {
  throw new Error("dismissed item was not remembered");
}
"""
    )


def test_late_user_turn_stop_does_not_recreate_dismissed_voice_bubble():
    _run_node(
        """
const { ChatView } = await import("./demo/ui/chat.js");
const view = Object.create(ChatView.prototype);
const bubble = {
  isConnected: true,
  classList: {
    contains(name) { return name === "voice"; },
  },
};
view._activeUserBubble = bubble;
view._activeUserItemId = "item_voice";
view._assistantDismissedUserItemId = "";
view._bubbleExpiry = new WeakMap();
view._reaperHandle = 0;
view._bubbleStack = {
  querySelector() { return null; },
};
view._dismissBubble = () => {};
view._scheduleBubbleReaper = () => {};
let spawned = 0;
view._spawnVoiceBubble = () => {
  spawned += 1;
  return bubble;
};

// RTC audio can become audible before the ordered data channel delivers the
// speech_stopped event. The late stop must not recreate the dismissed bubble.
view.onAssistantActivity();
view.onUserTurnStopped({ itemId: "item_voice" });

if (spawned !== 0) {
  throw new Error(`late stop recreated ${spawned} voice bubble(s)`);
}
if (view._activeUserBubble !== null) {
  throw new Error("late stop restored the active voice bubble");
}

// A genuine reopened turn may reuse the item id and must clear the tombstone.
view.onUserTurnStarted({ itemId: "item_voice" });
if (spawned !== 1 || view._activeUserBubble !== bubble) {
  throw new Error("reopened turn did not create a fresh listening bubble");
}
"""
    )
