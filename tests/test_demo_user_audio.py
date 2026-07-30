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
client.addEventListener("user-audio", (event) => { recording = event.detail; });
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
"""
    )
