import assert from "node:assert/strict";
import test from "node:test";
import { readTurnLatency } from "../turn-latency.js";
import { S2sRealtimeClient } from "../s2s-realtime-client.js";

const timing = {
  version: 1, turn_id: "turn_3", turn_revision: 0, response_key: "key-3", status: "completed",
  stt_s: 0.181284, llm_s: 1.241907, tts_ttfa_s: 0.121775, e2e_s: 1.613482, mlx_lock_wait_s: 0,
  vad_decision_s: 0.32, smart_analysis_s: 0.03, smart_grace_s: 2,
  smart_delay_s: 0.6, smart_wait_s: 0, smart_status: "incomplete",
};
const response = (value) => ({ status: "completed", metadata: { "speech_to_speech.turn_latency": value } });

test("timings retain raw precision and unavailable stages", () => {
  assert.deepEqual(readTurnLatency(response(JSON.stringify(timing))), timing);
  assert.equal(readTurnLatency(response(JSON.stringify({ ...timing, stt_s: null }))).stt_s, null);
  const legacy = { ...timing };
  for (const field of ["vad_decision_s", "smart_analysis_s", "smart_grace_s", "smart_delay_s", "smart_wait_s", "smart_status"]) delete legacy[field];
  assert.deepEqual(readTurnLatency(response(JSON.stringify(legacy))), legacy);
});

test("absent, malformed, unknown-version and invalid measurements are ignored", () => {
  for (const value of [undefined, {}, "{", "null", "[]", ...[
    { version: 3 }, { status: "cancelled" }, { e2e_s: -1 }, { llm_s: "1.2" },
    { stt_s: undefined }, { turn_revision: 0.5 }, { response_key: "" },
    { smart_wait_s: -1 }, { smart_status: "unknown" },
  ].map((patch) => JSON.stringify({ ...timing, ...patch }))]) {
    assert.equal(readTurnLatency(response(value)), null);
  }
  assert.equal(readTurnLatency(response('{"version":1,"e2e_s":1e999}')), null);
});

for (const transport of ["websocket", "webrtc"]) {
  test(`${transport} exports timings only at response.done and keeps response identity`, () => {
    const client = new S2sRealtimeClient({ transport, directUrl: "ws://unused" });
    const finished = [];
    client.addEventListener("response-finished", (event) => finished.push(event.detail));
    client._onTransportEvent({ type: "response.created", response: { id: "r1", ...response(JSON.stringify(timing)) } });
    assert.equal(finished.length, 0);
    for (const [id, data] of [["r1", timing], ["r2", { ...timing, response_key: "key-4", stt_s: null }]]) {
      client._onTransportEvent({ type: "response.done", response: { id, ...response(JSON.stringify(data)) } });
    }
    assert.equal(finished[0].responseId, "r1");
    assert.equal(finished[0].latency.stt_s, timing.stt_s);
    assert.equal(finished[1].responseId, "r2");
    assert.equal(finished[1].latency.stt_s, null);
    client._onTransportEvent({ type: "response.done", response: { id: "legacy", status: "completed" } });
    assert.equal(finished[2].latency, null);
  });
}

test("version 2 exports only the simplified measurements", () => {
  const current = { version: 2, turn_id: "turn_1", turn_revision: 0, response_key: "r", status: "completed",
    e2e_s: 1.1, vad_decision_s: 0.1, smart_status: "complete", stt_s: 0.2, llm_s: 0.4, tts_ttfa_s: 0.1, hold_s: 0.25 };
  assert.deepEqual(readTurnLatency(response(JSON.stringify(current))), current);
  assert.equal(readTurnLatency(response(JSON.stringify({ ...current, hold_s: -1 }))), null);
});
