import assert from "node:assert/strict";
import test from "node:test";
import { readTurnLatency } from "../turn-latency.js";
import { S2sRealtimeClient } from "../s2s-realtime-client.js";

const timing = {
  version: 1, turn_id: "turn_3", turn_revision: 0, response_key: "key-3", status: "completed",
  stt_s: 0.181284, llm_s: 1.241907, tts_ttfa_s: 0.121775, e2e_s: 1.613482, mlx_lock_wait_s: 0,
};
const response = (value) => ({ status: "completed", metadata: { "speech_to_speech.turn_latency": value } });

test("timings retain raw precision and unavailable stages", () => {
  assert.deepEqual(readTurnLatency(response(JSON.stringify(timing))), timing);
  assert.equal(readTurnLatency(response(JSON.stringify({ ...timing, stt_s: null }))).stt_s, null);
});

test("absent, malformed, unknown-version and invalid measurements are ignored", () => {
  for (const value of [undefined, {}, "{", "null", "[]", ...[
    { version: 2 }, { status: "cancelled" }, { e2e_s: -1 }, { llm_s: "1.2" },
    { stt_s: undefined }, { turn_revision: 0.5 }, { response_key: "" },
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
