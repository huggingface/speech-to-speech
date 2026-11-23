import assert from "node:assert/strict";
import test from "node:test";
import { S2sRealtimeClient } from "../s2s-realtime-client.js";

for (const transport of ["websocket", "webrtc"]) {
  test(`viseme extension is exposed to avatar consumers over ${transport}`, () => {
    const client = new S2sRealtimeClient({ transport });
    let received;
    client.addEventListener("visemes", (event) => { received = event.detail; });
    const event = {
      type: "speech_to_speech.output_audio.visemes", response_id: "response-a",
      item_id: "item-a", output_index: 0, content_index: 0,
      visemes: [{ viseme: 21, start_s: 0, end_s: 0.1 }],
    };
    client._onTransportEvent(event);
    assert.equal(received, event);
  });
}
