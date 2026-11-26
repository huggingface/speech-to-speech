import assert from "node:assert/strict";
import test from "node:test";
import { errorMessage } from "../ui/dom.js";

test("opaque WebSocket events become an actionable connection error", () => {
  assert.match(errorMessage(new Event("error")), /Voice connection failed/);
  assert.doesNotMatch(errorMessage(new Event("error")), /object Event/);
  assert.equal(errorMessage(new Error("Microphone permission denied")), "Microphone permission denied");
  assert.equal(errorMessage({ error: new Error("Microphone permission denied") }), "Microphone permission denied");
  assert.equal(errorMessage({ detail: { error: new Error("Session is busy") } }), "Session is busy");
  assert.equal(errorMessage("Server closed the connection"), "Server closed the connection");
  const cycle = {}; cycle.error = cycle;
  assert.match(errorMessage(cycle), /Voice connection failed/);
});
