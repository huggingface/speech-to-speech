import assert from "node:assert/strict";
import test from "node:test";
import { withSpaceAccess } from "../connection-url.js";
import { errorMessage } from "../ui/dom.js";

test("private Space page access follows its same-origin WebSocket", () => {
  const page = "https://demo.hf.space/?__theme=system&__sign=temporary-grant";
  const socket = new URL(withSpaceAccess("wss://demo.hf.space/v1/realtime?model=s2s", page));
  assert.equal(socket.searchParams.get("__sign"), "temporary-grant");
  assert.equal(socket.searchParams.get("model"), "s2s");
  assert.equal(socket.searchParams.has("__theme"), false);
});

test("Space access is never forwarded to another host or an insecure socket", () => {
  const page = "https://demo.hf.space/?__sign=temporary-grant";
  for (const socket of ["wss://other.example/v1/realtime", "ws://demo.hf.space/v1/realtime"]) {
    assert.equal(withSpaceAccess(socket, page), socket);
  }
  assert.equal(withSpaceAccess("ws://localhost:8765/v1/realtime", "http://localhost:7860/?__sign=test"), "ws://localhost:8765/v1/realtime");
});

test("an existing socket grant is preserved and unsigned local URLs work", () => {
  const socket = "wss://demo.hf.space/v1/realtime?__sign=socket-grant";
  assert.equal(new URL(withSpaceAccess(socket, "https://demo.hf.space/?__sign=page-grant")).searchParams.get("__sign"), "socket-grant");
  assert.equal(withSpaceAccess("ws://localhost:8765/v1/realtime", undefined), "ws://localhost:8765/v1/realtime");
});

test("opaque WebSocket events become an actionable connection error", () => {
  assert.match(errorMessage(new Event("error")), /Voice connection failed/);
  assert.doesNotMatch(errorMessage(new Event("error")), /object Event/);
  assert.equal(errorMessage({ error: new Error("Microphone permission denied") }), "Microphone permission denied");
  assert.equal(errorMessage({ detail: { error: new Error("Session is busy") } }), "Session is busy");
  const cycle = {}; cycle.error = cycle;
  assert.match(errorMessage(cycle), /Reload the Space/);
});
