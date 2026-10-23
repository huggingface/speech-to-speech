import assert from "node:assert/strict";
import test from "node:test";
import { PlaybackLatency } from "../playback-latency.js";

test("E2E stops at first output render, independently of generation completion", () => {
  const metric = new PlaybackLatency(24000);
  metric.capture(48000, 2000);
  metric.speechStopped(750); // Speech ended at local time 1750ms.
  metric.responseCreated("first");
  const context = { getOutputTimestamp: () => ({ performanceTime: 2500, contextTime: 10 }) };
  assert.deepEqual(metric.rendered("first", 10.25, context), { responseId: "first", seconds: 1 });
  assert.equal(metric.rendered("first", 99, context), null);
});

test("greeting, cancelled responses, and unavailable clocks have no fabricated E2E", () => {
  const metric = new PlaybackLatency();
  const context = { getOutputTimestamp: () => ({ performanceTime: 2000, contextTime: 0 }) };
  metric.responseCreated("greeting");
  assert.equal(metric.rendered("greeting", 0, context), null);
  metric.capture(48000, 1000);
  metric.speechStopped(1000);
  metric.responseCreated("reply");
  assert.equal(metric.rendered("reply", 0, {}), null);
  metric.interrupt();
  assert.equal(metric.rendered("reply", 0, context), null);
});
