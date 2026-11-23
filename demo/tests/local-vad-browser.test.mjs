import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
const vadAssets = new Set(["bundle.min.js", "vad.worklet.bundle.min.js", "silero_vad_v6.onnx"]);

test("real browser Silero inference detects speech and releases playback using only local assets", { timeout: 30_000 }, async (t) => {
  const requests = [];
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    requests.push(name);
    let file;
    if (name === "/") {
      res.setHeader("Content-Type", "text/html");
      res.end("<!doctype html><title>Local VAD test</title>");
      return;
    } else if (name === "/speech.wav") {
      file = path.resolve(root, "../src/speech_to_speech/TTS/ref_audio.wav");
    } else if (name.startsWith("/vendor/local-vad/")) {
      const asset = path.basename(name);
      const packageName = vadAssets.has(asset) ? "@ricky0123/vad-web" : "onnxruntime-web";
      file = path.resolve(root, "node_modules", packageName, "dist", asset);
    } else {
      file = path.resolve(root, `.${name}`);
      if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    }
    try {
      const mime = file.endsWith(".wasm") ? "application/wasm"
        : /\.m?js$/.test(file) ? "text/javascript" : "application/octet-stream";
      res.setHeader("Content-Type", mime);
      res.end(await readFile(file));
    } catch { res.writeHead(404).end(); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const browser = await chromium.launch({ headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
  t.after(() => browser.close());
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "warning") errors.push(message.text());
  });
  await page.route("https://**/*", (route) => route.abort());
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async () => {
    const { LocalVad } = await import("/ws/local-vad.js");
    const context = new AudioContext();
    await context.resume();
    const destination = context.createMediaStreamDestination();
    const source = context.createBufferSource();
    source.buffer = await context.decodeAudioData(await (await fetch("/speech.wav")).arrayBuffer());
    source.connect(destination);
    const changes = [];
    const vad = new LocalVad({
      audioContext: context, stream: destination.stream,
      isEnabled: () => true,
      onChange: (active) => changes.push(active),
    });
    await vad.start();
    const loaded = Boolean(vad.instance);
    source.start(0, 0, Math.min(3, source.buffer.duration));
    const deadline = performance.now() + 10_000;
    while (performance.now() < deadline && !(changes.includes(true) && changes.at(-1) === false)) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    const restoredBeforeClose = changes.includes(true) && changes.at(-1) === false;
    await vad.close();
    const trackAlive = destination.stream.getAudioTracks()[0].readyState === "live";
    const contextAlive = context.state === "running";
    destination.stream.getTracks().forEach((track) => track.stop());
    await context.close();
    return { loaded, changes, restoredBeforeClose, trackAlive, contextAlive };
  });
  assert.equal(result.loaded, true, errors.join("\n"));
  assert.ok(result.changes.includes(true), "real speech should duck playback");
  assert.equal(result.restoredBeforeClose, true, "real silence must restore volume before cleanup");
  assert.equal(result.changes.at(-1), false, "silence should restore playback");
  assert.equal(result.trackAlive, true, "VAD cleanup must preserve the conversation mic");
  assert.equal(result.contextAlive, true, "VAD cleanup must preserve the conversation audio context");
  assert.deepEqual(errors, []);
  assert.ok(requests.includes("/vendor/local-vad/silero_vad_v6.onnx"));
  assert.ok(requests.some((url) => url.endsWith(".wasm")));
});
