import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
const timing = {
  version: 1, turn_id: "turn_1", turn_revision: 0, response_key: "key-1", status: "completed",
  stt_s: 0.18, llm_s: 1.24, tts_ttfa_s: 0.12, e2e_s: 1.61, mlx_lock_wait_s: 0,
};

test("history shows per-response server timings on desktop and phone", async (t) => {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    const file = path.resolve(root, `.${name === "/" ? "/index.html" : name}`);
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      let content = await readFile(file, "utf8");
      // Render the actual demo shell without network/auth/audio startup.
      if (file.endsWith(".html")) content = content.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "");
      res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html");
      res.end(content);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  for (const width of [1280, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 850 }, reducedMotion: "reduce" });
    await page.route("https://**/*", (route) => route.abort());
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async (timing) => {
      const { ChatView } = await import("/ui/chat.js");
      const { S2sRealtimeClient } = await import("/s2s-realtime-client.js");
      document.body.classList.remove("booting");
      window.chat = new ChatView();
      window.client = new S2sRealtimeClient({ transport: "webrtc" });
      client.addEventListener("response-finished", (event) => chat.onResponseFinished(event.detail));
      window.finish = (id, timing, status = "completed") => client._onTransportEvent({
        type: "response.done", response: { id, status, metadata: timing ? {
          "speech_to_speech.turn_latency": JSON.stringify(timing),
        } : {}, output: [] },
      });
      chat.onTranscript({ role: "assistant", text: "The sky looks blue because air scatters blue light more strongly.", partial: false, responseId: "r1" });
      finish("r1", timing);
      chat._openPanel();
    }, timing);
    await page.evaluate(() => document.querySelector(".chat-panel-inner").getAnimations().forEach((animation) => animation.finish()));
    assert.equal(await page.locator(".hist-msg.assistant").count(), 1);
    const summary = page.locator(".hist-timings summary").first();
    assert.match(await summary.textContent(), /First audio 1.61 s/);
    await summary.focus();
    await page.keyboard.press("Enter");
    assert.equal(await page.locator(".hist-timings").first().getAttribute("open"), "");
    assert.match(await page.locator(".hist-timings dl").innerText(), /0.00 s/);
    assert.match(await page.locator(".hist-timings").innerText(), /excluding browser playback/);
    await page.waitForFunction(() => {
      const box = document.querySelector(".hist-timings summary").getBoundingClientRect();
      return box.right <= innerWidth;
    });
    const box = await summary.boundingBox();
    assert.ok(box.height >= 44 && box.x >= 0 && box.x + box.width <= width);
    if (process.env.S2S_UI_SCREENSHOTS) await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/timings-${width}.png` });
    await page.evaluate((timing) => {
      // A tool-only terminal still gets a record, distinct from its follow-up.
      finish("tool", { ...timing, response_key: "tool", stt_s: null, e2e_s: null, tts_ttfa_s: null });
      chat.onTranscript({ role: "assistant", text: "Checking that for you…", partial: true, responseId: "cancelled" });
      finish("cancelled", { ...timing, response_key: "cancelled", status: "cancelled" }, "cancelled");
      chat.onTranscript({ role: "assistant", text: "An older server reply.", partial: false, responseId: "legacy" });
      finish("legacy", null);
    }, timing);
    assert.equal(await page.locator(".hist-timings").count(), 3);
    const tool = page.locator(".hist-timings").nth(1);
    await tool.locator("summary").click();
    assert.match(await tool.innerText(), /Unavailable/);
    assert.equal(await page.locator(".hist-note").textContent(), "Interrupted");
    assert.equal(await page.locator(".hist-msg.assistant").last().locator(".hist-timings").count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.evaluate(() => { chat.clear(); chat.reset(); });
    assert.equal(await page.locator(".hist-timings").count(), 0);
    await page.close();
  }
});
