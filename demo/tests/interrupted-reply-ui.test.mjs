import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");

test("history rewrites an interrupted reply to what the user heard", async (t) => {
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
  const page = await browser.newPage({ viewport: { width: 1280, height: 850 }, reducedMotion: "reduce" });
  await page.route("https://**/*", (route) => route.abort());
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.evaluate(async () => {
    const { ChatView } = await import("/ui/chat.js");
    document.body.classList.remove("booting");
    window.chat = new ChatView();
    // A completed reply: the row is released before the heard text arrives.
    chat.onTranscript({ role: "assistant", text: "Sure, here are five ways to improve it.", partial: false, responseId: "r1" });
    chat.onResponseFinished({ responseId: "r1", status: "completed" });
    chat.onTranscript({ role: "assistant", text: "First, tighten the thesis.", partial: false, responseId: "r2" });
    chat.onResponseFinished({ responseId: "r2", status: "cancelled", transcript: "First, tighten the thesis." });
    chat._openPanel();
  });
  const rows = page.locator(".hist-msg.assistant");
  assert.equal(await rows.locator(".hist-note").count(), 1);

  await page.evaluate(() => {
    chat.onAssistantHeard({ responseId: "r1", text: "Sure, here are" });
    chat.onAssistantHeard({ responseId: "r2", text: "" });
  });

  // History rows keep the reply and strike through what was never said.
  const struckIn = (body) => body.locator("s.unheard").allTextContents();
  assert.equal(await rows.nth(0).locator(".hist-body").textContent(), "Sure, here are five ways to improve it.");
  assert.deepEqual(await struckIn(rows.nth(0).locator(".hist-body")), ["five ways to improve it."]);
  assert.equal(await rows.nth(0).locator(".hist-note").textContent(), "Interrupted");
  assert.deepEqual(await struckIn(rows.nth(1).locator(".hist-body")), ["First, tighten the thesis."]);
  assert.equal(await rows.nth(1).locator(".hist-note").textContent(), "Interrupted");
  // Only the bubble animates the strike.
  assert.equal(await rows.locator("s.unheard").first().evaluate((el) => getComputedStyle(el).animationName), "none");

  // The reply's bubble, still on screen, strikes through the words never said.
  const bubbles = page.locator(".bubble.assistant:not(.out)");
  const struck = (bubble) => bubble.locator(".bubble-body s.unheard").allTextContents();
  assert.equal(await bubbles.count(), 2);
  assert.equal(await bubbles.nth(0).locator(".bubble-body").textContent(), "Sure, here are five ways to improve it.");
  assert.deepEqual(await struck(bubbles.nth(0)), ["five ways to improve it."]);
  assert.deepEqual(await struck(bubbles.nth(1)), ["First, tighten the thesis."]);
  assert.equal(await bubbles.locator(".bubble-note").count(), 0);
  if (process.env.S2S_UI_SCREENSHOTS) {
    await page.evaluate(() => chat._closePanel());
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/interrupted-reply.png` });
  }
  // A later correction is measured against the whole reply again.
  await page.evaluate(() => chat.onAssistantHeard({ responseId: "r2", text: "First," }));
  assert.deepEqual(await struck(bubbles.nth(1)), ["tighten the thesis."]);
  assert.equal(await bubbles.nth(1).locator(".bubble-body").textContent(), "First, tighten the thesis.");
  // A faded bubble is not brought back.
  await page.evaluate(() => {
    const bubble = document.querySelector(".bubble.assistant");
    chat._dismissBubble(bubble);
    chat.onAssistantHeard({ responseId: "r1", text: "Sure," });
  });
  assert.equal(await bubbles.count(), 1);
  assert.deepEqual(await struckIn(rows.nth(0).locator(".hist-body")), ["here are five ways to improve it."]);
  // Text that does not match the bubble is shown as is, with nothing struck.
  await page.evaluate(() => {
    chat.onTranscript({ role: "assistant", text: "Hello there.", partial: false, responseId: "r3" });
    chat.onAssistantHeard({ responseId: "r3", text: "Goodbye" });
  });
  const latest = bubbles.last();
  assert.equal(await latest.locator(".bubble-body").textContent(), "Goodbye");
  assert.deepEqual(await struck(latest), []);
  // An emoji in the heard text still matches.
  await page.evaluate(() => {
    chat.onTranscript({ role: "assistant", text: "Great 🐶 dogs are the best.", partial: false, responseId: "r4" });
    chat.onAssistantHeard({ responseId: "r4", text: "Great 🐶" });
  });
  assert.deepEqual(await struck(bubbles.last()), ["dogs are the best."]);
  await page.close();
});
