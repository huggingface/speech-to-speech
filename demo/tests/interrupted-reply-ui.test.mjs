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

  assert.equal(await rows.nth(0).locator(".hist-body").textContent(), "Sure, here are");
  assert.equal(await rows.nth(0).locator(".hist-note").textContent(), "Interrupted");
  assert.equal(await rows.nth(1).locator(".hist-body").isHidden(), true);
  assert.equal(await rows.nth(1).locator(".hist-note").textContent(), "Interrupted before you heard it");
  await page.close();
});
