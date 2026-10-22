import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");

for (const tools of [false, true]) {
test(`server tools=${tools} restores controls without automatic camera access`, async (t) => {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    if (name === "/api/config") {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ tools, search: tools, lb: false, auth: false,
        s2sUrl: "ws://localhost:8765/v1/realtime", rtc: false }));
      return;
    }
    if (name === "/api/me") {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ enabled: false }));
      return;
    }
    const file = path.resolve(root, `.${name === "/" ? "/index.html" : name}`);
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      const content = await readFile(file);
      res.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : "text/html");
      res.end(content);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.route("https://**/*", (route) => route.abort());
  // No real camera/microphone is accessed by this regression test.
  await page.addInitScript(() => {
    window.mediaRequests = [];
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      window.mediaRequests.push(constraints);
      throw new Error("Test device is unavailable");
    };
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction((tools) => document.querySelector("#tools-btn").hidden === !tools, tools);
  assert.equal(await page.locator("#tools-btn").isVisible(), tools);
  if (tools) {
    await page.locator("#tools-btn").click();
    assert.equal(await page.locator("#tool-web").isChecked(), true);
    assert.equal(await page.locator("#tool-cam").isChecked(), false);
    assert.equal(await page.locator("#search-key").isDisabled(), true);
    await page.locator("#tools-close").click();
    await page.locator("#settings-btn").click();
    assert.equal(await page.locator("#playback-buffer-ms").isVisible(), true);
    assert.equal(await page.locator("#instructions").isVisible(), true);
  }
  assert.deepEqual(await page.evaluate(() => window.mediaRequests), []);
  assert.deepEqual(errors, []);
});

}
