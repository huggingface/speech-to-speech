import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
test("language pickers stay synchronized and persist on desktop and mobile", async (t) => {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    if (name.startsWith("/api/")) {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ allowDirect: true, enabled: false }));
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
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  for (const width of [1280, 360]) {
    const page = await browser.newPage({ viewport: { width, height: 850 } });
    await page.route("https://**/*", route => route.abort());
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(() => document.querySelector('#main-language').options.length > 1);
    await page.locator('#main-language').selectOption('es');
    assert.equal(await page.locator('#conversation-language').inputValue(), 'es');
    assert.equal(await page.evaluate(() => localStorage.getItem('s2s.ws.language')), 'es');
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#main-language').value === 'es');
    await page.locator('#main-language').selectOption('auto');
    assert.equal(await page.evaluate(() => localStorage.getItem('s2s.ws.language')), 'auto');
    const box = await page.locator('#main-language').boundingBox();
    assert.ok(box.width > 0 && box.x >= 0 && box.x + box.width <= width);
    assert.deepEqual(errors, []);
    await page.close();
  }
});
