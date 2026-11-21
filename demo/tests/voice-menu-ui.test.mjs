import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");
const VOICES = [
  { id: "af_heart", name: "Heart (female)", group: "American English" },
  { id: "bm_fable", name: "Fable (male)", group: "British English" },
  { id: "ef_dora", name: "Dora (female)", group: "Spanish" },
];

/** The page and a speech server on one origin: the server lists its engine's voices, or (voices: null) none. */
async function serve(t, voices) {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    if (name === "/v1/realtime/capabilities") {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ pipelines: [{ index: 0, tts: { voices, default_voice: voices ? "bm_fable" : null } }] }));
      return;
    }
    if (name.startsWith("/api/")) {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ allowDirect: true, enabled: false, s2sUrl: `ws://127.0.0.1:${server.address().port}/v1/realtime` }));
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
  return `http://127.0.0.1:${server.address().port}/`;
}

test("the voice menu shows the server engine's voices, keeps a valid choice and replaces one the engine lacks", async (t) => {
  const url = await serve(t, VOICES);
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.route("https://**/*", route => route.abort());
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));

  await page.addInitScript(() => { if (!sessionStorage.getItem("seeded")) { localStorage.setItem("s2s.ws.voice", "Vivian"); sessionStorage.setItem("seeded", "1"); } });
  await page.goto(url);
  await page.waitForFunction(() => document.querySelector("#voice option[value='af_heart']"));
  assert.deepEqual(await page.$$eval("#voice option", options => options.map(o => o.value)), ["af_heart", "bm_fable", "ef_dora"]);
  assert.deepEqual(await page.$$eval("#voice optgroup", groups => groups.map(g => g.label)),
    ["American English", "British English", "Spanish"]);
  assert.equal(await page.locator("#voice").inputValue(), "bm_fable", "Vivian is not a Kokoro voice: the server's own");
  assert.equal(await page.evaluate(() => localStorage.getItem("s2s.ws.voice")), "bm_fable");

  await page.evaluate(() => localStorage.setItem("s2s.ws.voice", "ef_dora"));
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#voice option[value='af_heart']"));
  assert.equal(await page.locator("#voice").inputValue(), "ef_dora", "a voice the engine has is kept");
  assert.deepEqual(errors, []);
});

test("a server that does not list voices keeps the page's own voice menu", async (t) => {
  const url = await serve(t, null);
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await page.route("https://**/*", route => route.abort());
  await page.goto(url);
  await page.waitForLoadState("networkidle");
  assert.ok((await page.$$eval("#voice option", options => options.map(o => o.value))).includes("Aiden"));
});
