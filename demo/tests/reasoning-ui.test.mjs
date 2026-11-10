import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const root = path.resolve(import.meta.dirname, "..");

test("reasoning picker supports touch, keyboard, persistence, and server default", async (t) => {
  const server = createServer(async (req, res) => {
    const name = new URL(req.url, "http://localhost").pathname;
    if (name.startsWith("/api/")) {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ allowDirect: true }));
      return;
    }
    const file = name === "/vendor/openai-realtime-agents.umd.js"
      ? path.join(root, "node_modules/@openai/agents-realtime/dist/bundle/openai-realtime-agents.umd.js")
      : path.resolve(root, `.${name === "/" ? "/index.html" : name}`);
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    try {
      let content = await readFile(file, "utf8");
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
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("https://**/*", (route) => route.abort());
    await page.addInitScript(() => {
      localStorage.setItem("s2s.ws.tools", JSON.stringify({ camera_snapshot: false }));
      // Corrupt/obsolete preferences fall back to the startup setting.
      if (!sessionStorage.getItem("initialized")) {
        localStorage.setItem("s2s.ws.reasoningEffort", "invalid");
        sessionStorage.setItem("initialized", "yes");
      }
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(() => !document.body.classList.contains("booting"));
    const rtc = await page.evaluate(async () => {
      const { S2sRealtimeClient } = await import("/s2s-realtime-client.js");
      const { RealtimeSession, OpenAIRealtimeWebRTC } = window.OpenAIAgentsRealtime;
      const client = new S2sRealtimeClient({ transport: "webrtc", reasoningEffort: "low", voice: "Aiden" });
      const sent = [];
      const transport = new OpenAIRealtimeWebRTC({ useInsecureApiKey: true });
      transport.sendEvent = (event) => sent.push(event);
      Object.defineProperty(transport, "status", { value: "connected" });
      client._transport = transport;
      client._agent = client._buildAgent();
      client._session = new RealtimeSession(client._agent, {
        transport, config: client._sessionConfig(), tracingDisabled: true,
      });
      transport.updateSessionConfig(await client._session.getInitialSessionConfig());
      const initial = sent.at(-1).session.reasoning;
      client.setReasoningEffort("high");
      const live = sent.at(-1).session.reasoning;
      await client._session.updateAgent(client._buildAgent());
      const afterAgentUpdate = sent.at(-1).session.reasoning;
      client.setReasoningEffort(null);
      await client._session.updateAgent(client._buildAgent());
      return { initial, live, afterAgentUpdate, reset: sent.at(-1).session.reasoning };
    });
    assert.deepEqual(rtc, {
      initial: { effort: "low" }, live: { effort: "high" }, afterAgentUpdate: { effort: "high" }, reset: null,
    });
    const button = page.locator("#reasoning-btn");
    const picker = page.locator("#reasoning-picker");
    assert.match(await button.innerText(), /Server default/);
    await button.click();
    await page.waitForFunction(() => document.querySelector("#reasoning-btn").getAttribute("aria-expanded") === "true");
    assert.equal(await picker.getByRole("radio").count(), 7);
    await picker.getByRole("radio", { name: "High", exact: true }).check();
    assert.match(await button.innerText(), /High/);
    assert.equal(await page.evaluate(() => localStorage.getItem("s2s.ws.reasoningEffort")), "high");
    await page.reload();
    await page.waitForFunction(() => !document.body.classList.contains("booting"));
    assert.match(await button.innerText(), /High/);
    await button.focus();
    await page.keyboard.press("Enter");
    await picker.getByRole("radio", { name: "High", exact: true }).waitFor({ state: "visible" });
    await page.waitForFunction(() => document.activeElement?.getAttribute("value") === "high");
    await page.keyboard.press("ArrowDown");
    assert.match(await button.innerText(), /Extra high/);
    await button.click();
    await page.keyboard.press("Escape");
    await picker.waitFor({ state: "hidden" });
    assert.match(await button.innerText(), /Extra high/);
    await button.click();
    const box = await picker.boundingBox();
    assert.ok(box.x >= 0 && box.x + box.width <= width);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    if (process.env.S2S_UI_SCREENSHOTS) await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/reasoning-picker-${width}.png` });
    await picker.getByRole("radio", { name: "Server default", exact: true }).check();
    assert.equal(await page.evaluate(() => localStorage.getItem("s2s.ws.reasoningEffort")), null);
    await page.reload();
    await page.waitForFunction(() => !document.body.classList.contains("booting"));
    assert.match(await button.innerText(), /Server default/);
    if (process.env.S2S_UI_SCREENSHOTS) await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/reasoning-${width}.png` });
    assert.deepEqual(errors, []);
    await page.close();
  }
});
