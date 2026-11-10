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
  const browser = await chromium.launch({ headless: true, args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });
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
    assert.equal(await page.locator("#chat-history").evaluate((el) => el.classList.contains("hide-reasoning")), true);
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

    // Exercise the real main.js lifecycle and adapter event path. Hold the
    // backend's acknowledgement until after checking that selection alone
    // cannot show a successful confirmation.
    await page.evaluate(async () => {
      const { S2sRealtimeClient } = await import("/s2s-realtime-client.js?v=reasoning-language-20261110");
      S2sRealtimeClient.prototype.connect = async function () {
        window.demoClient = this;
        this._setStatus("connected");
      };
      localStorage.setItem("s2s.ws.directUrl", "ws://unused/v1/realtime");
    });
    // Use the settings form to update the in-memory connection target too.
    await page.locator("#settings-btn").click();
    const showReasoning = page.getByRole("switch", { name: "Show reasoning", exact: true });
    assert.equal(await showReasoning.isChecked(), false, "reasoning is hidden by default");
    if (process.env.S2S_UI_SCREENSHOTS) {
      await showReasoning.scrollIntoViewIfNeeded();
      await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/reasoning-settings-${width}.png` });
    }
    await showReasoning.focus();
    await page.keyboard.press("Space");
    await page.locator("#lb-url").fill("ws://unused/v1/realtime");
    await page.locator("#settings-save").click();
    await page.locator("#main-circle").click();
    await page.waitForFunction(() => Boolean(window.demoClient));
    const status = page.locator("#reasoning-status");
    assert.equal(await status.getAttribute("data-state"), "pending");
    const echo = (reasoning) => page.evaluate((reasoning) => {
      demoClient._onTransportEvent({ type: "session.updated", session: { reasoning } });
    }, reasoning);
    await echo(null);
    assert.equal(await status.innerText(), "✓ Server confirmed: Server default");
    await button.click();
    await picker.getByRole("radio", { name: "High", exact: true }).check();
    assert.equal(await status.getAttribute("data-state"), "pending");
    await echo({ effort: "low" });
    assert.equal(await status.getAttribute("data-state"), "unconfirmed");
    assert.match(await status.innerText(), /Server reports Low/);
    await echo({ effort: "high" });
    assert.equal(await status.innerText(), "✓ Server confirmed: High");
    if (process.env.S2S_UI_SCREENSHOTS) await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/reasoning-confirmed-${width}.png` });
    await button.click();
    await picker.getByRole("radio", { name: "Off", exact: true }).check();
    assert.equal(await status.getAttribute("data-state"), "pending");
    await echo(null);
    assert.equal(await status.getAttribute("data-state"), "unconfirmed");
    await echo({ effort: "none" });
    assert.equal(await status.innerText(), "✓ Server confirmed: Off");
    await page.evaluate(() => demoClient._onTransportEvent({
      type: "error", error: { message: "Unsupported reasoning effort" },
    }));
    assert.equal(await status.innerText(), "Server reported a reasoning error");

    await page.evaluate(() => {
      demoClient._onTransportEvent({ type: "response.created", response: { id: "reply-reasoning" } });
      demoClient._onTransportEvent({ type: "speech_to_speech.reasoning.delta", response_id: "reply-reasoning", kind: "reasoning", delta: "First, compare the options.\n" });
    });
    await page.locator("#chat-btn").click();
    await page.evaluate(() => document.querySelector(".chat-panel-inner").getAnimations().forEach((animation) => animation.finish()));
    const row = page.locator(".hist-msg.assistant").first();
    const reasoningDetails = row.locator(".hist-reasoning");
    assert.equal(await reasoningDetails.getAttribute("open"), null);
    assert.equal(await reasoningDetails.locator(".reasoning-text").isVisible(), false);
    assert.equal(await page.locator(".bubble.assistant").count(), 0, "reasoning must never become an answer bubble");
    await reasoningDetails.locator("summary").focus();
    await page.keyboard.press("Enter");
    await page.evaluate(() => {
      demoClient._onTransportEvent({ type: "speech_to_speech.reasoning.delta", response_id: "reply-reasoning", kind: "reasoning", delta: "Treat <img src=x onerror=alert(1)> as plain text." });
      demoClient._onTransportEvent({ type: "response.output_audio_transcript.done", response_id: "reply-reasoning", item_id: "answer", transcript: "Option A is the better fit." });
      demoClient._onTransportEvent({ type: "response.done", response: { id: "reply-reasoning", status: "completed", output: [] } });
    });
    assert.equal(await page.locator(".hist-msg.assistant").count(), 1, "reasoning and answer share one row");
    assert.equal(await reasoningDetails.getAttribute("open"), "", "streaming must preserve the expanded state");
    assert.match(await reasoningDetails.innerText(), /compare the options/);
    assert.equal(await reasoningDetails.locator("img").count(), 0);
    assert.equal(await row.locator(".hist-body").innerText(), "Option A is the better fit.");
    if (process.env.S2S_UI_SCREENSHOTS) await page.screenshot({ path: `${process.env.S2S_UI_SCREENSHOTS}/reasoning-text-${width}.png` });
    await page.evaluate(() => {
      demoClient._onTransportEvent({ type: "speech_to_speech.reasoning.delta", response_id: "reply-summary", kind: "summary", delta: "A brief provider summary." });
      demoClient._onTransportEvent({ type: "response.done", response: { id: "reply-summary", status: "cancelled", output: [] } });
    });
    const summaryRow = page.locator(".hist-msg.assistant").last();
    assert.equal(await summaryRow.locator(".hist-reasoning summary").innerText(), "Reasoning summary");
    assert.equal(await summaryRow.locator(".hist-reasoning").getAttribute("open"), null);
    assert.equal(await summaryRow.locator(".hist-note").textContent(), "Interrupted");
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.locator("#chat-panel-close").click();

    // The saved display preference hides existing and future sections without
    // changing the answer text, and restoring it retains received reasoning.
    await page.locator("#settings-btn").click();
    assert.equal(await showReasoning.isChecked(), true);
    await showReasoning.focus();
    await page.keyboard.press("Space");
    assert.equal(await page.evaluate(() => localStorage.getItem("s2s.ws.showReasoning")), "off");
    assert.equal(await page.locator("#chat-history").evaluate((el) => el.classList.contains("hide-reasoning")), true);
    await page.locator("#settings-save").click();
    await page.locator("#chat-btn").click();
    assert.equal(await reasoningDetails.isVisible(), false);
    assert.equal(await summaryRow.isVisible(), false, "no empty reasoning-only row when hidden");
    assert.equal(await row.locator(".hist-body").innerText(), "Option A is the better fit.");
    await page.evaluate(() => demoClient._onTransportEvent({ type: "speech_to_speech.reasoning.delta", response_id: "hidden-reply", kind: "reasoning", delta: "Reasoning received while hidden." }));
    const hiddenRow = page.locator(".hist-msg.assistant").last();
    assert.equal(await hiddenRow.isVisible(), false);
    await page.locator("#chat-panel-close").click();
    await page.locator("#settings-btn").click();
    await page.locator("label.switch:has(#show-reasoning)").click();
    assert.equal(await page.evaluate(() => localStorage.getItem("s2s.ws.showReasoning")), "on");
    await page.locator("#settings-save").click();
    await page.locator("#chat-btn").click();
    assert.equal(await reasoningDetails.isVisible(), true);
    assert.equal(await hiddenRow.isVisible(), true);
    assert.equal(await hiddenRow.locator(".hist-reasoning").getAttribute("open"), null);
    await page.locator("#chat-panel-close").click();
    await page.locator("#stop-btn").click();
    assert.equal(await status.getAttribute("data-state"), "idle");
    // Late acknowledgements from a closed conversation cannot reconfirm it.
    await echo({ effort: "none" });
    assert.equal(await status.getAttribute("data-state"), "idle");
    await page.reload();
    await page.waitForFunction(() => !document.body.classList.contains("booting"));
    await page.locator("#settings-btn").click();
    assert.equal(await showReasoning.isChecked(), true, "enabled preference survives a reload");
    await showReasoning.focus();
    await page.keyboard.press("Space");
    await page.locator("#settings-save").click();
    await page.reload();
    await page.waitForFunction(() => !document.body.classList.contains("booting"));
    await page.locator("#settings-btn").click();
    assert.equal(await showReasoning.isChecked(), false, "disabled preference survives a reload");
    assert.deepEqual(errors, []);
    await page.close();
  }
});
