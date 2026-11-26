// @ts-check
/**
 * Minimal OpenAI-Realtime WebSocket client for the speech-to-speech backend,
 * used as the coach's ears and voice:
 *
 * - Ears: mic PCM16 @ 24 kHz streams up; the server VAD commits each turn and
 *   transcribes it, but `create_response: false` stops the backend from
 *   answering on its own. The app grades the transcript instead.
 * - Voice: `say(text)` sends an out-of-band `response.create`
 *   (`conversation: "none"`) whose only job is to speak the given script.
 *   Nothing piles up in the backend's conversation history.
 *
 * Events: status, queue, speech-started, speech-stopped, user-partial,
 * user-final, response-started, audio, visemes, assistant-text,
 * response-done, mic-level, error.
 */

const RATE = 24000;

const RELAY_INSTRUCTIONS =
  "You are the voice of Viva, a friendly quiz coach. The user message is a script. " +
  "Say it out loud exactly as written, word for word, then stop. " +
  "Never answer it, comment on it, translate it, or add any words.";

/** @param {Uint8Array} bytes */
function toBase64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, /** @type {any} */ (bytes.subarray(i, i + 0x8000)));
  }
  return btoa(s);
}

/** @param {string} b64 */
function fromBase64(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class RealtimeVoice extends EventTarget {
  /**
   * @param {{ audioCtx: AudioContext, voice?: string }} opts
   */
  constructor(opts) {
    super();
    this.ctx = opts.audioCtx;
    this.voice = opts.voice || "";
    /** @type {WebSocket | null} */
    this.ws = null;
    this.status = "idle";
    this.activeResponse = "";
    /** @type {{ text: string, kind: string }[]} */
    this.pending = [];
    this.awaitingCreate = false;
    /** @type {{ text: string, kind: string } | null} */
    this.inflight = null;
    this.closed = false;
    this.micNode = null;
    this.micStream = null;
    this.muted = false;
  }

  /** @param {string} type @param {any} [detail] */
  emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  /** @param {string} status @param {any} [extra] */
  setStatus(status, extra) {
    this.status = status;
    this.emit("status", { status, ...extra });
  }

  // ── Connection ──────────────────────────────────────────────────────────

  /**
   * @param {{ mode: string, wsUrl?: string }} cfg
   */
  async connect(cfg) {
    this.closed = false;
    this.setStatus("connecting");
    let url = cfg.wsUrl || "";
    if (cfg.mode === "lb") url = await this.handshake();
    if (!url) throw Object.assign(new Error("No speech server configured."), { code: "no-backend" });
    if (!/^wss?:\/\//.test(url)) url = `ws://${url.replace(/\/$/, "")}/v1/realtime`;
    await this.openSocket(url);
  }

  async handshake() {
    let resp = await fetch("api/session", { method: "POST" });
    let body = await resp.json().catch(() => ({}));
    for (;;) {
      if (this.closed) throw Object.assign(new Error("Cancelled"), { code: "aborted" });
      if (resp.status === 402) throw Object.assign(new Error("Daily voice time used up. Sign in or come back tomorrow."), { code: "limit" });
      if (resp.status === 401) throw Object.assign(new Error("Please sign in with Hugging Face to talk."), { code: "login", loginUrl: body.loginUrl });
      if (resp.status === 503) throw Object.assign(new Error("All voice servers are busy. Try again shortly."), { code: "busy" });
      if (resp.status === 404 && body.state === "expired") {
        resp = await fetch("api/session", { method: "POST" });
        body = await resp.json().catch(() => ({}));
        continue;
      }
      if (!resp.ok) throw Object.assign(new Error(body.detail || `Voice handshake failed (${resp.status})`), { code: "handshake" });
      if (body.connect_url) return body.connect_url;
      if (body.state === "queued") {
        this.setStatus("queued", { position: body.position });
        this.emit("queue", { position: body.position });
        await sleep(Math.max(1000, (body.poll_interval_s || 2) * 1000));
        resp = await fetch(`api/queue/${encodeURIComponent(body.queue_id)}`);
        const next = await resp.json().catch(() => ({}));
        body = { ...next, queue_id: next.queue_id || body.queue_id };
        continue;
      }
      throw Object.assign(new Error("Unexpected handshake reply."), { code: "handshake" });
    }
  }

  /** @param {string} url */
  openSocket(url) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      this.ws = ws;
      let configured = false;
      const timer = setTimeout(() => reject(new Error("Voice server did not answer.")), 20000);
      ws.onmessage = (e) => {
        let ev;
        try {
          ev = JSON.parse(e.data);
        } catch {
          return;
        }
        if (!configured && ev.type === "session.created") {
          configured = true;
          this.sendSessionUpdate();
        }
        if (ev.type === "session.updated" && this.status !== "ready") {
          clearTimeout(timer);
          this.setStatus("ready");
          resolve(undefined);
        }
        this.onEvent(ev);
      };
      ws.onerror = () => {
        clearTimeout(timer);
        reject(new Error("Could not reach the voice server."));
      };
      ws.onclose = (e) => {
        clearTimeout(timer);
        if (this.status !== "ready") reject(new Error(`Voice server closed the connection (${e.code}).`));
        if (!this.closed) {
          this.setStatus("closed", { code: e.code });
          this.emit("error", { message: "The voice connection dropped.", code: "closed" });
        }
      };
    });
  }

  sendSessionUpdate() {
    /** @type {Record<string, any>} */
    const output = { format: { type: "audio/pcm", rate: RATE } };
    if (this.voice) output.voice = this.voice;
    this.send({
      type: "session.update",
      session: {
        type: "realtime",
        instructions: RELAY_INSTRUCTIONS,
        audio: {
          input: {
            format: { type: "audio/pcm", rate: RATE },
            turn_detection: { type: "server_vad", create_response: false, interrupt_response: true },
          },
          output,
        },
      },
    });
  }

  /** @param {any} msg */
  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  close() {
    this.closed = true;
    this.pending = [];
    try {
      this.ws?.close(1000);
    } catch {}
    this.ws = null;
    this.micNode?.disconnect();
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micNode = null;
    this.micStream = null;
    this.setStatus("closed");
  }

  // ── Microphone ──────────────────────────────────────────────────────────

  /** @param {MediaStream} stream */
  async startMic(stream) {
    this.micStream = stream;
    await this.ctx.audioWorklet.addModule("static/worklets/mic-capture.js");
    const node = new AudioWorkletNode(this.ctx, "mic-capture", { numberOfInputs: 1, numberOfOutputs: 0 });
    node.port.onmessage = (e) => {
      const { pcm, rms } = e.data || {};
      if (pcm) {
        this.send({ type: "input_audio_buffer.append", audio: toBase64(new Uint8Array(pcm)) });
        this.emit("mic-level", { rms: this.muted ? 0 : rms });
      }
    };
    this.ctx.createMediaStreamSource(stream).connect(node);
    this.micNode = node;
    node.port.postMessage({ kind: "mute", value: this.muted });
  }

  /** Half-duplex gate: send silence instead of the mic while the coach talks. */
  setMuted(value) {
    this.muted = !!value;
    this.micNode?.port.postMessage({ kind: "mute", value: this.muted });
  }

  // ── Speaking ────────────────────────────────────────────────────────────

  /**
   * Speak a script. Queued behind any response already in flight.
   * @param {string} text @param {string} [kind]
   */
  say(text, kind = "line") {
    text = (text || "").trim();
    if (!text) return;
    this.pending.push({ text, kind });
    this.pump();
  }

  /** Drop queued lines and cancel the line being generated. */
  stopSpeaking() {
    this.pending = [];
    if (this.activeResponse || this.awaitingCreate) this.send({ type: "response.cancel" });
  }

  pump() {
    if (this.activeResponse || this.awaitingCreate || !this.pending.length) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const line = /** @type {{text: string, kind: string}} */ (this.pending.shift());
    const { text, kind } = line;
    this.inflight = line;
    this.awaitingCreate = true;
    this.send({
      type: "response.create",
      response: {
        conversation: "none",
        instructions: RELAY_INSTRUCTIONS,
        output_modalities: ["audio"],
        metadata: { kind },
        input: [{ type: "message", role: "user", content: [{ type: "input_text", text }] }],
      },
    });
  }

  /** @param {any} ev */
  onEvent(ev) {
    switch (ev.type) {
      case "input_audio_buffer.speech_started":
        this.emit("speech-started", {});
        break;
      case "input_audio_buffer.speech_stopped":
        this.emit("speech-stopped", {});
        break;
      case "conversation.item.input_audio_transcription.delta":
        this.emit("user-partial", { itemId: ev.item_id, delta: ev.delta || "" });
        break;
      case "conversation.item.input_audio_transcription.completed":
        this.emit("user-final", { itemId: ev.item_id, transcript: (ev.transcript || "").trim() });
        break;
      case "response.created":
        this.awaitingCreate = false;
        this.activeResponse = ev.response?.id || "resp";
        this.emit("response-started", { responseId: this.activeResponse });
        break;
      case "response.output_audio.delta": {
        const bytes = fromBase64(ev.delta);
        const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
        this.emit("audio", { responseId: ev.response_id || this.activeResponse, pcm, rate: RATE });
        break;
      }
      case "speech_to_speech.output_audio.visemes":
        this.emit("visemes", { responseId: ev.response_id, visemes: ev.visemes || [] });
        break;
      case "response.output_audio_transcript.delta":
        this.emit("assistant-text", { responseId: ev.response_id, delta: ev.delta || "" });
        break;
      case "response.done": {
        const id = ev.response?.id || this.activeResponse;
        this.activeResponse = "";
        this.awaitingCreate = false;
        this.emit("response-done", { responseId: id, status: ev.response?.status });
        this.pump();
        break;
      }
      case "error": {
        const err = ev.error || {};
        console.warn("[voice] server error", err);
        if (err.type === "conversation_already_has_active_response" || err.code === "conversation_already_has_active_response") {
          this.awaitingCreate = false;
          // Retry the line once the response in progress finishes.
          if (this.inflight) this.pending.unshift(this.inflight);
          this.inflight = null;
          break;
        }
        this.awaitingCreate = false;
        this.emit("error", { message: err.message || "Voice server error", code: err.type || err.code });
        this.pump();
        break;
      }
    }
  }
}
