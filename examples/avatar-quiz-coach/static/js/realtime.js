// @ts-check
/**
 * Minimal OpenAI-Realtime WebSocket client for the speech-to-speech backend.
 *
 * The backend's LLM is the coach in a live conversation: server VAD commits
 * each turn and the backend answers on its own (`create_response: true`), and
 * talking over the coach cancels its reply (`interrupt_response: true`). The
 * coach never knows the answers; it calls the app's grader tools, which the
 * app runs and answers with `sendToolOutput`.
 *
 * Events: status, queue, speech-started, speech-stopped, user-partial,
 * user-snapshot, user-final, response-started, assistant-item, audio,
 * visemes, assistant-text, tool-call, response-done, mic-level, error.
 */

const RATE = 24000;

// Until a quiz starts, the coach only makes small talk.
const IDLE_INSTRUCTIONS =
  "You are a friendly quiz coach on a voice call. The quiz is being prepared; if the student talks, " +
  "reply in one short sentence and say the questions are coming up.";

// Short fixed lines (fillers) are spoken out of band, outside the
// conversation; few-shot examples keep the model to the exact words.
const RELAY_INSTRUCTIONS =
  "Each user message holds one line between <line> tags. Your whole reply is that line, read out word for word, " +
  "with nothing before or after it.";
const RELAY_EXAMPLES = ["Got it.", "Alright, one moment."].flatMap((line) => [
  { type: "message", role: "user", content: [{ type: "input_text", text: `<line>${line}</line>` }] },
  { type: "message", role: "assistant", content: [{ type: "output_text", text: line }] },
]);

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

const sleep = (/** @type {number} */ ms) => new Promise((r) => setTimeout(r, ms));

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
    this.awaitingCreate = false;
    /** A response.create held back until the response in progress ends. @type {{ instructions?: string, extra?: any, at: number } | null} */
    this.wanted = null;
    /** Out-of-band lines being recorded, by response id. @type {Map<string, { key: string, chunks: Int16Array[] }>} */
    this.captures = new Map();
    /** @type {Map<string, (pcm: Int16Array | null) => void>} */
    this.captureWaiters = new Map();
    this.lastSpeechStart = 0;
    /** What the response.create in flight was for, to retry it right if the server refuses it. @type {{ instructions?: string, extra?: any, capture?: string } | null} */
    this.pendingCreate = null;
    this.speaking = false;
    /** Assistant message item of each response (for truncation). @type {Map<string, string>} */
    this.items = new Map();
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

  get busy() {
    return !!(this.activeResponse || this.awaitingCreate);
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
      if (resp.status === 429) throw Object.assign(new Error("Too many voice sessions just now. Try again in a minute."), { code: "busy" });
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
        instructions: IDLE_INSTRUCTIONS,
        // Speculative transcripts: the backend shares each pause's transcript
        // before it commits the turn, so grading can start during the grace.
        extensions: ["speech_to_speech.input_audio_transcription.snapshot"],
        audio: {
          input: {
            format: { type: "audio/pcm", rate: RATE },
            turn_detection: { type: "server_vad", create_response: true, interrupt_response: true },
          },
          output,
        },
      },
    });
  }

  /**
   * Coach instructions and tools for the quiz (deep-merged into the session).
   * @param {{ instructions?: string, tools?: any[] }} cfg
   */
  configure(cfg) {
    /** @type {Record<string, any>} */
    const session = { type: "realtime" };
    if (cfg.instructions) session.instructions = cfg.instructions;
    if (cfg.tools) Object.assign(session, { tools: cfg.tools, tool_choice: cfg.tools.length ? "auto" : "none" });
    this.send({ type: "session.update", session });
  }

  /** Back to small talk between quizzes. */
  reset() {
    this.configure({ instructions: IDLE_INSTRUCTIONS, tools: [] });
  }

  /** @param {any} msg */
  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  close() {
    this.closed = true;
    this.wanted = null;
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
        this.emit("mic-level", { rms: this.muted ? 0 : rms, raw: rms });
      }
    };
    this.ctx.createMediaStreamSource(stream).connect(node);
    this.micNode = node;
    node.port.postMessage({ kind: "mute", value: this.muted });
  }

  /** Half-duplex gate (when barge-in is off): send silence while the coach talks. */
  setMuted(value) {
    this.muted = !!value;
    this.micNode?.port.postMessage({ kind: "mute", value: this.muted });
  }

  // ── Conversation ────────────────────────────────────────────────────────

  /**
   * Ask the coach to speak now, or as soon as the reply in progress ends.
   * @param {string} [instructions] per-response instructions
   * @param {Record<string, any>} [extra] more response fields (e.g. tool_choice)
   */
  respond(instructions, extra) {
    if (this.busy) {
      this.wanted = { instructions, extra, at: performance.now() };
      return;
    }
    this.awaitingCreate = true;
    this.pendingCreate = { instructions, extra };
    this.send({ type: "response.create", response: { ...(instructions ? { instructions } : {}), ...extra } });
  }

  /**
   * Record a short fixed line in the coach's voice without playing it or
   * adding it to the conversation (only when nothing else is going on).
   * @param {string} line @returns {Promise<Int16Array | null>}
   */
  synthesize(line) {
    if (this.busy || this.captureWaiters.size) return Promise.resolve(null);
    const key = `l${Math.random().toString(36).slice(2, 8)}`;
    this.awaitingCreate = true;
    this.pendingCreate = { capture: key };
    this.send({
      type: "response.create",
      response: {
        conversation: "none",
        output_modalities: ["audio"],
        instructions: RELAY_INSTRUCTIONS,
        metadata: { viva_capture: key },
        input: [...RELAY_EXAMPLES, { type: "message", role: "user", content: [{ type: "input_text", text: `<line>${line}</line>` }] }],
      },
    });
    return new Promise((resolve) => {
      this.captureWaiters.set(key, resolve);
      setTimeout(() => {
        if (this.captureWaiters.delete(key)) resolve(null);
      }, 15000);
    });
  }

  /** Remove a user item (e.g. the coach's own voice picked up by the mic). @param {string} itemId */
  deleteItemId(itemId) {
    if (itemId) this.send({ type: "conversation.item.delete", item_id: itemId });
  }

  /** Remove a reply from the conversation history. @param {string} responseId */
  deleteItem(responseId) {
    const itemId = this.items.get(responseId);
    if (itemId) this.send({ type: "conversation.item.delete", item_id: itemId });
  }

  /** A typed message from the student. @param {string} text */
  sendText(text) {
    this.send({
      type: "conversation.item.create",
      item: { type: "message", role: "user", content: [{ type: "input_text", text }] },
    });
    this.respond();
  }

  /** @param {string} callId @param {string} output @param {boolean} [reply] ask the coach to speak about it */
  sendToolOutput(callId, output, reply = true) {
    this.send({ type: "conversation.item.create", item: { type: "function_call_output", call_id: callId, output } });
    if (reply) this.respond();
  }

  /** Cancel the reply being generated (the server does this itself on barge-in). */
  cancel() {
    this.wanted = null;
    if (this.busy) this.send({ type: "response.cancel" });
  }

  /**
   * Cut a reply's history to what the student actually heard.
   * @param {string} responseId @param {number} audioEndMs
   */
  truncate(responseId, audioEndMs) {
    const itemId = this.items.get(responseId);
    if (itemId) this.send({ type: "conversation.item.truncate", item_id: itemId, content_index: 0, audio_end_ms: Math.max(0, Math.round(audioEndMs)) });
  }

  /** @param {any} ev */
  onEvent(ev) {
    switch (ev.type) {
      case "input_audio_buffer.speech_started":
        this.lastSpeechStart = performance.now();
        this.speaking = true;
        this.emit("speech-started", {});
        break;
      case "input_audio_buffer.speech_stopped":
        this.speaking = false;
        this.emit("speech-stopped", {});
        break;
      case "conversation.item.input_audio_transcription.delta":
        this.emit("user-partial", { itemId: ev.item_id, delta: ev.delta || "" });
        break;
      case "speech_to_speech.input_audio_transcription.snapshot":
        this.emit("user-snapshot", { itemId: ev.item_id, transcript: (ev.transcript || "").trim() });
        break;
      case "conversation.item.input_audio_transcription.completed":
        this.emit("user-final", { itemId: ev.item_id, transcript: (ev.transcript || "").trim() });
        break;
      case "response.created": {
        this.awaitingCreate = false;
        this.pendingCreate = null;
        this.activeResponse = ev.response?.id || "resp";
        const key = ev.response?.metadata?.viva_capture;
        if (key) {
          this.captures.set(this.activeResponse, { key, chunks: [] });
          break;
        }
        this.emit("response-started", { responseId: this.activeResponse });
        break;
      }
      case "response.output_item.added":
        if (ev.item?.type === "message" && ev.item.id) {
          this.items.set(ev.response_id || this.activeResponse, ev.item.id);
          this.emit("assistant-item", { responseId: ev.response_id || this.activeResponse, itemId: ev.item.id });
        }
        break;
      case "response.output_audio.delta": {
        const bytes = fromBase64(ev.delta);
        const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
        const capture = this.captures.get(ev.response_id || this.activeResponse);
        if (capture) capture.chunks.push(pcm.slice());
        else this.emit("audio", { responseId: ev.response_id || this.activeResponse, pcm, rate: RATE });
        break;
      }
      case "speech_to_speech.output_audio.visemes":
        if (this.captures.has(ev.response_id)) break;
        this.emit("visemes", { responseId: ev.response_id, visemes: ev.visemes || [] });
        break;
      case "response.output_audio_transcript.delta":
      case "response.output_text.delta":
        if (this.captures.has(ev.response_id)) break;
        this.emit("assistant-text", { responseId: ev.response_id || this.activeResponse, delta: ev.delta || "" });
        break;
      case "response.function_call_arguments.done": {
        let args = {};
        try {
          args = JSON.parse(ev.arguments || "{}");
        } catch {}
        this.emit("tool-call", { responseId: ev.response_id || this.activeResponse, callId: ev.call_id, name: ev.name, args });
        break;
      }
      case "response.done": {
        const id = ev.response?.id || this.activeResponse;
        this.activeResponse = "";
        this.awaitingCreate = false;
        const capture = this.captures.get(id);
        if (capture) {
          this.captures.delete(id);
          const done = this.captureWaiters.get(/** @type {any} */ (capture).key);
          this.captureWaiters.delete(/** @type {any} */ (capture).key);
          const n = capture.chunks.reduce((a, c) => a + c.length, 0);
          const pcm = new Int16Array(n);
          let o = 0;
          for (const c of capture.chunks) {
            pcm.set(c, o);
            o += c.length;
          }
          done?.(ev.response?.status === "completed" && n ? pcm : null);
        } else {
          this.emit("response-done", { responseId: id, status: ev.response?.status });
        }
        // A held-back response.create goes out now, unless the student has
        // started talking since: the backend answers their turn by itself.
        const wanted = this.wanted;
        this.wanted = null;
        if (wanted && this.lastSpeechStart < wanted.at) this.respond(wanted.instructions, wanted.extra);
        break;
      }
      case "error": {
        const err = ev.error || {};
        const code = err.code || err.type;
        console.warn("[voice] server error", err);
        if (code === "conversation_already_has_active_response") {
          // Our response.create raced one the server started (the student
          // spoke). A recording is simply skipped (re-sent as a plain reply,
          // with no new input, the model would carry on talking, even as the
          // student); a reply we asked for (a tool result to announce) goes
          // out after it, unless the student's new turn covers it.
          const failed = this.pendingCreate;
          this.awaitingCreate = false;
          this.pendingCreate = null;
          if (failed?.capture) {
            const done = this.captureWaiters.get(failed.capture);
            this.captureWaiters.delete(failed.capture);
            done?.(null);
          } else if (failed) {
            this.wanted = this.wanted || { instructions: failed.instructions, extra: failed.extra, at: performance.now() };
          }
          break;
        }
        this.awaitingCreate = false;
        this.emit("error", { message: err.message || "Voice server error", code });
        break;
      }
    }
  }
}
