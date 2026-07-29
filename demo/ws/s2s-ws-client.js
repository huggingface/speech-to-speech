// @ts-check
/**
 * Minimal WebSocket client for the Hugging Face speech-to-speech load balancer.
 *
 * Two-step handshake (same /session route as the WebRTC client):
 *
 *   1. POST `<lb_url>/session` -> JSON `{ connect_url: wss://<compute>/v1/realtime?session_token=<JWT>, ... }`
 *   2. Open a WebSocket directly on `connect_url` (no rewrite, unlike the WebRTC client).
 *
 * Once the socket is open we follow the OpenAI Realtime GA WebSocket
 * protocol:
 *
 *   - Server pushes `session.created` immediately after upgrade.
 *   - We send `session.update` (GA schema: `session.audio.{input,output}`,
 *     `session.output_modalities`, ...).
 *   - We stream mic audio as PCM16 16 kHz mono base64 chunks via
 *     `input_audio_buffer.append`.
 *   - The server pushes `response.output_audio.delta` (PCM16 24 kHz mono
 *     base64) and transcript deltas.
 *
 * Audio is handled internally via two AudioWorklet processors so the
 * client owns the full mic-in / speaker-out pipeline. The main app only
 * sees high-level lifecycle events (`status`, `transcript`, `error`,
 * `session`), the same shape as the WebRTC client.
 *
 * @typedef {"idle" | "creating-session" | "queued" | "your-turn" | "connecting" |
 *           "connected" | "user-speaking" | "processing" | "ai-speaking" |
 *           "closed" | "error"
 * } WsStatus
 *
 * @typedef {Object} WsSessionInfo
 * @property {string} sessionId
 * @property {string} connectUrl
 * @property {string} websocketUrl
 * @property {string} sessionToken
 * @property {number} pendingTimeoutS
 * @property {string} [tier] Login tier from the session proxy ("anon"|"free"|"pro").
 * @property {boolean} [limited] Whether this session is metered (heartbeat needed).
 * @property {number} [heartbeatSec] Suggested heartbeat cadence in seconds.
 * @property {number} [remainingSec] Daily budget left after this grant (display).
 *
 * @typedef {Object} WsClientOptions
 * @property {string} [sessionUrl] URL to POST for the session handshake (returns
 *   `{ connect_url, ... }`). Usually a same-origin proxy like `api/session` so the
 *   load-balancer address stays server-side. Provide this OR `directUrl`.
 * @property {string} [loadBalancerUrl] Load-balancer base URL. Legacy/direct
 *   alternative to `sessionUrl`: the client POSTs `<lb>/session` itself. Prefer
 *   `sessionUrl` so the LB address isn't exposed to the browser.
 * @property {string} [directUrl] Full WebSocket URL of an s2s realtime endpoint
 *   (e.g. `ws://localhost:8080/v1/realtime`). When set, the client skips the
 *   session POST and dials it directly — no load balancer in between.
 * @property {string} voice
 * @property {string} instructions
 * @property {MediaStream} [micStream] Live mic stream. Provide this OR `acquireMic`.
 * @property {() => Promise<MediaStream>} [acquireMic] Lazily obtain the mic stream,
 *   called only once a session is actually granted (after any queue wait). Lets the
 *   caller prime mic permission up front but not hold the mic 'in use' indicator on
 *   while waiting in line. Ignored if `micStream` is already set.
 * @property {AudioContext} [audioContext] Pre-created (and resumed) context.
 *   iOS Safari only lets an AudioContext start from within a user gesture, so
 *   the caller creates/resumes it synchronously on the orb tap and hands it
 *   here; otherwise it stays suspended (silent) after the mic/session awaits.
 * @property {ToolDef[]} [tools] Function tools declared to the backend in the
 *   initial `session.update`. The model decides when to call them; the caller
 *   executes and replies via `sendToolOutput` + `requestResponse`.
 * @property {NoiseGate} [noiseGate] Client-side noise gate applied to the mic
 *   before it's sent. Tunable live via `setNoiseGate`.
 *
 * @typedef {Object} NoiseGate
 * @property {boolean} enabled
 * @property {number} thresholdDb Open threshold in dBFS (e.g. -45).
 *
 * @typedef {Object} ToolDef
 * @property {"function"} type
 * @property {string} name
 * @property {string} description
 * @property {object} parameters JSON Schema for the call arguments.
 *
 * @typedef {Object} TranscriptEvent
 * @property {"user" | "assistant"} role
 * @property {string} text
 * @property {boolean} partial
 */

import {
  base64FromArrayBuffer,
  base64ToBytes,
  extractResponseTranscript,
  trimTrailingSlash,
} from "./codec.js";
import { OrbVisualiser, VIS_FFT_SIZE } from "./orb-visualizer.js";

/** Build an Error carrying a `code` (and optional extra fields) so callers can
 *  branch on the failure kind: "limit" | "queue-full" | "queue-expired" | "aborted".
 *  @param {string} message @param {string} code @param {object} [extra] */
function _codedError(message, code, extra) {
  const err = /** @type {Error & { code?: string }} */ (new Error(message));
  err.code = code;
  if (extra) Object.assign(err, extra);
  return err;
}

// The s2s pipeline runs internally at 16 kHz mono PCM. The WebRTC transport
// resamples to 48 kHz for Opus, but the WebSocket transport emits the
// native pipeline rate. We don't (can't) override it via `audio.output.format`
// because the server's pydantic validator rejects the whole `session.update`
// as soon as a sub-field shape it doesn't know about appears.
const OUTPUT_SAMPLE_RATE = 16000;
const MIC_CHUNK_MS = 40;

export class S2sWsRealtimeClient extends EventTarget {
  /** @param {WsClientOptions} options */
  constructor(options) {
    super();
    /** @type {WsClientOptions} */
    this.options = options;
    /** @type {ToolDef[]} Function tools declared to the backend. */
    this._tools = options.tools ?? [];
    /** @type {string} Direct realtime WS URL (set => skip the LB session POST). */
    this._directUrl = options.directUrl ?? "";
    /** @type {string} Where to POST for the session handshake. Prefer the
     *  explicit `sessionUrl`; fall back to `<loadBalancerUrl>/session` for callers
     *  that still pass the LB address directly. */
    this._sessionUrl = options.sessionUrl
      ? options.sessionUrl
      : options.loadBalancerUrl
        ? `${trimTrailingSlash(options.loadBalancerUrl)}/session`
        : "";
    /** @type {(() => Promise<MediaStream>) | null} Lazy mic acquisition (post-grant). */
    this._acquireMic = options.acquireMic ?? null;
    /** @type {boolean} Set by close() to abort a queue wait in progress. */
    this._closed = false;
    /** @type {string} The active queue ticket id while waiting (else ""). */
    this._queueId = "";
    /** @type {(() => void) | null} Wakes the queue poll sleep early on close(). */
    this._queueWake = null;
    /** @type {ReturnType<typeof setTimeout> | 0} */
    this._queueTimer = 0;
    // Join gate: after waiting in line the caller must explicitly `join()` before
    // we dial, so a slot isn't spent on someone who walked away. Resolved by
    // join(), rejected on timeout (the LB reclaims the slot) or close().
    /** @type {(() => void) | null} */
    this._joinResolve = null;
    /** @type {((err: Error) => void) | null} */
    this._joinReject = null;
    /** @type {ReturnType<typeof setTimeout> | 0} */
    this._joinTimer = 0;
    /** @type {NoiseGate} Mic noise gate; off by default. */
    this._noiseGate = options.noiseGate ?? { enabled: false, thresholdDb: -45 };
    /** @type {WebSocket | null} */
    this._ws = null;
    /** @type {AudioContext | null} */
    this._ctx = null;
    /** @type {MediaStreamAudioSourceNode | null} */
    this._micSrc = null;
    /** @type {AudioWorkletNode | null} */
    this._captureNode = null;
    /** @type {AudioWorkletNode | null} */
    this._playbackNode = null;
    /** @type {GainNode | null} */
    this._captureSink = null;
    /** @type {AnalyserNode | null} */
    this._micAnalyser = null;
    /** @type {AnalyserNode | null} */
    this._outAnalyser = null;
    /** @type {OrbVisualiser | null} */
    this._visualiser = null;
    /** @type {WsStatus} */
    this._status = "idle";
    this._aiSpeaking = false;
    /** @type {Set<string>} response_ids that have actually played audio, so the
     * UI can tell a barge-in cut (keep it) from a never-heard speculative
     * response (drop it). */
    this._audibleResponses = new Set();
    /** @type {Map<string, string>} The CURRENT assistant transcript segment per
     * response, accumulated from streamed deltas (reset on each segment's done). */
    this._asstTranscriptByResp = new Map();
    /** @type {Map<string, string>} Completed assistant transcript segments per
     * response, space-joined. A single response can emit several
     * `*.transcript.done` events; we concatenate them until response.done. */
    this._asstFullByResp = new Map();
    this._muted = false;
    // ── Response lock ────────────────────────────────────────────────────
    // The backend allows only ONE response in flight: creating a second while
    // one is active fails with `conversation_already_has_active_response`. So
    // we serialize response.create. `_openResponses` counts responses the
    // server has confirmed (response.created) but not yet finished
    // (response.done) — it's cumulative, so every create maps to one done.
    // `_createInFlight` covers the window after we send a create but before its
    // response.created echo. Any requestResponse() made while locked is queued
    // and replayed, one at a time, as each response.done frees the slot.
    this._openResponses = 0;
    this._createInFlight = false;
    /** @type {{ image?: string }[]} Pending response.create payloads, one per
     * queued requestResponse(). A payload may carry an image to send just
     * before its create (so the frame travels with the create, not eagerly). */
    this._createQueue = [];
    /** @type {Promise<void> | null} */
    this._readyPromise = null;
    this._sessionConfigured = false;
    this._debug = (() => { try { return localStorage.getItem("s2s.debug") === "1"; } catch { return false; } })();
  }

  get status() {
    return this._status;
  }

  /** @param {WsStatus} status */
  _setStatus(status) {
    if (this._status === status) return;
    this._status = status;
    this.dispatchEvent(new CustomEvent("status", { detail: { status } }));
  }

  /** Full assistant transcript so far for a response: the completed segments
   *  plus the in-progress one, all space-joined.
   *  @param {string} rid @returns {string} */
  _asstDisplay(rid) {
    const full = this._asstFullByResp.get(rid) || "";
    const seg = this._asstTranscriptByResp.get(rid) || "";
    if (!seg) return full;
    return full ? `${full} ${seg}` : seg;
  }

  _markAudible() {
    if (this._status === "ai-speaking") return;
    if (this._status === "closed" || this._status === "error") return;
    this._setStatus("ai-speaking");
  }

  /**
   * Full handshake. Resolves once the WS is open AND the audio pipeline is
   * ready to send/receive samples.
   * @returns {Promise<void>}
   */
  async connect() {
    if (this._ws) throw new Error("Already connected");

    let connectUrl;
    if (this._directUrl) {
      // Direct mode: no load balancer, no /session POST — dial the realtime
      // endpoint straight away (e.g. a local s2s server).
      connectUrl = this._directUrl;
      this._setStatus("connecting");
    } else {
      if (!this._sessionUrl) {
        throw new Error("No session endpoint or direct URL configured");
      }
      this._setStatus("creating-session");
      const { grant, waited } = await this._createSessionOrQueue();
      if (this._closed) throw _codedError("connect aborted", "aborted");
      // If we waited in line, don't dial until the user explicitly joins — this
      // keeps a freed slot from being spent on someone who stepped away, and the
      // click is a fresh gesture (re-arms the AudioContext on iOS).
      if (waited) {
        await this._awaitJoin(grant);
        if (this._closed) throw _codedError("connect aborted", "aborted");
      }
      this.dispatchEvent(new CustomEvent("session", { detail: { info: grant } }));
      connectUrl = grant.connectUrl;
      this._setStatus("connecting");
    }

    // Acquire the mic now — only once a slot is actually ours. The caller primed
    // permission up front, so this is silent and the 'in use' indicator lights
    // only for a real, connecting session (never during a queue wait).
    if (!this.options.micStream && this._acquireMic) {
      this.options.micStream = await this._acquireMic();
    }

    // Spin up the AudioContext + worklets in parallel with the WS dial.
    const audioReady = this._setupAudio();
    const wsReady = this._openWebSocket(connectUrl);
    await Promise.all([audioReady, wsReady]);
  }

  /**
   * POST the session handshake; if the pool is busy, wait in the queue (polling
   * position) until a slot is claimed. Resolves to a grant plus whether we had to
   * wait (which decides if an explicit join is required before dialing).
   * @returns {Promise<{ grant: WsSessionInfo, waited: boolean }>}
   */
  async _createSessionOrQueue() {
    const first = await this._postSession();
    if (first.state === "queued") {
      this._setStatus("queued");
      const grant = await this._pollQueue(first);
      return { grant, waited: true };
    }
    return { grant: first.grant, waited: false };
  }

  /**
   * Hold at the front of the line until the user clicks join (resolves the gate)
   * or the grant lapses. Announces "your-turn" + a deadline the UI counts down.
   * @param {WsSessionInfo} grant
   * @returns {Promise<void>}
   */
  _awaitJoin(grant) {
    // The LB reclaims an unclaimed slot at its pending timeout; expire the gate a
    // touch earlier so we never dial a session the LB just reaped.
    const windowS = Math.max(3, (grant.pendingTimeoutS || 60) - 3);
    this._setStatus("your-turn");
    this.dispatchEvent(
      new CustomEvent("ready-to-join", { detail: { info: grant, expiresSec: windowS } }),
    );
    return new Promise((resolve, reject) => {
      this._joinResolve = resolve;
      this._joinReject = reject;
      this._joinTimer = setTimeout(() => {
        this._joinResolve = null;
        this._joinReject = null;
        reject(_codedError("Your spot expired", "join-expired"));
      }, windowS * 1000);
    });
  }

  /** Accept the held slot and let connect() proceed to dial. Called from the
   *  "Join now" click, so it's a user gesture: re-resume the AudioContext, which
   *  iOS may have suspended while we waited. */
  join() {
    if (this._joinTimer) {
      clearTimeout(this._joinTimer);
      this._joinTimer = 0;
    }
    try {
      void this.options.audioContext?.resume();
    } catch {
      // best-effort; _setupAudio resumes again
    }
    const resolve = this._joinResolve;
    this._joinResolve = null;
    this._joinReject = null;
    resolve?.();
  }

  /**
   * POST /session once. Returns either a granted session or a queue ticket.
   * @returns {Promise<{ state: "granted", grant: WsSessionInfo } | { state: "queued", queueId: string, position: number, pollIntervalS: number }>}
   */
  async _postSession() {
    const url = this._sessionUrl;
    console.log("[ws] POST", url);
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (response.status === 402) {
      // The session proxy refused: today's per-tier time budget is spent. Surface
      // it as a typed error so the UI shows the limit modal, not a crash.
      const body = await response.json().catch(() => ({}));
      throw _codedError("Daily conversation limit reached", "limit", { tier: body?.tier });
    }
    if (response.status === 503) {
      const body = await response.json().catch(() => ({}));
      if (body?.state === "at_capacity") {
        throw _codedError("The queue is full — try again shortly.", "queue-full");
      }
      throw new Error("/session failed (503)");
    }
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`/session failed (${response.status}): ${text}`);
    }
    const json = await response.json();
    if (json.state === "queued") {
      return {
        state: "queued",
        queueId: json.queue_id,
        position: json.position,
        pollIntervalS: json.poll_interval_s,
      };
    }
    return { state: "granted", grant: this._parseGrant(json) };
  }

  /**
   * Poll the waiting queue until this ticket claims a slot. Emits `queue` events
   * ({ position }) as the line advances. Throws on limit (402), expiry (404), or
   * close(). Transient network/5xx blips are ignored and retried next tick.
   * @param {{ queueId: string, position: number, pollIntervalS: number }} ticket
   * @returns {Promise<WsSessionInfo>}
   */
  async _pollQueue(ticket) {
    const intervalMs = Math.max(1, ticket.pollIntervalS || 2) * 1000;
    this._queueId = ticket.queueId;
    this._emitQueue(ticket.position);

    while (true) {
      await this._queueSleep(intervalMs);
      if (this._closed) throw _codedError("queue wait aborted", "aborted");

      let response;
      try {
        response = await fetch(`api/queue/${encodeURIComponent(this._queueId)}`, {
          headers: { "Content-Type": "application/json" },
        });
      } catch {
        continue; // network blip — keep our place, retry next tick
      }

      if (response.status === 402) {
        const body = await response.json().catch(() => ({}));
        throw _codedError("Daily conversation limit reached", "limit", { tier: body?.tier });
      }
      if (response.status === 404) {
        throw _codedError("Queue timed out", "queue-expired");
      }
      if (!response.ok) continue; // 502/503 — transient, retry

      const json = await response.json().catch(() => null);
      if (!json) continue;
      if (json.state === "queued") {
        this._emitQueue(json.position);
        continue;
      }
      // Reached the front and claimed a slot.
      this._queueId = "";
      return this._parseGrant(json);
    }
  }

  /** @param {number} position */
  _emitQueue(position) {
    this.dispatchEvent(
      new CustomEvent("queue", { detail: { position, queueId: this._queueId } }),
    );
  }

  /** A sleep that close() can cut short so a queued client tears down promptly.
   *  @param {number} ms */
  _queueSleep(ms) {
    return new Promise((resolve) => {
      this._queueWake = resolve;
      this._queueTimer = setTimeout(() => {
        this._queueWake = null;
        resolve();
      }, ms);
    });
  }

  /** @param {any} json @returns {WsSessionInfo} */
  _parseGrant(json) {
    return {
      sessionId: json.session_id,
      connectUrl: json.connect_url,
      websocketUrl: json.websocket_url,
      sessionToken: json.session_token,
      pendingTimeoutS: json.pending_timeout_s,
      tier: json.tier,
      limited: json.limited,
      heartbeatSec: json.heartbeatSec,
      remainingSec: json.remainingSec,
    };
  }

  async _setupAudio() {
    // Prefer a context the caller already created + resumed inside the tap
    // gesture (required on iOS). Fall back to creating one here for callers
    // that don't (desktop is lenient about the gesture timing).
    // Most desktops give us 48 kHz, mobiles can give 44.1/24/16 kHz; the
    // capture worklet handles any rate (linear interp fallback).
    const ctx = this.options.audioContext ?? new AudioContext({ latencyHint: "interactive" });
    this._ctx = ctx;

    // Resume if still suspended. This is best-effort here — on iOS the resume
    // that actually counts is the one the caller did synchronously on tap.
    if (ctx.state === "suspended") {
      try {
        await ctx.resume();
      } catch (err) {
        console.warn("[ws] AudioContext resume failed:", err);
      }
    }

    // The worklets live at the repo root, one level up from this module.
    const base = new URL("../worklets/", import.meta.url);
    await ctx.audioWorklet.addModule(new URL("mic-capture.js", base).href);
    await ctx.audioWorklet.addModule(new URL("audio-playback.js", base).href);

    const captureNode = new AudioWorkletNode(ctx, "mic-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { chunkMs: MIC_CHUNK_MS },
    });
    captureNode.port.onmessage = (e) => {
      const data = e.data;
      if (data instanceof ArrayBuffer) {
        this._onMicChunk(data);
      } else if (data?.kind === "level") {
        // Raw pre-gate mic RMS for the Settings meter.
        this.dispatchEvent(new CustomEvent("input-level", { detail: { rms: data.rms } }));
      }
    };
    // Push the initial gate config now that the worklet exists.
    captureNode.port.postMessage({ kind: "gate", ...this._noiseGate });
    this._captureNode = captureNode;

    const micSrc = ctx.createMediaStreamSource(this.options.micStream);
    micSrc.connect(captureNode);
    this._micSrc = micSrc;

    // Mic analyser: tap the mic in parallel with the worklet so we get the
    // raw (un-resampled, un-clipped) signal for the visualiser.
    const micAnalyser = ctx.createAnalyser();
    micAnalyser.fftSize = VIS_FFT_SIZE;
    micAnalyser.smoothingTimeConstant = 0;
    micSrc.connect(micAnalyser);
    this._micAnalyser = micAnalyser;

    const playbackNode = new AudioWorkletNode(ctx, "audio-playback", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    playbackNode.port.postMessage({ kind: "config", inputRate: OUTPUT_SAMPLE_RATE });
    playbackNode.port.onmessage = (e) => this._onPlaybackMessage(e.data);

    // Output analyser sits between the playback worklet and the speakers.
    const outAnalyser = ctx.createAnalyser();
    outAnalyser.fftSize = VIS_FFT_SIZE;
    outAnalyser.smoothingTimeConstant = 0.3;
    playbackNode.connect(outAnalyser);
    outAnalyser.connect(ctx.destination);
    this._outAnalyser = outAnalyser;
    this._playbackNode = playbackNode;

    this._visualiser = new OrbVisualiser(micAnalyser, outAnalyser, () => this._aiSpeaking);
    this._visualiser.start();
  }

  /** @param {string} connectUrl */
  _openWebSocket(connectUrl) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(connectUrl);
      ws.binaryType = "arraybuffer";
      this._ws = ws;

      const onceOpen = () => {
        ws.removeEventListener("open", onceOpen);
        ws.removeEventListener("error", onceErr);
        resolve();
      };
      const onceErr = (e) => {
        ws.removeEventListener("open", onceOpen);
        ws.removeEventListener("error", onceErr);
        reject(new Error(`WebSocket failed to open: ${e?.type ?? "error"}`));
      };
      ws.addEventListener("open", onceOpen);
      ws.addEventListener("error", onceErr);

      ws.addEventListener("message", (e) => this._onWsMessage(e.data));
      ws.addEventListener("close", (e) => this._onWsClose(e));
      ws.addEventListener("error", (e) => {
        console.error("[ws] socket error", e);
      });
    });
  }

  /**
   * @param {{ kind: string; queuedMs?: number; played?: number }} data
   */
  _onPlaybackMessage(data) {
    if (data?.kind === "underrun") {
      // Server stopped sending audio mid-response. Most likely the turn
      // ended cleanly (a response.done usually arrives just before/after
      // this). We let the state machine fall back to "connected" via the
      // response.done event handler.
    }
  }

  /**
   * Mic worklet just sent us a ~40 ms PCM16 16 kHz mono chunk.
   * Base64-encode and forward via the WS.
   * @param {ArrayBuffer} pcm16Buffer
   */
  _onMicChunk(pcm16Buffer) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    if (!this._sessionConfigured) return; // Server rejects audio before session.update.
    if (this._muted) return;
    const b64 = base64FromArrayBuffer(pcm16Buffer);
    this._send({ type: "input_audio_buffer.append", audio: b64 });
  }

  /**
   * @param {string | ArrayBuffer | Blob} raw
   */
  async _onWsMessage(raw) {
    let text;
    if (typeof raw === "string") {
      text = raw;
    } else if (raw instanceof ArrayBuffer) {
      text = new TextDecoder("utf-8").decode(raw);
    } else if (raw instanceof Blob) {
      text = await raw.text();
    } else {
      return;
    }

    let event;
    try {
      event = JSON.parse(text);
    } catch {
      return;
    }

    const type = event?.type;
    if (typeof type !== "string") return;
    // Opt-in event tracing for diagnosing turn/transcript issues. Enable with
    // `localStorage.setItem("s2s.debug", "1")` in the browser console.
    if (this._debug) {
      const extra = type.startsWith("conversation.item.input_audio_transcription")
        ? ` item=${event.item_id} ci=${event.content_index} ${event.delta ?? event.transcript ?? ""}`
        : type.startsWith("response.")
          ? ` resp=${event.response_id ?? event.response?.id ?? ""} status=${event.response?.status ?? ""} ${event.transcript ?? ""}`
          : "";
      console.debug(`[ws] ${type}${extra}`);
    }

    switch (type) {
      case "session.created":
        // Server-side defaults for the s2s pipeline are already what we
        // want (server_vad, whisper-1 transcription, PCM16 16k in / 24k
        // out). We only push the user-tunable bits: voice + instructions.
        this._sendSessionUpdate();
        this._sessionConfigured = true;
        if (this._status === "connecting") this._setStatus("connected");
        break;

      case "session.updated":
        // Acknowledged by server, nothing to do.
        break;

      case "input_audio_buffer.speech_started":
        // User started speaking — stop any audio still playing OR queued, every
        // time. We clear unconditionally (not just when `_aiSpeaking`): after a
        // reply or a tool result the worklet's ring buffer can still be draining
        // even though we already flipped `_aiSpeaking` off, and that tail would
        // otherwise keep playing over the user's barge-in.
        this._playbackNode?.port.postMessage({ kind: "clear" });
        this._aiSpeaking = false;
        this._setStatus("user-speaking");
        break;

      case "input_audio_buffer.speech_stopped":
        if (this._status === "user-speaking") this._setStatus("processing");
        break;

      case "response.created":
        // A response now owns the slot — count it and clear our create guard
        // (this confirms either our create or a server-initiated one).
        this._openResponses++;
        this._createInFlight = false;
        if (this._status === "connected" || this._status === "user-speaking") {
          this._setStatus("processing");
        }
        break;

      case "response.output_item.added":
        if (this._status === "connected" || this._status === "user-speaking") {
          this._setStatus("processing");
        }
        break;

      case "response.audio.delta":
      case "response.output_audio.delta": {
        this._pushAudioDelta(event.delta);
        const rid = event.response_id ?? event.response?.id;
        if (rid) this._audibleResponses.add(rid);
        if (!this._aiSpeaking) {
          this._aiSpeaking = true;
          this._markAudible();
        }
        break;
      }

      case "response.content_part.added": {
        const part = event.part;
        if (part?.type === "audio" || part?.type === "output_audio") {
          this._markAudible();
        }
        break;
      }

      case "response.done": {
        this._aiSpeaking = false;
        // This response freed the slot (completion OR cancellation both arrive
        // as response.done). Decrement and, if a create was waiting, replay it.
        this._openResponses = Math.max(0, this._openResponses - 1);
        if (this._status === "ai-speaking" || this._status === "processing") {
          this._setStatus("connected");
        }
        // A response closes here for BOTH normal completion and cancellation
        // (the s2s server signals a speculative-turn interrupt as
        // `response.done` with status "cancelled" — there is no separate
        // `response.cancelled` event). Surface the id + status so the UI can
        // drop a cancelled response's transcript and commit a completed one.
        const status = event.response?.status ?? "completed";
        const responseId = event.response?.id ?? "";
        // Did this response ever play audio? Distinguishes a barge-in cut (the
        // user heard part of it) from a speculative response that never played.
        const audible = responseId ? this._audibleResponses.has(responseId) : false;
        this._audibleResponses.delete(responseId);
        // Pull whatever transcript the response carries, falling back to the
        // segments we concatenated from the `*.transcript.done` events (plus any
        // in-progress delta). For an interrupted reply the response payload may
        // be empty, so this is the last chance to capture the text.
        const transcript =
          extractResponseTranscript(event.response) ||
          this._asstDisplay(responseId) ||
          "";
        // Response finished — clear both transcript accumulators for it.
        this._asstTranscriptByResp.delete(responseId);
        this._asstFullByResp.delete(responseId);
        this.dispatchEvent(new CustomEvent("response-finished", {
          detail: { responseId, status, audible, transcript },
        }));
        // The slot is free now — replay a queued create (e.g. a tool follow-up
        // that arrived while this response was still running).
        this._flushQueuedCreate();
        break;
      }

      case "response.function_call_arguments.done": {
        const name = typeof event.name === "string" ? event.name : "";
        const args = typeof event.arguments === "string" ? event.arguments : "{}";
        const callId = typeof event.call_id === "string" ? event.call_id : "";
        if (name) {
          this.dispatchEvent(new CustomEvent("toolcall", {
            detail: { name, arguments: args, callId },
          }));
        } else {
          // A nameless call can't be executed, so no function_call_output is
          // ever sent and the model would wait forever for a result. The
          // backend shouldn't emit these; warn loudly rather than stall silently.
          console.warn(`[ws] function_call_arguments.done with no name (call_id=${callId}); cannot run tool — turn may stall`);
        }
        break;
      }

      case "conversation.item.input_audio_transcription.delta": {
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          // `itemId` is REUSED across a speculative continuation, so the UI
          // groups both segments into one message. The delta carries the full
          // cumulative transcript so far (not an increment).
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: {
                role: "user",
                text: delta,
                partial: true,
                itemId: typeof event.item_id === "string" ? event.item_id : "",
              },
            }),
          );
        }
        break;
      }

      case "conversation.item.input_audio_transcription.completed": {
        const transcript = typeof event.transcript === "string" ? event.transcript : "";
        if (transcript) {
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: {
                role: "user",
                text: transcript,
                partial: false,
                itemId: typeof event.item_id === "string" ? event.item_id : "",
              },
            }),
          );
        }
        break;
      }

      case "response.audio_transcript.delta":
      case "response.output_audio_transcript.delta": {
        // Stream the assistant transcript live: accumulate the incremental
        // deltas and push the running text to the UI. Every transcribe event we
        // receive reaches the conversation, so an interrupted reply already has
        // its partial text even if the `.done` never fires.
        this._markAudible();
        const rid = typeof event.response_id === "string" ? event.response_id : "";
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          this._asstTranscriptByResp.set(rid, (this._asstTranscriptByResp.get(rid) || "") + delta);
          // Show completed segments + the segment streaming in right now.
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: { role: "assistant", text: this._asstDisplay(rid), partial: true, responseId: rid },
            }),
          );
        }
        break;
      }

      case "response.audio_transcript.done":
      case "response.output_audio_transcript.done": {
        const rid = typeof event.response_id === "string" ? event.response_id : "";
        // This is ONE completed segment. A response can emit several; concatenate
        // them, space-separated, until response.done clears the accumulator.
        const segment =
          (typeof event.transcript === "string" && event.transcript) ||
          this._asstTranscriptByResp.get(rid) ||
          "";
        this._asstTranscriptByResp.delete(rid); // segment finished; next one starts fresh
        if (segment) {
          const prev = this._asstFullByResp.get(rid) || "";
          this._asstFullByResp.set(rid, prev ? `${prev} ${segment}` : segment);
        }
        const full = this._asstFullByResp.get(rid) || "";
        if (full) {
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: { role: "assistant", text: full, partial: false, responseId: rid },
            }),
          );
        }
        break;
      }

      case "error": {
        const err = event.error;
        console.error("[ws] server error:", err);
        // The "another response is already active" race: our optimistic create
        // collided with a still-running response. Don't surface it — clear the
        // in-flight guard and re-queue, so the create replays on the next
        // response.done (never retried immediately, which would just collide
        // again).
        if (err?.type === "conversation_already_has_active_response" ||
            err?.code === "conversation_already_has_active_response") {
          if (this._createInFlight) {
            this._createInFlight = false;
            // Re-queue a BARE create: any image on the original payload was
            // already sent before this (rejected) create, so don't resend it.
            this._createQueue.push({});
          }
          break;
        }
        // Every other server error is non-fatal: surface it for logging but
        // NEVER tear the socket down. Only transport failures (close / failed
        // open) are fatal, and those come through their own paths.
        this.dispatchEvent(
          new CustomEvent("server-error", { detail: { error: new Error(err?.message ?? "Server error") } }),
        );
        break;
      }
    }
  }

  /** @param {string} b64 */
  _pushAudioDelta(b64) {
    if (!this._playbackNode) return;
    if (!b64) return;
    const bytes = base64ToBytes(b64);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const samples = new Float32Array(bytes.byteLength / 2);
    for (let i = 0; i < samples.length; i++) {
      const s = view.getInt16(i * 2, true);
      samples[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
    }
    this._playbackNode.port.postMessage({ kind: "audio", samples }, [samples.buffer]);
  }

  /** @param {CloseEvent} ev */
  _onWsClose(ev) {
    console.log("[ws] socket closed:", ev.code, ev.reason);
    if (this._status === "closed" || this._status === "error") return;
    if (ev.code === 1000) {
      this._setStatus("closed");
    } else {
      this.dispatchEvent(
        new CustomEvent("error", {
          detail: { error: new Error(`WebSocket closed (${ev.code}) ${ev.reason || ""}`.trim()) },
        }),
      );
      this._setStatus("error");
    }
  }

  _sendSessionUpdate() {
    // Minimal payload: only the bits the user is allowed to configure.
    // The s2s server already defaults to server_vad, whisper-1
    // transcription, 16 kHz PCM input and 24 kHz PCM output, so we don't
    // need (and must not send) `audio.input.format`, `audio.input.transcription`,
    // `audio.input.turn_detection` or `audio.output.format`: the pydantic
    // validator on the server rejects the whole event if any unknown or
    // future-shaped sub-field shows up.
    /** @type {Record<string, any>} */
    const session = {
      type: "realtime",
      instructions: this.options.instructions,
      audio: {
        output: { voice: this.options.voice },
      },
    };
    // Tools are declared here; the backend already accepts them in
    // session.update and emits response.function_call_arguments.done when the
    // model decides to call one. Only include the keys when we actually have
    // tools — the server's pydantic validator is strict about shapes.
    if (this._tools.length) {
      session.tools = this._tools;
      session.tool_choice = "auto";
    }
    this._send({ type: "session.update", session });
  }

  /** Update voice/instructions on a live session without tearing down. */
  /** @param {{ voice?: string; instructions?: string }} patch */
  updateSession(patch) {
    /** @type {Record<string, any>} */
    const session = { type: "realtime" };
    if (patch.instructions) session.instructions = patch.instructions;
    if (patch.voice) session.audio = { output: { voice: patch.voice } };
    if (Object.keys(session).length > 1) {
      this._send({ type: "session.update", session });
    }
  }

  /**
   * Replace the declared tool set on a live session (e.g. the user flipped a
   * tool switch mid-conversation). Always sends `tools` — an empty array
   * clears them — so toggling the last tool off actually removes it.
   * @param {ToolDef[]} tools
   */
  setTools(tools) {
    this._tools = tools;
    this._send({
      type: "session.update",
      session: { type: "realtime", tools, tool_choice: tools.length ? "auto" : "none" },
    });
  }

  /**
   * Return a tool's result to the model. Pairs with the `toolcall` event's
   * `callId`. Caller follows this with `requestResponse()` so the model speaks.
   * @param {string} callId
   * @param {string} output Plain text / JSON string the model will read.
   */
  sendToolOutput(callId, output) {
    if (!callId) return; // Can't target a result without the call id.
    this._send({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: callId, output },
    });
  }

  /**
   * Add an image to the conversation as user content, so the vision-language
   * model can see it (used by the camera tool). `dataUrl` is a
   * `data:image/jpeg;base64,...` string.
   * @param {string} dataUrl
   */
  sendUserImage(dataUrl) {
    this._send({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_image", image_url: dataUrl }],
      },
    });
  }

  /**
   * Ask the model to generate a response now (after feeding tool results).
   * Serialized: if a response is already in flight we queue this request and
   * replay it once the active response finishes, so we never trip the
   * backend's `conversation_already_has_active_response` guard.
   *
   * @param {{ image?: string }} [opts] Optional `image` (a data URL) sent as a
   *   user `input_image` immediately before this response.create — so the frame
   *   travels with the create (and is deferred together with it if queued),
   *   rather than being added to the conversation eagerly. Used by the camera
   *   tool so the model sees the snapshot in the response it's about to speak.
   */
  requestResponse(opts = {}) {
    if (this._responseActive()) {
      this._createQueue.push(opts);
      if (this._debug) console.debug(`[ws] response.create queued (a response is active); pending=${this._createQueue.length}`);
      return;
    }
    this._createResponseNow(opts);
  }

  /** True while a response occupies the single backend slot. */
  _responseActive() {
    return this._openResponses > 0 || this._createInFlight;
  }

  /** Send a response.create immediately and arm the in-flight guard. Any image
   *  on the payload is added as user content right before the create.
   *  @param {{ image?: string }} [opts] */
  _createResponseNow(opts = {}) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    if (opts.image) this.sendUserImage(opts.image);
    this._createInFlight = true;
    this._send({ type: "response.create" });
  }

  /** Replay one queued response.create if the slot is now free. Called on every
   *  response.done, so queued creates drain one-per-completion. */
  _flushQueuedCreate() {
    if (this._createQueue.length > 0 && !this._responseActive()) {
      const opts = this._createQueue.shift();
      if (this._debug) console.debug(`[ws] replaying queued response.create; remaining=${this._createQueue.length}`);
      this._createResponseNow(opts);
    }
  }

  /** @param {boolean} muted */
  setMuted(muted) {
    this._muted = muted;
  }

  /**
   * Update the mic noise gate live (the user moved the Settings cursor).
   * @param {NoiseGate} gate
   */
  setNoiseGate(gate) {
    this._noiseGate = gate;
    this._captureNode?.port.postMessage({ kind: "gate", ...gate });
  }

  /** @param {Record<string, unknown>} event */
  _send(event) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    this._ws.send(JSON.stringify(event));
  }

  async close() {
    // Abort a queue wait in progress: flag it and wake the poll sleep so
    // `_pollQueue` throws "aborted" and connect() unwinds cleanly.
    this._closed = true;
    if (this._queueWake) {
      clearTimeout(this._queueTimer);
      const wake = this._queueWake;
      this._queueWake = null;
      wake();
    }
    if (this._joinTimer) {
      clearTimeout(this._joinTimer);
      this._joinTimer = 0;
    }
    if (this._joinReject) {
      const reject = this._joinReject;
      this._joinResolve = null;
      this._joinReject = null;
      reject(_codedError("join aborted", "aborted"));
    }
    this._visualiser?.stop();
    this._visualiser = null;
    try {
      if (this._ws && this._ws.readyState <= WebSocket.OPEN) {
        this._ws.close(1000, "client closed");
      }
    } catch {
      // ignored
    }
    this._ws = null;

    try {
      this._captureNode?.port.close?.();
    } catch {
      // ignored
    }
    try {
      this._micSrc?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._captureNode?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._micAnalyser?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._outAnalyser?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._playbackNode?.disconnect();
    } catch {
      // ignored
    }
    try {
      await this._ctx?.close();
    } catch {
      // ignored
    }
    this._ctx = null;
    this._captureNode = null;
    this._playbackNode = null;
    this._micSrc = null;
    this._micAnalyser = null;
    this._outAnalyser = null;
    this._setStatus("closed");
  }
}
