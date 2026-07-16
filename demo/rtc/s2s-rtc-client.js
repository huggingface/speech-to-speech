// @ts-check
/**
 * Minimal WebRTC client for a speech-to-speech realtime endpoint.
 *
 * Sibling of `ws/s2s-ws-client.js` with the SAME public surface (constructor
 * options subset, methods, dispatched events), so `main.js` can pick either
 * class from the transport setting and wire it identically. Direct mode only:
 * there is no load-balancer/queue path over WebRTC yet.
 *
 * Handshake (OpenAI Realtime GA "calls" endpoint, via the same-origin proxy):
 *
 *   1. getUserMedia -> add the mic track + create the `oai-events` data
 *      channel on an RTCPeerConnection, `createOffer`, wait for ICE gathering.
 *   2. POST the offer SDP (Content-Type: application/sdp) to `callsUrl`
 *      (usually the same-origin `api/calls` proxy, which forwards it to the
 *      s2s server's /v1/realtime/calls) -> 201 + answer SDP.
 *   3. `setRemoteDescription(answer)`; once the data channel opens the server
 *      pushes `session.created` and we reply with `session.update`.
 *
 * After that the JSON protocol on the data channel is the one the WebSocket
 * transport speaks, with the audio moved out of it:
 *
 *   - Mic audio rides the RTP media track (Opus), NOT
 *     `input_audio_buffer.append` — the server rejects `append` over WebRTC.
 *   - Assistant audio arrives as a remote media track, NOT as
 *     `response.output_audio.delta` events. There is no client playback
 *     buffer: barge-in flushing happens server-side, so `speech_started`
 *     only needs to flip the UI state here.
 *
 * Because no audio events exist, "the assistant is audibly speaking" is
 * detected from the output analyser's RMS (with a short hang time), and
 * `response.done` / `speech_started` remain the authoritative exits — the
 * same status contract the WS client exposes.
 *
 * @typedef {"idle" | "connecting" | "connected" | "user-speaking" |
 *           "processing" | "ai-speaking" | "closed" | "error"
 * } RtcStatus
 *
 * @typedef {Object} RtcClientOptions
 * @property {string} callsUrl URL to POST the SDP offer to (same-origin proxy
 *   like `api/calls`, or a direct `/v1/realtime/calls` URL when CORS allows).
 * @property {RTCIceServer[]} [iceServers] STUN/TURN servers for the peer
 *   connection (from /api/config). Empty/absent -> browser defaults (host
 *   candidates only — fine locally, may not traverse NATs).
 * @property {string} voice
 * @property {string} instructions
 * @property {MediaStream} [micStream] Live mic stream. Provide this OR `acquireMic`.
 * @property {() => Promise<MediaStream>} [acquireMic] Lazily obtain the mic
 *   stream once connect() actually runs (same contract as the WS client).
 * @property {AudioContext} [audioContext] Pre-created (and resumed) context —
 *   created inside the tap gesture so iOS lets it start.
 * @property {import("../ws/s2s-ws-client.js").ToolDef[]} [tools] Function tools
 *   declared in the initial `session.update`.
 */

import { extractResponseTranscript } from "../ws/codec.js";
import { OrbVisualiser, VIS_FFT_SIZE } from "../ws/orb-visualizer.js";

// The server's data channel label (aiortc side ignores any other label).
const DATA_CHANNEL_LABEL = "oai-events";
// Give up on the handshake if the data channel hasn't opened this long after
// the SDP answer was applied (ICE failed silently, e.g. NAT without STUN).
// The server holds its slot behind a 30 s watchdog; stay under it.
const DC_OPEN_TIMEOUT_MS = 20_000;
// Don't wait forever for ICE gathering before POSTing the offer — host
// candidates land near-instantly; STUN answers within a couple of seconds.
const ICE_GATHERING_TIMEOUT_MS = 3_000;
// Output-RMS gate for "the assistant is audibly speaking": open above the
// threshold, and hang on briefly so inter-word gaps don't flap the status.
const SPEAKING_OPEN_DB = -50;
const SPEAKING_HANG_MS = 250;
const LEVEL_POLL_MS = 50;

/** Build an Error carrying a `code` so callers can branch on the failure kind.
 *  @param {string} message @param {string} code */
function _codedError(message, code) {
  const err = /** @type {Error & { code?: string }} */ (new Error(message));
  err.code = code;
  return err;
}

export class S2sRtcRealtimeClient extends EventTarget {
  /** @param {RtcClientOptions} options */
  constructor(options) {
    super();
    /** @type {RtcClientOptions} */
    this.options = options;
    /** @type {import("../ws/s2s-ws-client.js").ToolDef[]} */
    this._tools = options.tools ?? [];
    /** @type {(() => Promise<MediaStream>) | null} */
    this._acquireMic = options.acquireMic ?? null;
    /** @type {boolean} Set by close() so late async steps unwind quietly. */
    this._closed = false;
    /** @type {RTCPeerConnection | null} */
    this._pc = null;
    /** @type {RTCDataChannel | null} */
    this._dc = null;
    /** @type {AudioContext | null} */
    this._ctx = null;
    /** @type {MediaStreamAudioSourceNode | null} */
    this._micSrc = null;
    /** @type {MediaStreamAudioSourceNode | null} */
    this._remoteSrc = null;
    /** @type {HTMLAudioElement | null} Muted sink for the Chrome quirk: a
     * remote WebRTC track stays silent in WebAudio unless the stream is ALSO
     * attached to an <audio> element. Never added to the DOM. */
    this._quirkAudio = null;
    /** @type {AnalyserNode | null} */
    this._micAnalyser = null;
    /** @type {AnalyserNode | null} */
    this._outAnalyser = null;
    /** @type {OrbVisualiser | null} */
    this._visualiser = null;
    /** @type {number} Level/status poll timer (setInterval id). */
    this._levelTimer = 0;
    /** @type {Uint8Array} Scratch buffer for time-domain RMS reads. */
    this._levelBuf = new Uint8Array(VIS_FFT_SIZE);
    /** @type {number} Last time the output RMS was above the speaking gate. */
    this._lastAudibleAt = 0;
    /** @type {RtcStatus} */
    this._status = "idle";
    this._aiSpeaking = false;
    /** @type {string} The response currently holding the backend slot (from
     * response.created), so RMS-detected audio can be attributed to it. */
    this._activeResponseId = "";
    /** @type {Set<string>} response_ids that audibly played, so the UI can
     * tell a barge-in cut (keep it) from a never-heard speculative response
     * (drop it). Attribution is by "RMS opened while this response was
     * active" — sound can't be tied to a response id without audio events. */
    this._audibleResponses = new Set();
    /** @type {Map<string, string>} CURRENT assistant transcript segment per
     * response, accumulated from streamed deltas (reset on segment done). */
    this._asstTranscriptByResp = new Map();
    /** @type {Map<string, string>} Completed segments per response, joined. */
    this._asstFullByResp = new Map();
    this._muted = false;
    // ── Response lock ────────────────────────────────────────────────────
    // Same scheme as the WS client: the backend allows ONE response in
    // flight, so response.create is serialized. `_openResponses` counts
    // confirmed-but-unfinished responses; `_createInFlight` covers the window
    // between our create and its response.created echo; extra creates queue.
    this._openResponses = 0;
    this._createInFlight = false;
    /** @type {{ image?: string }[]} */
    this._createQueue = [];
    this._sessionConfigured = false;
    this._debug = (() => { try { return localStorage.getItem("s2s.debug") === "1"; } catch { return false; } })();
  }

  get status() {
    return this._status;
  }

  /** @param {RtcStatus} status */
  _setStatus(status) {
    if (this._status === status) return;
    this._status = status;
    this.dispatchEvent(new CustomEvent("status", { detail: { status } }));
  }

  /** Full assistant transcript so far for a response: completed segments plus
   *  the in-progress one. @param {string} rid @returns {string} */
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
   * Full handshake. Resolves once the data channel is open AND the audio
   * graph is wired (mic track sending, remote track feeding the analysers).
   * @returns {Promise<void>}
   */
  async connect() {
    if (this._pc) throw new Error("Already connected");
    this._setStatus("connecting");

    if (!this.options.micStream && this._acquireMic) {
      this.options.micStream = await this._acquireMic();
    }
    if (this._closed) throw _codedError("connect aborted", "aborted");

    this._setupAudioGraph();

    const pc = new RTCPeerConnection(
      this.options.iceServers?.length ? { iceServers: this.options.iceServers } : {},
    );
    this._pc = pc;

    pc.addEventListener("track", (e) => this._onRemoteTrack(e));
    pc.addEventListener("connectionstatechange", () => this._onConnectionState());

    const micTrack = this.options.micStream?.getAudioTracks()[0];
    if (!micTrack) throw new Error("No microphone track available");
    micTrack.enabled = !this._muted;
    pc.addTrack(micTrack, /** @type {MediaStream} */ (this.options.micStream));

    // The client opens the events channel (OpenAI convention); the server
    // waits for it and ignores channels with any other label.
    const dc = pc.createDataChannel(DATA_CHANNEL_LABEL, { ordered: true });
    this._dc = dc;
    dc.addEventListener("message", (e) => this._onDcMessage(e.data));
    dc.addEventListener("close", () => this._onDcClose());

    await pc.setLocalDescription(await pc.createOffer());
    // No trickle ICE over the one-shot HTTP handshake: the offer must carry
    // our candidates, so wait for gathering (bounded — host candidates are
    // near-instant, and a STUN timeout shouldn't stall the connect).
    await this._waitIceGathering(pc);
    if (this._closed) throw _codedError("connect aborted", "aborted");

    const answerSdp = await this._postOffer(pc.localDescription?.sdp ?? "");
    if (this._closed) throw _codedError("connect aborted", "aborted");
    await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

    await this._waitDataChannelOpen(dc);
    this._startLevelLoop();
    // The server pushes session.created once it sees the channel open; the
    // session.update reply is sent from that handler (mirrors the WS client).
  }

  /** @param {RTCPeerConnection} pc */
  _waitIceGathering(pc) {
    if (pc.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        pc.removeEventListener("icegatheringstatechange", check);
        clearTimeout(timer);
        resolve(undefined);
      };
      const check = () => {
        if (pc.iceGatheringState === "complete") done();
      };
      const timer = setTimeout(() => {
        console.warn("[rtc] ICE gathering timed out; sending offer with partial candidates");
        done();
      }, ICE_GATHERING_TIMEOUT_MS);
      pc.addEventListener("icegatheringstatechange", check);
    });
  }

  /**
   * POST the SDP offer, return the answer SDP. Non-2xx bodies are surfaced as
   * errors ("all slots in use" arrives as a JSON error event with a 503).
   * @param {string} offerSdp @returns {Promise<string>}
   */
  async _postOffer(offerSdp) {
    console.log("[rtc] POST", this.options.callsUrl);
    const response = await fetch(this.options.callsUrl, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: offerSdp,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      if (response.status === 503) {
        // The s2s server rejects with a JSON error event when the pool is full.
        try {
          const j = JSON.parse(text);
          const msg = j?.error?.message;
          if (msg) throw _codedError(msg, "busy");
        } catch (err) {
          if (err instanceof Error && /** @type {any} */ (err).code === "busy") throw err;
        }
        throw _codedError("The speech service is busy — try again shortly.", "busy");
      }
      throw new Error(`WebRTC handshake failed (${response.status}): ${text.slice(0, 200)}`);
    }
    return response.text();
  }

  /** @param {RTCDataChannel} dc */
  _waitDataChannelOpen(dc) {
    if (dc.readyState === "open") return Promise.resolve();
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer);
        dc.removeEventListener("open", onOpen);
        dc.removeEventListener("close", onClose);
        dc.removeEventListener("error", onClose);
      };
      const onOpen = () => {
        cleanup();
        resolve(undefined);
      };
      const onClose = () => {
        cleanup();
        reject(new Error("WebRTC data channel closed before opening"));
      };
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error(
          "WebRTC connection timed out. If the server is remote, it may need a STUN/TURN server (RTC_ICE_SERVERS).",
        ));
      }, DC_OPEN_TIMEOUT_MS);
      dc.addEventListener("open", onOpen);
      dc.addEventListener("close", onClose);
      dc.addEventListener("error", onClose);
    });
  }

  // ── Audio graph ───────────────────────────────────────────────────────────
  // Mic:    micStream ─→ micAnalyser (orb bars + level meter; the track itself
  //         goes to the peer connection untouched).
  // Output: remote track ─→ remoteSrc ─→ outAnalyser ─→ destination, plus the
  //         hidden muted <audio> that makes Chrome actually run the track.

  _setupAudioGraph() {
    const ctx = this.options.audioContext ?? new AudioContext({ latencyHint: "interactive" });
    this._ctx = ctx;
    if (ctx.state === "suspended") {
      // Best-effort here — on iOS the resume that counts happened in the tap.
      void ctx.resume().catch((err) => console.warn("[rtc] AudioContext resume failed:", err));
    }

    const micAnalyser = ctx.createAnalyser();
    micAnalyser.fftSize = VIS_FFT_SIZE;
    micAnalyser.smoothingTimeConstant = 0;
    const micSrc = ctx.createMediaStreamSource(/** @type {MediaStream} */ (this.options.micStream));
    micSrc.connect(micAnalyser);
    this._micSrc = micSrc;
    this._micAnalyser = micAnalyser;

    const outAnalyser = ctx.createAnalyser();
    outAnalyser.fftSize = VIS_FFT_SIZE;
    outAnalyser.smoothingTimeConstant = 0.3;
    outAnalyser.connect(ctx.destination);
    this._outAnalyser = outAnalyser;

    this._visualiser = new OrbVisualiser(micAnalyser, outAnalyser, () => this._aiSpeaking);
    this._visualiser.start();
  }

  /** @param {RTCTrackEvent} e */
  _onRemoteTrack(e) {
    if (e.track.kind !== "audio" || !this._ctx || !this._outAnalyser) return;
    const stream = e.streams[0] ?? new MediaStream([e.track]);
    // Chrome quirk: without an <audio> sink the remote track produces silence
    // in WebAudio. Muted so playback happens once, through the analyser path.
    const quirk = new Audio();
    quirk.muted = true;
    quirk.srcObject = stream;
    this._quirkAudio = quirk;
    this._remoteSrc?.disconnect();
    this._remoteSrc = this._ctx.createMediaStreamSource(stream);
    this._remoteSrc.connect(this._outAnalyser);
    if (this._debug) console.debug("[rtc] remote audio track attached");
  }

  // ── Output-RMS status + mic level ─────────────────────────────────────────

  _startLevelLoop() {
    if (this._levelTimer) return;
    this._levelTimer = window.setInterval(() => this._pollLevels(), LEVEL_POLL_MS);
  }

  /** RMS (0..1) of an analyser's current time-domain buffer.
   *  @param {AnalyserNode} analyser */
  _rmsOf(analyser) {
    analyser.getByteTimeDomainData(this._levelBuf);
    let sum = 0;
    for (let i = 0; i < this._levelBuf.length; i++) {
      const s = (this._levelBuf[i] - 128) / 128;
      sum += s * s;
    }
    return Math.sqrt(sum / this._levelBuf.length);
  }

  _pollLevels() {
    if (!this._micAnalyser || !this._outAnalyser) return;

    // Mic level for the Settings meter / gate arc (raw, pre-mute).
    this.dispatchEvent(
      new CustomEvent("input-level", { detail: { rms: this._rmsOf(this._micAnalyser) } }),
    );

    // Speaking gate on the output: RMS opens the state; `response.done` and
    // `speech_started` (the protocol's authoritative signals) close it.
    const rms = this._rmsOf(this._outAnalyser);
    const db = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
    const now = performance.now();
    if (db > SPEAKING_OPEN_DB) this._lastAudibleAt = now;
    const audible = now - this._lastAudibleAt < SPEAKING_HANG_MS;
    if (audible && !this._aiSpeaking) {
      this._aiSpeaking = true;
      if (this._activeResponseId) this._audibleResponses.add(this._activeResponseId);
      this._markAudible();
    }
  }

  // ── Data channel protocol ─────────────────────────────────────────────────

  /** @param {unknown} raw */
  _onDcMessage(raw) {
    if (typeof raw !== "string") return;
    let event;
    try {
      event = JSON.parse(raw);
    } catch {
      return;
    }

    const type = event?.type;
    if (typeof type !== "string") return;
    if (this._debug) {
      const extra = type.startsWith("conversation.item.input_audio_transcription")
        ? ` item=${event.item_id} ${event.delta ?? event.transcript ?? ""}`
        : type.startsWith("response.")
          ? ` resp=${event.response_id ?? event.response?.id ?? ""} status=${event.response?.status ?? ""}`
          : "";
      console.debug(`[rtc] ${type}${extra}`);
    }

    switch (type) {
      case "session.created":
        // Server-side defaults are already what we want; push only the
        // user-tunable bits (voice, instructions, tools) — same as WS.
        this._sendSessionUpdate();
        this._sessionConfigured = true;
        if (this._status === "connecting") this._setStatus("connected");
        break;

      case "session.updated":
        break;

      case "input_audio_buffer.speech_started":
        // Barge-in: unlike WS there is no client playback buffer to clear —
        // the server flushes its track buffer — so this is UI state only.
        this._aiSpeaking = false;
        this._lastAudibleAt = 0;
        this._setStatus("user-speaking");
        break;

      case "input_audio_buffer.speech_stopped":
        if (this._status === "user-speaking") this._setStatus("processing");
        break;

      case "response.created":
        this._openResponses++;
        this._createInFlight = false;
        this._activeResponseId = event.response?.id ?? "";
        if (this._status === "connected" || this._status === "user-speaking") {
          this._setStatus("processing");
        }
        break;

      case "response.output_item.added":
        if (this._status === "connected" || this._status === "user-speaking") {
          this._setStatus("processing");
        }
        break;

      case "response.done": {
        this._aiSpeaking = false;
        this._lastAudibleAt = 0;
        this._openResponses = Math.max(0, this._openResponses - 1);
        if (this._status === "ai-speaking" || this._status === "processing") {
          this._setStatus("connected");
        }
        // Completion AND cancellation both arrive as response.done (status
        // "cancelled") — same contract the WS client documents.
        const status = event.response?.status ?? "completed";
        const responseId = event.response?.id ?? "";
        if (this._activeResponseId === responseId) this._activeResponseId = "";
        const audible = responseId ? this._audibleResponses.has(responseId) : false;
        this._audibleResponses.delete(responseId);
        const transcript =
          extractResponseTranscript(event.response) ||
          this._asstDisplay(responseId) ||
          "";
        this._asstTranscriptByResp.delete(responseId);
        this._asstFullByResp.delete(responseId);
        this.dispatchEvent(new CustomEvent("response-finished", {
          detail: { responseId, status, audible, transcript },
        }));
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
          console.warn(`[rtc] function_call_arguments.done with no name (call_id=${callId}); cannot run tool — turn may stall`);
        }
        break;
      }

      case "conversation.item.input_audio_transcription.delta": {
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          // The delta carries the full cumulative transcript so far; itemId is
          // reused across a speculative continuation so the UI groups them.
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
        this._markAudible();
        const rid = typeof event.response_id === "string" ? event.response_id : "";
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          this._asstTranscriptByResp.set(rid, (this._asstTranscriptByResp.get(rid) || "") + delta);
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
        // ONE completed segment; a response can emit several — concatenate
        // until response.done clears the accumulator.
        const segment =
          (typeof event.transcript === "string" && event.transcript) ||
          this._asstTranscriptByResp.get(rid) ||
          "";
        this._asstTranscriptByResp.delete(rid);
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
        console.error("[rtc] server error:", err);
        // Our optimistic create collided with a still-running response: clear
        // the guard and re-queue for the next response.done (never retried
        // immediately — that would just collide again).
        if (err?.type === "conversation_already_has_active_response" ||
            err?.code === "conversation_already_has_active_response") {
          if (this._createInFlight) {
            this._createInFlight = false;
            this._createQueue.push({});
          }
          break;
        }
        // Every other server error is non-fatal: surface for logging, keep
        // the session alive. Transport failures come through their own paths.
        this.dispatchEvent(
          new CustomEvent("server-error", { detail: { error: new Error(err?.message ?? "Server error") } }),
        );
        break;
      }
    }
  }

  _onDcClose() {
    if (this._closed) return;
    if (this._status === "closed" || this._status === "error") return;
    console.log("[rtc] data channel closed by server");
    this.dispatchEvent(
      new CustomEvent("error", { detail: { error: new Error("Connection closed by the server") } }),
    );
    this._setStatus("error");
  }

  _onConnectionState() {
    const state = this._pc?.connectionState;
    if (this._debug) console.debug(`[rtc] connection state: ${state}`);
    if (this._closed) return;
    if (state === "failed" || state === "disconnected") {
      if (this._status === "closed" || this._status === "error") return;
      this.dispatchEvent(
        new CustomEvent("error", { detail: { error: new Error(`WebRTC connection ${state}`) } }),
      );
      this._setStatus("error");
    }
  }

  _sendSessionUpdate() {
    // Minimal payload, exactly like the WS client: the server's pydantic
    // validator rejects the whole event on unknown sub-field shapes.
    /** @type {Record<string, any>} */
    const session = {
      type: "realtime",
      instructions: this.options.instructions,
      audio: {
        output: { voice: this.options.voice },
      },
    };
    if (this._tools.length) {
      session.tools = this._tools;
      session.tool_choice = "auto";
    }
    this._send({ type: "session.update", session });
  }

  /** Update voice/instructions on a live session without tearing down.
   *  @param {{ voice?: string; instructions?: string }} patch */
  updateSession(patch) {
    /** @type {Record<string, any>} */
    const session = { type: "realtime" };
    if (patch.instructions) session.instructions = patch.instructions;
    if (patch.voice) session.audio = { output: { voice: patch.voice } };
    if (Object.keys(session).length > 1) {
      this._send({ type: "session.update", session });
    }
  }

  /** Replace the declared tool set on a live session. Always sends `tools` —
   *  an empty array clears them. @param {import("../ws/s2s-ws-client.js").ToolDef[]} tools */
  setTools(tools) {
    this._tools = tools;
    this._send({
      type: "session.update",
      session: { type: "realtime", tools, tool_choice: tools.length ? "auto" : "none" },
    });
  }

  /** Return a tool's result to the model. @param {string} callId @param {string} output */
  sendToolOutput(callId, output) {
    if (!callId) return;
    this._send({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: callId, output },
    });
  }

  /** Add an image to the conversation as user content (camera tool). The
   *  caller keeps `dataUrl` under the data-channel message budget — SCTP
   *  messages above the negotiated max (64 KiB on aiortc) fail to send.
   *  @param {string} dataUrl */
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

  /** Ask the model to respond now (after tool results). Serialized behind the
   *  single-response slot, same as the WS client.
   *  @param {{ image?: string }} [opts] */
  requestResponse(opts = {}) {
    if (this._responseActive()) {
      this._createQueue.push(opts);
      if (this._debug) console.debug(`[rtc] response.create queued; pending=${this._createQueue.length}`);
      return;
    }
    this._createResponseNow(opts);
  }

  _responseActive() {
    return this._openResponses > 0 || this._createInFlight;
  }

  /** @param {{ image?: string }} [opts] */
  _createResponseNow(opts = {}) {
    if (!this._dc || this._dc.readyState !== "open") return;
    if (opts.image) this.sendUserImage(opts.image);
    this._createInFlight = true;
    this._send({ type: "response.create" });
  }

  _flushQueuedCreate() {
    if (this._createQueue.length > 0 && !this._responseActive()) {
      const opts = this._createQueue.shift();
      if (this._debug) console.debug(`[rtc] replaying queued response.create; remaining=${this._createQueue.length}`);
      this._createResponseNow(opts);
    }
  }

  /** @param {boolean} muted */
  setMuted(muted) {
    this._muted = muted;
    // The caller (main.js) also toggles the shared micStream tracks; doing it
    // here too keeps the client correct when driven standalone. A disabled
    // track makes the browser transmit silence — the server VAD stays quiet.
    for (const track of this.options.micStream?.getAudioTracks() ?? []) {
      track.enabled = !muted;
    }
  }

  /** Noise gate is a WebSocket-transport feature (implemented in its capture
   *  worklet); the WebRTC mic path sends the raw track, so this is a no-op.
   *  @param {import("../ws/s2s-ws-client.js").NoiseGate} _gate */
  setNoiseGate(_gate) {}

  /** Queue "join" gate — LB-mode only, which WebRTC doesn't support. Present
   *  so the two clients keep the same surface for the caller. */
  join() {}

  /** @param {Record<string, unknown>} event */
  _send(event) {
    if (!this._dc || this._dc.readyState !== "open") return;
    try {
      this._dc.send(JSON.stringify(event));
    } catch (err) {
      // A message over the SCTP limit (or a racing close) lands here; keep
      // the session alive and let the caller's flow recover.
      console.error("[rtc] data channel send failed:", err);
    }
  }

  async close() {
    this._closed = true;
    if (this._levelTimer) {
      clearInterval(this._levelTimer);
      this._levelTimer = 0;
    }
    this._visualiser?.stop();
    this._visualiser = null;
    try {
      this._dc?.close();
    } catch {
      // ignored
    }
    this._dc = null;
    try {
      this._pc?.close();
    } catch {
      // ignored
    }
    this._pc = null;
    if (this._quirkAudio) {
      this._quirkAudio.srcObject = null;
      this._quirkAudio = null;
    }
    try {
      this._remoteSrc?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._micSrc?.disconnect();
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
      await this._ctx?.close();
    } catch {
      // ignored
    }
    this._ctx = null;
    this._remoteSrc = null;
    this._micSrc = null;
    this._micAnalyser = null;
    this._outAnalyser = null;
    this._setStatus("closed");
  }
}
