// @ts-check
/**
 * Demo adapter around the official OpenAI Agents SDK RealtimeSession.
 *
 * Protocol ownership stays with the SDK's stock WebSocket and WebRTC
 * transports. This adapter only preserves demo concerns: the HF queue grant,
 * browser audio graphs, user-audio replay, visualization, and UI events.
 *
 * @typedef {"websocket" | "webrtc"} TransportKind
 * @typedef {Object} NoiseGate
 * @property {boolean} enabled
 * @property {number} thresholdDb
 * @typedef {Object} ToolDef
 * @property {"function"} type
 * @property {string} name
 * @property {string} description
 * @property {object} parameters
 * @typedef {Object} SessionInfo
 * @property {string} sessionId
 * @property {string} connectUrl
 * @property {string} websocketUrl
 * @property {string} sessionToken
 * @property {number} pendingTimeoutS
 * @property {string} [tier]
 * @property {boolean} [limited]
 * @property {number} [heartbeatSec]
 * @property {number} [remainingSec]
 * @typedef {Object} ClientOptions
 * @property {TransportKind} transport
 * @property {string} [sessionUrl]
 * @property {string} [loadBalancerUrl]
 * @property {string} [directUrl]
 * @property {string} [callsUrl]
 * @property {RTCIceServer[]} [iceServers]
 * @property {string} voice
 * @property {string} instructions
 * @property {string} [startupGreeting]
 * @property {MediaStream} [micStream]
 * @property {() => Promise<MediaStream>} [acquireMic]
 * @property {AudioContext} [audioContext]
 * @property {ToolDef[]} [tools]
 * @property {NoiseGate} [noiseGate]
 * @property {string} [audioOutputId]
 * @property {(call: {name: string, arguments: string, callId: string}) => Promise<{output: string, image?: string}>} [executeTool]
 */

import { extractResponseTranscript, trimTrailingSlash } from "./ws/codec.js";
import { OrbVisualiser, VIS_FFT_SIZE } from "./ws/orb-visualizer.js";
import { SentAudioRecorder } from "./ws/user-audio-recorder.js";

const AUDIO_SAMPLE_RATE = 24_000;
const MIC_CHUNK_MS = 40;
const SPEAKING_OPEN_DB = -50;
const SPEAKING_HANG_MS = 250;

/** @param {string} message @param {string} code @param {object} [extra] */
function codedError(message, code, extra) {
  const error = /** @type {Error & {code?: string}} */ (new Error(message));
  error.code = code;
  if (extra) Object.assign(error, extra);
  return error;
}

function sdk() {
  const value = globalThis.OpenAIAgentsRealtime;
  if (!value) throw new Error("OpenAI Agents SDK bundle is not loaded");
  return value;
}

export class S2sRealtimeClient extends EventTarget {
  /** @param {ClientOptions} options */
  constructor(options) {
    super();
    this.options = options;
    this._tools = options.tools ?? [];
    this._directUrl = options.directUrl ?? "";
    this._sessionUrl = options.sessionUrl
      ? options.sessionUrl
      : options.loadBalancerUrl
        ? `${trimTrailingSlash(options.loadBalancerUrl)}/session`
        : "";
    this._acquireMic = options.acquireMic ?? null;
    this._noiseGate = options.noiseGate ?? { enabled: false, thresholdDb: -45 };
    this._status = "idle";
    this._closed = false;
    this._closing = false;
    this._muted = false;
    this._session = null;
    this._transport = null;
    this._agent = null;
    this._queueId = "";
    this._queueWake = null;
    this._queueTimer = 0;
    this._joinResolve = null;
    this._joinReject = null;
    this._joinTimer = 0;
    this._ctx = null;
    this._micSrc = null;
    this._captureNode = null;
    this._playbackNode = null;
    this._micAnalyser = null;
    this._outAnalyser = null;
    this._remoteSrc = null;
    this._audioElement = null;
    this._visualiser = null;
    this._levelTimer = 0;
    this._levelBuf = new Uint8Array(VIS_FFT_SIZE);
    this._lastAudibleAt = 0;
    this._aiSpeaking = false;
    this._activeResponseId = "";
    this._responseRequested = false;
    this._audibleResponses = new Set();
    this._asstTranscriptByResp = new Map();
    this._asstFullByResp = new Map();
    // Input transcription deltas are append-only and may overlap across turns.
    // Keep each unresolved prefix until its authoritative completion arrives.
    this._userTranscriptByItem = new Map();
    this._currentUserItemId = "";
    this._userAudioRecorder = new SentAudioRecorder({ sampleRate: AUDIO_SAMPLE_RATE });
    this._debug = (() => {
      try { return localStorage.getItem("s2s.debug") === "1"; } catch { return false; }
    })();
  }

  get status() { return this._status; }

  /** @param {string} status */
  _setStatus(status) {
    if (this._status === status) return;
    this._status = status;
    this.dispatchEvent(new CustomEvent("status", { detail: { status } }));
  }

  async connect() {
    if (this._session) throw new Error("Already connected");
    let url;
    if (this.options.transport === "websocket") {
      if (this._directUrl) {
        url = this._directUrl;
        this._setStatus("connecting");
      } else {
        if (!this._sessionUrl) throw new Error("No session endpoint or direct URL configured");
        this._setStatus("creating-session");
        const { grant, waited } = await this._createSessionOrQueue();
        if (this._closed) throw codedError("connect aborted", "aborted");
        if (waited) await this._awaitJoin(grant);
        if (this._closed) throw codedError("connect aborted", "aborted");
        this.dispatchEvent(new CustomEvent("session", { detail: { info: grant } }));
        url = grant.connectUrl;
        this._setStatus("connecting");
      }
    } else {
      url = new URL(this.options.callsUrl || "api/calls", globalThis.location?.href).href;
      this._setStatus("connecting");
    }

    if (!this.options.micStream && this._acquireMic) {
      this.options.micStream = await this._acquireMic();
    }
    if (this._closed) throw codedError("connect aborted", "aborted");
    if (!this.options.micStream?.getAudioTracks()[0]) throw new Error("No microphone track available");

    await this._setupAudio();
    const { OpenAIRealtimeWebRTC, OpenAIRealtimeWebSocket, RealtimeSession } = sdk();
    if (this.options.transport === "websocket") {
      this._transport = new OpenAIRealtimeWebSocket({ useInsecureApiKey: true });
    } else {
      // Keep startup noise from racing the initial config and greeting. The
      // prior demo client held this same gate until its data channel was ready.
      this.options.micStream.getAudioTracks()[0].enabled = false;
      this._audioElement = new Audio();
      this._audioElement.autoplay = true;
      this._transport = new OpenAIRealtimeWebRTC({
        mediaStream: this.options.micStream,
        audioElement: this._audioElement,
        useInsecureApiKey: true,
        changePeerConnection: (pc) => {
          if (this.options.iceServers?.length) {
            pc.setConfiguration({ ...pc.getConfiguration(), iceServers: this.options.iceServers });
          }
          return pc;
        },
      });
    }

    this._agent = this._buildAgent();
    this._session = new RealtimeSession(this._agent, {
      transport: this._transport,
      model: "s2s-local",
      config: this._sessionConfig(),
      tracingDisabled: true,
    });
    this._transport.on("*", (event) => this._onTransportEvent(event));
    this._transport.on("connection_change", (state) => {
      if (state === "disconnected" && !this._closing && this._status !== "error") {
        this._setStatus("error");
        this.dispatchEvent(new CustomEvent("error", {
          detail: { error: new Error("Realtime transport disconnected") },
        }));
      }
    });
    this._session.on("audio", (event) => this._onAudio(event));
    this._session.on("audio_interrupted", () => this._clearPlayback());

    await this._session.connect({ apiKey: "s2s-local", url });
    if (this.options.transport === "webrtc") this._attachRtcOutput();
    const greeting = this.options.startupGreeting?.trim();
    if (greeting) {
      this._responseRequested = true;
      this._session.sendMessage(greeting);
    }
    this.setMuted(this._muted);
    this._setStatus("connected");
  }

  _sessionConfig() {
    return {
      outputModalities: ["audio"],
      audio: {
        input: {
          format: { type: "audio/pcm", rate: AUDIO_SAMPLE_RATE },
          transcription: { model: "whisper-1" },
          turnDetection: { type: "server_vad", interruptResponse: true },
          noiseReduction: null,
        },
        output: {
          format: { type: "audio/pcm", rate: AUDIO_SAMPLE_RATE },
          speed: 1,
        },
      },
    };
  }

  _buildAgent() {
    const { RealtimeAgent, tool } = sdk();
    const tools = this._tools.map((definition) => tool({
      name: definition.name,
      description: definition.description,
      parameters: definition.parameters,
      strict: false,
      execute: async (args, _context, details) => {
        const callId = details?.toolCall?.callId || "";
        const argumentsJson = JSON.stringify(args ?? {});
        if (!this.options.executeTool) throw new Error(`No executor for ${definition.name}`);
        const result = await this.options.executeTool({
          name: definition.name,
          arguments: argumentsJson,
          callId,
        });
        if (result.image) this._session?.addImage(result.image, { triggerResponse: false });
        return result.output;
      },
    }));
    return new RealtimeAgent({
      name: "speech-to-speech-demo",
      instructions: this.options.instructions,
      voice: this.options.voice,
      tools,
    });
  }

  async _setupAudio() {
    const ctx = this.options.audioContext ?? new AudioContext({ latencyHint: "interactive" });
    this._ctx = ctx;
    if (ctx.state === "suspended") await ctx.resume().catch(() => {});
    const micSrc = ctx.createMediaStreamSource(this.options.micStream);
    this._micSrc = micSrc;
    const micAnalyser = ctx.createAnalyser();
    micAnalyser.fftSize = VIS_FFT_SIZE;
    micAnalyser.smoothingTimeConstant = 0;
    micSrc.connect(micAnalyser);
    this._micAnalyser = micAnalyser;

    if (this.options.transport === "websocket") {
      const base = new URL("./worklets/", import.meta.url);
      await ctx.audioWorklet.addModule(new URL("mic-capture.js", base).href);
      await ctx.audioWorklet.addModule(new URL("audio-playback.js", base).href);
      const capture = new AudioWorkletNode(ctx, "mic-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        processorOptions: { chunkMs: MIC_CHUNK_MS },
      });
      capture.port.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) this._onMicChunk(event.data);
        else if (event.data?.kind === "level") {
          this.dispatchEvent(new CustomEvent("input-level", { detail: { rms: event.data.rms } }));
        }
      };
      capture.port.postMessage({ kind: "gate", ...this._noiseGate });
      micSrc.connect(capture);
      this._captureNode = capture;

      const playback = new AudioWorkletNode(ctx, "audio-playback", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      playback.port.postMessage({ kind: "config", inputRate: AUDIO_SAMPLE_RATE });
      const output = ctx.createAnalyser();
      output.fftSize = VIS_FFT_SIZE;
      output.smoothingTimeConstant = 0.3;
      playback.connect(output);
      output.connect(ctx.destination);
      this._playbackNode = playback;
      this._outAnalyser = output;
      this._visualiser = new OrbVisualiser(micAnalyser, output, () => this._aiSpeaking);
      this._visualiser.start();
    }
    await this.setAudioOutputDevice(this.options.audioOutputId || "");
  }

  _attachRtcOutput() {
    if (this._remoteSrc) return;
    const stream = this._audioElement?.srcObject;
    if (!(stream instanceof MediaStream) || !this._ctx || !this._micAnalyser) {
      this._audioElement?.addEventListener("loadedmetadata", () => this._attachRtcOutput(), { once: true });
      return;
    }
    this._audioElement.muted = true;
    const source = this._ctx.createMediaStreamSource(stream);
    const output = this._ctx.createAnalyser();
    output.fftSize = VIS_FFT_SIZE;
    output.smoothingTimeConstant = 0.3;
    source.connect(output);
    output.connect(this._ctx.destination);
    this._remoteSrc = source;
    this._outAnalyser = output;
    this._visualiser = new OrbVisualiser(this._micAnalyser, output, () => this._aiSpeaking);
    this._visualiser.start();
    this._levelTimer = window.setInterval(() => this._pollRtcLevels(), 50);
  }

  _pollRtcLevels() {
    const input = this._rms(this._micAnalyser);
    this.dispatchEvent(new CustomEvent("input-level", { detail: { rms: input } }));
    const output = this._rms(this._outAnalyser);
    const now = performance.now();
    if (output > Math.pow(10, SPEAKING_OPEN_DB / 20)) {
      this._lastAudibleAt = now;
      if (this._activeResponseId) this._audibleResponses.add(this._activeResponseId);
      this._aiSpeaking = true;
      this._markAudible();
    } else if (this._aiSpeaking && now - this._lastAudibleAt > SPEAKING_HANG_MS) {
      this._aiSpeaking = false;
    }
  }

  /** @param {AnalyserNode | null} analyser */
  _rms(analyser) {
    if (!analyser) return 0;
    analyser.getByteTimeDomainData(this._levelBuf);
    let sum = 0;
    for (const value of this._levelBuf) {
      const sample = (value - 128) / 128;
      sum += sample * sample;
    }
    return Math.sqrt(sum / this._levelBuf.length);
  }

  /** @param {{data: ArrayBuffer, responseId?: string}} event */
  _onAudio(event) {
    if (this.options.transport !== "websocket" || !this._playbackNode) return;
    const view = new DataView(event.data);
    const samples = new Float32Array(event.data.byteLength / 2);
    for (let i = 0; i < samples.length; i += 1) {
      const sample = view.getInt16(i * 2, true);
      samples[i] = sample < 0 ? sample / 0x8000 : sample / 0x7fff;
    }
    this._playbackNode.port.postMessage({ kind: "audio", samples }, [samples.buffer]);
    if (event.responseId) this._audibleResponses.add(event.responseId);
    this._aiSpeaking = true;
    this._markAudible();
  }

  _clearPlayback() {
    this._playbackNode?.port.postMessage({ kind: "clear" });
    this._aiSpeaking = false;
  }

  /** @param {ArrayBuffer} buffer */
  _onMicChunk(buffer) {
    if (!this._session || this._muted || this._status === "connecting") return;
    this._session.sendAudio(buffer);
    this._userAudioRecorder.append(buffer);
  }

  /** @param {any} event */
  _onTransportEvent(event) {
    const type = event?.type;
    if (typeof type !== "string") return;
    if (this._debug) console.debug(`[${this.options.transport}]`, event);
    switch (type) {
      case "input_audio_buffer.speech_started": {
        this._clearPlayback();
        const itemId = typeof event.item_id === "string" ? event.item_id : "";
        this._currentUserItemId = itemId;
        if (this.options.transport === "websocket") {
          this._userAudioRecorder.speechStarted({
            itemId,
            audioStartMs: Number(event.audio_start_ms),
          });
        }
        this.dispatchEvent(new CustomEvent("user-turn-started", { detail: { itemId } }));
        this._setStatus("user-speaking");
        break;
      }
      case "input_audio_buffer.speech_stopped": {
        const itemId = typeof event.item_id === "string" ? event.item_id : "";
        if (this.options.transport === "websocket") {
          const recording = this._userAudioRecorder.speechStopped({
            itemId,
            audioEndMs: Number(event.audio_end_ms),
          });
          if (recording) this.dispatchEvent(new CustomEvent("user-audio", { detail: recording }));
        }
        this.dispatchEvent(new CustomEvent("user-turn-stopped", { detail: { itemId } }));
        if (this._status === "user-speaking") this._setStatus("processing");
        break;
      }
      case "response.created":
        this._responseRequested = false;
        this._activeResponseId = event.response?.id ?? "";
        if (this._status === "connected" || this._status === "user-speaking") this._setStatus("processing");
        break;
      case "response.content_part.added":
        if (event.part?.type === "audio" || event.part?.type === "output_audio") this._markAudible();
        break;
      case "conversation.item.input_audio_transcription.delta": {
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          const itemId = typeof event.item_id === "string" ? event.item_id : "";
          const text = (this._userTranscriptByItem.get(itemId) || "") + delta;
          this._userTranscriptByItem.set(itemId, text);
          this.dispatchEvent(new CustomEvent("transcript", { detail: {
            role: "user", text, partial: true, itemId,
          } }));
        }
        break;
      }
      case "conversation.item.input_audio_transcription.completed": {
        const text = typeof event.transcript === "string" ? event.transcript : "";
        const itemId = typeof event.item_id === "string" ? event.item_id : "";
        const hadPartial = Boolean(this._userTranscriptByItem.get(itemId));
        this._userTranscriptByItem.delete(itemId);
        const isCurrentUserItem = this._currentUserItemId
          ? Boolean(itemId) && itemId === this._currentUserItemId
          : true;
        if (text || hadPartial) this.dispatchEvent(new CustomEvent("transcript", { detail: {
          role: "user", text, partial: false, itemId,
        } }));
        if (
          !text
          && isCurrentUserItem
          && this._status === "processing"
          && !this._activeResponseId
          && !this._responseRequested
        ) {
          this._setStatus("connected");
        }
        if (itemId && itemId === this._currentUserItemId) this._currentUserItemId = "";
        break;
      }
      case "response.audio_transcript.delta":
      case "response.output_audio_transcript.delta": {
        const responseId = typeof event.response_id === "string" ? event.response_id : "";
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          this._asstTranscriptByResp.set(responseId, (this._asstTranscriptByResp.get(responseId) || "") + delta);
          this.dispatchEvent(new CustomEvent("transcript", { detail: {
            role: "assistant", text: this._asstDisplay(responseId), partial: true, responseId,
          } }));
        }
        break;
      }
      case "response.audio_transcript.done":
      case "response.output_audio_transcript.done": {
        const responseId = typeof event.response_id === "string" ? event.response_id : "";
        const segment = event.transcript || this._asstTranscriptByResp.get(responseId) || "";
        this._asstTranscriptByResp.delete(responseId);
        if (segment) {
          const previous = this._asstFullByResp.get(responseId) || "";
          this._asstFullByResp.set(responseId, previous ? `${previous} ${segment}` : segment);
          this.dispatchEvent(new CustomEvent("transcript", { detail: {
            role: "assistant", text: this._asstFullByResp.get(responseId), partial: false, responseId,
          } }));
        }
        break;
      }
      case "response.done": {
        const responseId = event.response?.id ?? "";
        this._responseRequested = false;
        this._activeResponseId = "";
        this._aiSpeaking = false;
        if (this._status === "ai-speaking" || this._status === "processing") this._setStatus("connected");
        const transcript = extractResponseTranscript(event.response) || this._asstDisplay(responseId) || "";
        this.dispatchEvent(new CustomEvent("response-finished", { detail: {
          responseId,
          status: event.response?.status ?? "completed",
          audible: this._audibleResponses.has(responseId),
          transcript,
        } }));
        this._audibleResponses.delete(responseId);
        this._asstTranscriptByResp.delete(responseId);
        this._asstFullByResp.delete(responseId);
        break;
      }
      case "error": {
        const message = event.error?.message ?? "Server error";
        this.dispatchEvent(new CustomEvent("server-error", { detail: { error: new Error(message) } }));
        break;
      }
    }
  }

  /** @param {string} responseId */
  _asstDisplay(responseId) {
    const full = this._asstFullByResp.get(responseId) || "";
    const partial = this._asstTranscriptByResp.get(responseId) || "";
    return full && partial ? `${full} ${partial}` : full || partial;
  }

  _markAudible() {
    if (this._status === "closed" || this._status === "error") return;
    this._setStatus("ai-speaking");
  }

  /** @param {{voice?: string, instructions?: string}} patch */
  updateSession(patch) {
    if (patch.voice) this.options.voice = patch.voice;
    if (patch.instructions) this.options.instructions = patch.instructions;
    if (!this._session) return;
    this._agent = this._buildAgent();
    void this._session.updateAgent(this._agent);
  }

  /** @param {ToolDef[]} tools */
  setTools(tools) {
    this._tools = tools;
    if (!this._session) return;
    this._agent = this._buildAgent();
    const clearTools = tools.length === 0;
    void this._session.updateAgent(this._agent).then(() => {
      // @openai/agents 0.14.3 omits the `tools` field when its list is empty.
      // Send the explicit GA clear through the stock transport hook so the
      // server does not retain tools from its deep-merged session config.
      if (clearTools && this._tools.length === 0) {
        this._transport?.sendEvent({
          type: "session.update",
          session: { type: "realtime", tools: [], tool_choice: "none" },
        });
      }
    });
  }

  /** @param {{image?: string}} [options] */
  requestResponse(options = {}) {
    if (options.image) this._session?.addImage(options.image, { triggerResponse: false });
    this._responseRequested = true;
    this._transport?.requestResponse();
  }

  /** @param {string} dataUrl */
  sendUserImage(dataUrl) {
    this._session?.addImage(dataUrl, { triggerResponse: false });
  }

  /** @param {boolean} muted */
  setMuted(muted) {
    this._muted = muted;
    if (this.options.transport === "webrtc") this._session?.mute(muted);
  }

  /** @param {NoiseGate} gate */
  setNoiseGate(gate) {
    this._noiseGate = gate;
    this._captureNode?.port.postMessage({ kind: "gate", ...gate });
  }

  /** @param {string} [deviceId] */
  async setAudioOutputDevice(deviceId = "") {
    if (!this._ctx || typeof /** @type {any} */ (this._ctx).setSinkId !== "function") return false;
    try {
      await /** @type {any} */ (this._ctx).setSinkId(deviceId || "");
      return true;
    } catch (error) {
      console.warn("[audio] setSinkId failed:", error);
      return false;
    }
  }

  async _createSessionOrQueue() {
    const first = await this._postSession();
    if (first.state !== "queued") return { grant: first.grant, waited: false };
    this._setStatus("queued");
    return { grant: await this._pollQueue(first), waited: true };
  }

  /** @param {SessionInfo} grant */
  _awaitJoin(grant) {
    const windowSeconds = Math.max(3, (grant.pendingTimeoutS || 60) - 3);
    this._setStatus("your-turn");
    this.dispatchEvent(new CustomEvent("ready-to-join", {
      detail: { info: grant, expiresSec: windowSeconds },
    }));
    return new Promise((resolve, reject) => {
      this._joinResolve = resolve;
      this._joinReject = reject;
      this._joinTimer = setTimeout(() => {
        this._joinResolve = null;
        this._joinReject = null;
        reject(codedError("Your spot expired", "join-expired"));
      }, windowSeconds * 1000);
    });
  }

  join() {
    if (this._joinTimer) clearTimeout(this._joinTimer);
    this._joinTimer = 0;
    void this.options.audioContext?.resume().catch(() => {});
    const resolve = this._joinResolve;
    this._joinResolve = null;
    this._joinReject = null;
    resolve?.();
  }

  async _postSession() {
    const response = await fetch(this._sessionUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (response.status === 402) {
      const body = await response.json().catch(() => ({}));
      throw codedError("Daily conversation limit reached", "limit", { tier: body?.tier });
    }
    if (response.status === 401) {
      const body = await response.json().catch(() => ({}));
      throw codedError("Sign in again to continue", "login-required", { loginUrl: body?.loginUrl });
    }
    if (response.status === 503) {
      const body = await response.json().catch(() => ({}));
      if (body?.state === "at_capacity") throw codedError("The queue is full — try again shortly.", "queue-full");
    }
    if (!response.ok) throw new Error(`/session failed (${response.status}): ${await response.text()}`);
    const json = await response.json();
    if (json.state === "queued") return {
      state: "queued", queueId: json.queue_id, position: json.position, pollIntervalS: json.poll_interval_s,
    };
    return { state: "granted", grant: this._parseGrant(json) };
  }

  /** @param {{queueId: string, position: number, pollIntervalS: number}} ticket */
  async _pollQueue(ticket) {
    const interval = Math.max(1, ticket.pollIntervalS || 2) * 1000;
    this._queueId = ticket.queueId;
    this._emitQueue(ticket.position);
    while (true) {
      await this._queueSleep(interval);
      if (this._closed) throw codedError("queue wait aborted", "aborted");
      let response;
      try {
        response = await fetch(`api/queue/${encodeURIComponent(this._queueId)}`);
      } catch { continue; }
      if (response.status === 402) {
        const body = await response.json().catch(() => ({}));
        throw codedError("Daily conversation limit reached", "limit", { tier: body?.tier });
      }
      if (response.status === 401) {
        const body = await response.json().catch(() => ({}));
        throw codedError("Sign in again to continue", "login-required", { loginUrl: body?.loginUrl });
      }
      if (response.status === 404) throw codedError("Queue timed out", "queue-expired");
      if (!response.ok) continue;
      const json = await response.json().catch(() => null);
      if (!json) continue;
      if (json.state === "queued") {
        this._emitQueue(json.position);
        continue;
      }
      this._queueId = "";
      return this._parseGrant(json);
    }
  }

  /** @param {number} position */
  _emitQueue(position) {
    this.dispatchEvent(new CustomEvent("queue", { detail: { position, queueId: this._queueId } }));
  }

  /** @param {number} ms */
  _queueSleep(ms) {
    return new Promise((resolve) => {
      this._queueWake = resolve;
      this._queueTimer = setTimeout(() => {
        this._queueWake = null;
        resolve();
      }, ms);
    });
  }

  /** @param {any} json @returns {SessionInfo} */
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

  async close() {
    this._closed = true;
    this._closing = true;
    this._userAudioRecorder.reset();
    this._userTranscriptByItem.clear();
    this._currentUserItemId = "";
    if (this._queueWake) {
      clearTimeout(this._queueTimer);
      this._queueWake();
      this._queueWake = null;
    }
    if (this._joinTimer) clearTimeout(this._joinTimer);
    this._joinTimer = 0;
    if (this._joinReject) this._joinReject(codedError("join aborted", "aborted"));
    this._joinResolve = null;
    this._joinReject = null;
    if (this._levelTimer) clearInterval(this._levelTimer);
    this._levelTimer = 0;
    this._visualiser?.stop();
    this._visualiser = null;
    this._session?.close();
    this._session = null;
    this._transport = null;
    for (const node of [this._captureNode, this._playbackNode, this._micSrc, this._micAnalyser, this._remoteSrc, this._outAnalyser]) {
      try { node?.disconnect(); } catch { /* ignored */ }
    }
    try { await this._ctx?.close(); } catch { /* ignored */ }
    this._ctx = null;
    this._setStatus("closed");
  }
}
