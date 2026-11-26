// @ts-check
/**
 * Viva app controller: lobby → research → quiz → results.
 *
 * The speech backend is the coach's ears and mouth (see realtime.js); the
 * Python server researches topics, grades every answer and keeps score.
 * The browser glues them together and drives the avatar's body language.
 */

import { Avatar, COACHES } from "./avatar.js";
import { RealtimeVoice } from "./realtime.js";
import { confetti } from "./confetti.js";

const $ = (/** @type {string} */ sel) => /** @type {HTMLElement} */ (document.querySelector(sel));
const LEVELS = { 1: ["Easy", "easy"], 2: ["Medium", ""], 3: ["Hard", "hard"] };

const store = {
  get(/** @type {string} */ k, d = "") {
    try {
      return localStorage.getItem(`viva.${k}`) ?? d;
    } catch {
      return d;
    }
  },
  set(/** @type {string} */ k, /** @type {string} */ v) {
    try {
      localStorage.setItem(`viva.${k}`, v);
    } catch {}
  },
};

const S = {
  /** @type {any} */ cfg: { mode: "manual" },
  /** Latest speculative transcript, the one already prefetched, and when the mic was last loud. */
  snap: { text: "", sent: "", lastLoud: 0 },
  /** Line text by response id, shown when that line's audio starts. */
  lineText: new Map(),
  /** Seconds of audio received per response id, and ids whose audio is complete. */
  lineAudio: new Map(),
  lineDone: new Set(),
  coachKey: "max",
  /** @type {any} */ coach: null,
  phase: "lobby",
  difficulty: "pro",
  /** @type {any} */ pack: null,
  /** @type {any} */ game: null,
  /** @type {any} */ question: null,
  /** @type {any} */ lastState: null,
  /** @type {Avatar | null} */ avatar: null,
  avatarReady: /** @type {Promise<void> | null} */ (null),
  /** @type {RealtimeVoice | null} */ voice: null,
  voiceOk: false,
  bargeIn: store.get("bargein") === "1",
  coachTalking: false,
  responseActive: false,
  responseHadAudio: false,
  unmuteTimer: 0,
  turnChain: Promise.resolve(),
  researchAbort: /** @type {AbortController | null} */ (null),
  partial: "",
  runId: 0,
  /** @type {any} */ lastPlacement: null,
};

// ── Boot ─────────────────────────────────────────────────────────────────

async function boot() {
  S.cfg = await fetch("api/config").then((r) => r.json()).catch(() => ({ mode: "manual" }));
  const wanted = new URLSearchParams(location.search).get("coach") || store.get("coach", "max");
  S.coachKey = wanted in COACHES ? wanted : "max";
  S.coach = COACHES[/** @type {keyof typeof COACHES} */ (S.coachKey)];
  document.querySelectorAll(".coach-name").forEach((el) => (el.textContent = S.coach.name));
  setupUi();
  loadTopics();
  renderLobbyBoard();

  const avatar = new Avatar($("#avatar"), /** @type {any} */ (S.coachKey));
  S.avatar = avatar;
  S.avatarReady = avatar
    .init((ev) => {
      if (ev.lengthComputable) $("#avatar-pct").textContent = `${Math.round((ev.loaded / ev.total) * 100)}%`;
    })
    .then(() => {
      $("#avatar-loading").classList.add("done");
      updateLipsyncInfo();
    })
    .catch((err) => {
      console.error(err);
      $("#avatar-loading").innerHTML = "<span>Couldn't load the 3D avatar (WebGL needed). The quiz still works.</span>";
    });
}

function setupUi() {
  $("#coach-pick").querySelectorAll("button").forEach((b) => {
    const key = /** @type {HTMLElement} */ (b).dataset.coach || "max";
    b.classList.toggle("on", key === S.coachKey);
    b.addEventListener("click", () => {
      if (key === S.coachKey) return;
      store.set("coach", key);
      const url = new URL(location.href);
      url.searchParams.delete("coach");
      location.href = url.toString();
    });
  });
  $("#btn-signin").hidden = !S.cfg.oauth || !!S.cfg.user;
  const nameInput = /** @type {HTMLInputElement} */ ($("#player-name"));
  if (S.cfg.user) {
    $("#name-field").innerHTML = `<span>Playing as</span><span class="verified">@${escapeHtml(S.cfg.user)} ✓</span>`;
  } else {
    nameInput.value = store.get("name");
    nameInput.addEventListener("change", () => store.set("name", nameInput.value.trim()));
  }

  $("#difficulty").addEventListener("click", (e) => {
    const b = /** @type {HTMLElement} */ (e.target).closest("button");
    if (!b) return;
    S.difficulty = b.dataset.v || "pro";
    $("#difficulty").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  });

  $("#topic-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const topic = /** @type {HTMLInputElement} */ ($("#topic-input")).value.trim();
    if (topic.length < 2) return $("#topic-input").focus();
    startRun({ topic, difficulty: S.difficulty });
  });

  $("#type-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = /** @type {HTMLInputElement} */ ($("#type-input"));
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    submitTurn(text);
  });

  $("#view-quiz").querySelectorAll("[data-intent]").forEach((b) =>
    b.addEventListener("click", () => {
      const intent = /** @type {HTMLElement} */ (b).dataset.intent || "";
      submitTurn("", intent);
    }),
  );

  $("#research-cancel").addEventListener("click", () => backToLobby());
  $("#btn-new").addEventListener("click", () => backToLobby());
  $("#btn-again").addEventListener("click", () => S.pack && startRun({ packId: S.pack.id, title: S.pack.title, emoji: S.pack.emoji }));
  $("#btn-drill").addEventListener("click", () => {
    const missed = S.game?.summary?.missed || [];
    if (S.pack && missed.length) startRun({ packId: S.pack.id, title: S.pack.title, emoji: S.pack.emoji, questionIds: missed });
  });

  // Dialogs
  document.querySelectorAll("dialog [data-close]").forEach((b) =>
    b.addEventListener("click", () => /** @type {HTMLDialogElement} */ (b.closest("dialog")).close()),
  );
  document.querySelectorAll("dialog").forEach((d) =>
    d.addEventListener("click", (e) => {
      if (e.target === d) d.close();
    }),
  );
  $("#btn-board").addEventListener("click", openBoard);
  $("#lobby-board-more").addEventListener("click", openBoard);
  $("#board-period").addEventListener("click", (e) => {
    const b = /** @type {HTMLElement} */ (e.target).closest("button");
    if (!b) return;
    $("#board-period").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    refreshBoardDialog();
  });
  $("#board-topic").addEventListener("change", refreshBoardDialog);

  const barge = /** @type {HTMLInputElement} */ ($("#opt-bargein"));
  barge.checked = S.bargeIn;
  barge.addEventListener("change", () => {
    S.bargeIn = barge.checked;
    store.set("bargein", S.bargeIn ? "1" : "0");
    applyMicGate();
  });
  if (S.cfg.mode === "manual") {
    $("#server-field").hidden = false;
    const server = /** @type {HTMLInputElement} */ ($("#opt-server"));
    server.value = store.get("server", "ws://localhost:8765/v1/realtime");
    server.addEventListener("change", () => store.set("server", server.value.trim()));
  }
  $("#btn-settings").addEventListener("click", () => {
    updateLipsyncInfo();
    /** @type {HTMLDialogElement} */ ($("#settings-dialog")).showModal();
  });
}

function updateLipsyncInfo() {
  const src = S.avatar?.lipsyncSource === "server" ? "server visemes (phoneme-based)" : "in-browser HeadAudio (audio-based)";
  $("#lipsync-info").textContent = `Lip-sync: ${src}.`;
}

// ── Lobby ────────────────────────────────────────────────────────────────

async function loadTopics() {
  const data = await fetch("api/topics").then((r) => r.json()).catch(() => ({ featured: [], popular: [] }));
  const plays = Object.fromEntries((data.popular || []).map((/** @type {any} */ p) => [p.slug, p.plays]));
  const grid = $("#featured");
  grid.innerHTML = "";
  for (const p of data.featured) {
    const b = document.createElement("button");
    b.className = "card";
    b.innerHTML = `<span class="emoji">${escapeHtml(p.emoji)}</span><b>${escapeHtml(p.title)}</b><span>${escapeHtml(p.tagline || "")}</span>${
      plays[p.slug] ? `<span class="plays">${plays[p.slug]} plays</span>` : ""
    }`;
    b.addEventListener("click", () => startRun({ packId: p.id, title: p.title, emoji: p.emoji }));
    grid.append(b);
  }
  const select = /** @type {HTMLSelectElement} */ ($("#board-topic"));
  const seen = new Set();
  for (const p of [...data.featured, ...(data.popular || [])]) {
    if (seen.has(p.slug)) continue;
    seen.add(p.slug);
    select.append(new Option(`${p.emoji || "🎓"} ${p.title}`, p.slug));
  }
}

async function fetchBoard(/** @type {string} */ slug, /** @type {string} */ period, limit = 10) {
  const q = new URLSearchParams({ period, limit: String(limit) });
  if (slug) q.set("slug", slug);
  const data = await fetch(`api/leaderboard?${q}`).then((r) => r.json()).catch(() => ({ rows: [] }));
  return data.rows || [];
}

/** @param {HTMLElement} el @param {any[]} rows @param {{ me?: string, showTopic?: boolean }} [opt] */
function renderBoard(el, rows, opt = {}) {
  el.innerHTML = "";
  if (!rows.length) {
    el.innerHTML = `<li class="empty">No scores yet. Be the first!</li>`;
    return;
  }
  rows.forEach((r, i) => {
    const li = document.createElement("li");
    if (opt.me && r.player === opt.me) li.classList.add("me");
    const detail = opt.showTopic ? `${r.emoji || ""} ${r.title || ""}` : `${r.correct}/${r.total} correct · ${r.rank || ""}`;
    li.innerHTML = `<span class="pos">${i + 1}</span><span class="who"><b>${escapeHtml(r.player)}${
      r.verified ? '<span class="check">✓</span>' : ""
    }</b><span>${escapeHtml(detail)}</span></span><span class="pts">${r.score}</span>`;
    el.append(li);
  });
}

async function renderLobbyBoard() {
  let rows = await fetchBoard("", "week", 5);
  if (!rows.length) rows = await fetchBoard("", "all", 5);
  renderBoard($("#lobby-board"), rows, { showTopic: true, me: playerName() });
}

function openBoard() {
  /** @type {HTMLDialogElement} */ ($("#board-dialog")).showModal();
  refreshBoardDialog();
}

async function refreshBoardDialog() {
  const period = /** @type {HTMLElement} */ ($("#board-period").querySelector(".on"))?.dataset.v || "week";
  const slug = /** @type {HTMLSelectElement} */ ($("#board-topic")).value;
  const rows = await fetchBoard(slug, period, 25);
  renderBoard($("#board-list"), rows, { showTopic: !slug, me: playerName() });
}

function playerName() {
  return S.cfg.user || /** @type {HTMLInputElement} */ ($("#player-name")).value.trim() || store.get("name");
}

// ── Phases ───────────────────────────────────────────────────────────────

/** @param {string} phase */
function setPhase(phase) {
  S.phase = phase;
  document.body.dataset.phase = phase;
  for (const v of ["lobby", "research", "quiz", "results"]) $(`#view-${v}`).hidden = v !== phase;
  $("#hud").hidden = phase !== "quiz" && phase !== "results";
  $("#panel").scrollTop = 0;
}

function backToLobby() {
  S.runId++;
  S.researchAbort?.abort();
  S.voice?.stopSpeaking();
  S.avatar?.interrupt();
  hideCaption();
  setPhase("lobby");
  renderLobbyBoard();
  loadTopics();
}

/**
 * Start a quiz: research (or load) the pack while the voice connects.
 * @param {{ topic?: string, difficulty?: string, packId?: string, title?: string, emoji?: string, questionIds?: number[] }} opts
 */
async function startRun(opts) {
  const run = ++S.runId;
  S.avatar?.resume(); // inside the click: unlocks audio on iOS/Safari
  S.game = null;
  S.question = null;
  setPhase("research");
  $("#research-title").textContent = opts.title || opts.topic || "";
  $("#research-emoji").textContent = opts.emoji || "📚";
  $("#research-sources").innerHTML = "";
  $("#research-steps").querySelectorAll("li").forEach((li) => li.classList.remove("active", "done"));
  setResearchStage(opts.packId ? "write" : "plan");
  $("#research-steps").hidden = !!opts.packId;

  const voiceP = ensureVoice();
  let greeted = false;
  voiceP.then((ok) => {
    if (ok && run === S.runId && S.phase === "research" && !opts.packId) {
      greeted = true;
      const topic = opts.topic || "";
      const intro = topic.length <= 40 && !/[:;/]/.test(topic) ? `Ooh, ${topic}!` : "Ooh, great topic!";
      say(`${intro} Give me a few seconds to read up on it.`, "greeting");
    }
  });

  let pack = null;
  try {
    pack = opts.packId ? { id: opts.packId } : await research(opts.topic || "", opts.difficulty || "pro");
  } catch (err) {
    if (run !== S.runId) return;
    toast(/** @type {Error} */ (err).message || "Research failed.");
    backToLobby();
    return;
  }
  if (run !== S.runId) return;

  let game;
  try {
    game = await postJson("api/game", {
      pack_id: pack.id,
      player: playerName(),
      coach: S.coach.name,
      question_ids: opts.questionIds || null,
    });
  } catch (err) {
    toast(/** @type {Error} */ (err).message);
    backToLobby();
    return;
  }
  if (run !== S.runId) return;
  await Promise.race([voiceP, sleep(opts.packId ? 6000 : 1500)]);
  if (run !== S.runId) return;

  S.pack = game.pack;
  S.game = game;
  S.question = game.question;
  S.lastState = game.state;
  $("#hud-topic").textContent = `${game.pack.emoji} ${game.pack.title}`;
  updateHud(game.state, true);
  renderQuestion(game.question);
  resetAnswer();
  setPhase("quiz");
  if (!S.voiceOk) {
    toast(`Voice isn't available, so ${S.coach.name}'s lines appear as captions and you can type your answers.`, "info");
  }
  S.avatar?.setMood("happy");
  say(greeted ? `Okay, I'm ready! ${game.say}` : game.say, "opening");
}

/** @param {string} stage */
function setResearchStage(stage) {
  const order = ["plan", "search", "read", "write"];
  const idx = order.indexOf(stage);
  $("#research-steps").querySelectorAll("li").forEach((li) => {
    const i = order.indexOf(/** @type {HTMLElement} */ (li).dataset.stage || "");
    li.classList.toggle("done", i < idx);
    li.classList.toggle("active", i === idx);
  });
}

/**
 * Stream /api/research (SSE over POST) and resolve with the pack.
 * @param {string} topic @param {string} difficulty
 */
async function research(topic, difficulty) {
  S.researchAbort = new AbortController();
  const resp = await fetch("api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, difficulty, count: 8 }),
    signal: S.researchAbort.signal,
  });
  if (!resp.ok || !resp.body) throw new Error(`Research failed (${resp.status})`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) !== -1) {
      const line = buf.slice(0, cut).replace(/^data: /, "");
      buf = buf.slice(cut + 2);
      let ev;
      try {
        ev = JSON.parse(line);
      } catch {
        continue;
      }
      if (ev.type === "status") setResearchStage(ev.stage);
      else if (ev.type === "source") addSource(ev);
      else if (ev.type === "error") throw new Error(ev.message);
      else if (ev.type === "pack") {
        setResearchStage("done");
        $("#research-title").textContent = ev.pack.title;
        $("#research-emoji").textContent = ev.pack.emoji;
        return ev.pack;
      }
    }
  }
  throw new Error("Research ended without a quiz.");
}

/** @param {{ title: string, url: string, site: string }} src */
function addSource(src) {
  const a = document.createElement("a");
  a.className = "source";
  a.href = src.url;
  a.target = "_blank";
  a.rel = "noopener";
  a.innerHTML = `<b>${escapeHtml(src.title)}</b><span>${escapeHtml(src.site)}</span>`;
  $("#research-sources").append(a);
}

// ── Voice ────────────────────────────────────────────────────────────────

/** Connect mic + speech backend once; resolves true when ready. */
async function ensureVoice() {
  if (S.voice && S.voiceOk) return true;
  setVoiceLine("Connecting voice…", "");
  try {
    await S.avatarReady;
    const avatar = /** @type {Avatar} */ (S.avatar);
    const ctx = avatar.audioCtx;
    if (!ctx) throw new Error("Audio unavailable");
    const mic = new URLSearchParams(location.search).has("fakemic")
      ? fakeMic(ctx)
      : await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
        });
    const voice = new RealtimeVoice({ audioCtx: ctx, voice: S.cfg.voice || S.coach.voice });
    wireVoice(voice);
    S.voice = voice;
    const wsUrl = S.cfg.mode === "manual" ? store.get("server", "ws://localhost:8765/v1/realtime") : S.cfg.wsUrl;
    await voice.connect({ mode: S.cfg.mode, wsUrl });
    await voice.startMic(mic);
    S.voiceOk = true;
    applyMicGate();
    setVoiceLine("Voice connected", "ok");
    return true;
  } catch (err) {
    console.warn("[voice]", err);
    S.voiceOk = false;
    S.voice?.close();
    S.voice = null;
    const e = /** @type {any} */ (err);
    const msg = e?.name === "NotAllowedError" ? "Microphone blocked. You can still type." : e?.message;
    setVoiceLine(msg || "Voice unavailable", "bad");
    if (e?.code === "login") {
      $("#btn-signin").hidden = false;
      toast(`Sign in with Hugging Face (top right) to talk with ${S.coach.name}. You can still type your answers.`, "info");
    }
    return false;
  }
}

/**
 * `?fakemic=1`: a silent synthetic microphone for testing without a real one.
 * `vivaFakeMic.play(url)` speaks an audio file into it.
 * @param {AudioContext} ctx
 */
function fakeMic(ctx) {
  const dest = ctx.createMediaStreamDestination();
  Object.assign(window, {
    vivaFakeMic: {
      async play(/** @type {string} */ url) {
        const buf = await ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(dest);
        src.start();
        return new Promise((r) => (src.onended = r));
      },
    },
  });
  return dest.stream;
}

/** @param {string} text @param {string} cls */
function setVoiceLine(text, cls) {
  $("#voice-status").textContent = text;
  $("#voice-line").className = `voice-line ${cls}`;
}

/** @param {RealtimeVoice} voice */
function wireVoice(voice) {
  const avatar = /** @type {Avatar} */ (S.avatar);
  voice.addEventListener("queue", (e) => setVoiceLine(`Waiting for a voice server… #${/** @type {any} */ (e).detail.position} in line`, ""));
  voice.addEventListener("response-started", (e) => {
    const id = /** @type {any} */ (e).detail.responseId;
    S.responseActive = true;
    S.responseHadAudio = false;
    setCoachTalking(true);
    avatar.beginResponse(id);
    S.lineText.set(id, /** @type {any} */ (e).detail.text || "");
  });
  voice.addEventListener("visemes", (e) => {
    const d = /** @type {any} */ (e).detail;
    avatar.addVisemes(d.responseId, d.visemes);
  });
  voice.addEventListener("audio", (e) => {
    const d = /** @type {any} */ (e).detail;
    S.responseHadAudio = true;
    S.lineAudio.set(d.responseId, (S.lineAudio.get(d.responseId) || 0) + d.pcm.length / d.rate);
    avatar.feed(d.responseId, d.pcm);
  });
  voice.addEventListener("response-done", (e) => {
    S.responseActive = false;
    avatar.endResponse();
    // Now the line's length in seconds is known: pace its subtitles to it.
    const id = /** @type {any} */ (e).detail.responseId;
    S.lineDone.add(id);
    const secs = S.lineAudio.get(id) || 0;
    if (id === SUB.id && secs > 0.5) SUB.cps = SUB.total / secs;
    if (!S.responseHadAudio || !avatar.playing) scheduleCoachDone();
  });
  avatar.addEventListener("line-start", (e) => {
    const id = /** @type {any} */ (e).detail.responseId;
    const text = S.lineText.get(id);
    S.lineText.delete(id);
    if (text) subtitlesStart(text, id);
  });
  avatar.addEventListener("playback", (e) => {
    if (!(/** @type {any} */ (e).detail.playing) && !S.responseActive) scheduleCoachDone();
  });
  voice.addEventListener("speech-started", () => {
    if (S.phase !== "quiz") return;
    if (S.coachTalking && S.bargeIn) {
      voice.stopSpeaking();
      avatar.interrupt();
      subtitlesStop();
      setCoachTalking(false);
    }
    S.partial = "";
    if ($("#verdict").classList.contains("previous")) hideVerdict();
    setMic("hearing", "Hearing you…", "Pause when you're done");
    avatar.listen();
  });
  voice.addEventListener("user-partial", (e) => {
    if (S.phase !== "quiz") return;
    S.partial += /** @type {any} */ (e).detail.delta;
    $("#answer-text").innerHTML = `<span class="partial">${escapeHtml(S.partial)}</span>`;
  });
  voice.addEventListener("user-snapshot", (e) => {
    if (S.phase !== "quiz") return;
    S.snap.text = /** @type {any} */ (e).detail.transcript;
    if (S.snap.text) $("#answer-text").innerHTML = `<span class="partial">${escapeHtml(S.snap.text)}</span>`;
    maybeSpeculate();
  });
  voice.addEventListener("user-final", (e) => {
    S.snap.text = "";
    S.snap.sent = "";
    const text = /** @type {any} */ (e).detail.transcript;
    if (S.phase !== "quiz" || !text) {
      if (S.phase === "quiz") setMicIdle();
      return;
    }
    submitTurn(text);
  });
  voice.addEventListener("mic-level", (e) => {
    const rms = /** @type {any} */ (e).detail.rms;
    if (rms > 0.02) S.snap.lastLoud = performance.now();
    else if (S.snap.text) maybeSpeculate();
    $("#mic-ring").style.setProperty("--lvl", String(0.9 + Math.min(0.3, rms * 4)));
  });
  voice.addEventListener("error", (e) => {
    const d = /** @type {any} */ (e).detail;
    if (d.code === "closed") {
      S.voiceOk = false;
      toast("The voice connection dropped. Keep typing, or start a new quiz to reconnect.");
      setMic("off", "Voice offline", "Type your answers below");
    }
  });
}

/**
 * Grade early: once the student has gone quiet and the backend has shared the
 * pause's transcript, the coach server starts grading it while the backend
 * waits out its reopen grace. The authoritative transcript reuses the result
 * if it matches; if the student keeps talking, the guess is simply unused.
 */
const SPECULATE = !new URLSearchParams(location.search).has("nospec");

function maybeSpeculate() {
  if (!SPECULATE) return;
  const text = S.snap.text;
  if (!text || text === S.snap.sent || S.phase !== "quiz" || !S.game || S.coachTalking) return;
  if (performance.now() - S.snap.lastLoud < 250) return;
  S.snap.sent = text;
  fetch(`api/game/${S.game.game_id}/prefetch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript: text }),
  }).catch(() => {});
}

function scheduleCoachDone() {
  clearTimeout(S.unmuteTimer);
  S.unmuteTimer = window.setTimeout(() => {
    if (!S.responseActive && !S.avatar?.playing && !S.voice?.pending.length) {
      setCoachTalking(false);
      subtitlesStop(S.phase === "results" ? 6000 : 3500);
    }
  }, 350);
}

/** @param {boolean} on */
function setCoachTalking(on) {
  S.coachTalking = on;
  applyMicGate();
  if (S.phase !== "quiz") return;
  if (on) setMic("speaking", `${S.coach.name} is talking`, S.bargeIn ? "Talk over to interrupt" : "Your turn is next");
  else setMicIdle();
}

function setMicIdle() {
  if (!S.voiceOk) setMic("off", "Voice offline", "Type your answer below");
  else setMic("listening", "Your turn", "Answer out loud, or say hint, repeat or skip");
}

function applyMicGate() {
  S.voice?.setMuted(S.coachTalking && !S.bargeIn);
}

/** @param {string} state @param {string} title @param {string} sub */
function setMic(state, title, sub) {
  $("#mic").dataset.state = state;
  $("#mic-title").textContent = title;
  $("#mic-sub").textContent = sub;
}

/** Speak a line (or just caption it when there is no voice). */
function say(/** @type {string} */ text, kind = "line") {
  if (!text) return;
  if (S.voice && S.voiceOk) {
    S.voice.say(text, kind);
  } else {
    showCaption(text);
  }
}

/**
 * Subtitles: the line is cut into sentence groups and advanced at the voice's
 * pace (characters per second, re-estimated once the line's audio length is known).
 */
const SUB = { id: "", chunks: /** @type {string[]} */ ([]), ends: /** @type {number[]} */ ([]), idx: -1, total: 0, cps: 14, playAt: 0, timer: 0 };

/** @param {string} text @param {string} [id] response id, to pace by its audio length */
function subtitlesStart(text, id = "") {
  const sentences = text.match(/[^.!?]+(?:[.!?]+["')\]]*|$)\s*/g) || [text];
  const chunks = [];
  let cur = "";
  for (const sentence of sentences) {
    if (cur && (cur + sentence).length > 110) {
      chunks.push(cur.trim());
      cur = "";
    }
    cur += sentence;
  }
  if (cur.trim()) chunks.push(cur.trim());
  let acc = 0;
  const ends = chunks.map((c) => (acc += c.length + 1));
  const secs = S.lineDone.has(id) ? S.lineAudio.get(id) || 0 : 0;
  Object.assign(SUB, { id, chunks, ends, idx: 0, total: acc, cps: secs > 0.5 ? acc / secs : 14, playAt: performance.now() });
  showCaption(chunks[0] || text);
  clearInterval(SUB.timer);
  SUB.timer = window.setInterval(() => {
    if (!SUB.playAt) return;
    const chars = ((performance.now() - SUB.playAt) / 1000) * SUB.cps;
    let i = SUB.ends.findIndex((end) => chars < end);
    if (i < 0) i = SUB.chunks.length - 1;
    if (i !== SUB.idx) {
      SUB.idx = i;
      showCaption(SUB.chunks[i]);
    }
  }, 120);
}

function subtitlesStop(delay = 0) {
  clearInterval(SUB.timer);
  hideCaption(delay);
}

let captionTimer = 0;
/** @param {string} text */
function showCaption(text) {
  clearTimeout(captionTimer);
  $("#caption-text").textContent = text;
  const el = $("#caption");
  el.hidden = false;
  el.style.animation = "none";
  void el.offsetWidth;
  el.style.animation = "";
}
function hideCaption(delay = 0) {
  clearTimeout(captionTimer);
  captionTimer = window.setTimeout(() => ($("#caption").hidden = true), delay);
}

// ── Quiz turns ───────────────────────────────────────────────────────────

/** @param {string} text @param {string} [intent] */
function submitTurn(text, intent) {
  if (S.phase !== "quiz" || !S.game) return;
  S.turnChain = S.turnChain.then(() => doTurn(text, intent)).catch((err) => {
    console.error(err);
    toast(err.message || "Something went wrong.");
    setMicIdle();
  });
}

/** @param {string} text @param {string} [intent] */
async function doTurn(text, intent) {
  const game = S.game;
  if (!game || game.state?.finished) return;
  if (S.coachTalking && (intent || S.bargeIn)) {
    S.voice?.stopSpeaking();
    S.avatar?.interrupt();
    subtitlesStop();
  }
  if (text) $("#answer-text").textContent = text;
  if ($("#verdict").classList.contains("previous")) hideVerdict();
  $("#answer-box").classList.add("grading");
  setMic("thinking", "Checking…", `${S.coach.name} is listening to your answer`);
  if (!intent) S.avatar?.think();

  const out = await postJson(`api/game/${game.game_id}/turn`, { transcript: text, intent: intent || null });
  if (S.game !== game) return;
  $("#answer-box").classList.remove("grading");
  handleOutcome(out);
}

/** @param {any} out */
function handleOutcome(out) {
  const st = out.state;
  const avatar = S.avatar;
  if (out.retry || out.hint || out.clue) {
    // Still on this question: show the coach's clue, never the answer.
    showClue(out);
    avatar?.react(out.retry ? "clue" : "hint");
  } else if (out.verdict) {
    showVerdict(out);
    avatar?.react(out.verdict, { streak: st?.streak });
    if (out.points > 0) pop(`+${out.points}`, out.multiplier > 1 ? `streak ×${out.multiplier.toFixed(1)}` : "", out.verdict === "correct");
    if (out.verdict === "correct" && (st?.streak ?? 0) >= 3) confetti({ count: 60 });
  }
  if (st) {
    updateHud(st);
    if (!out.reveal) updateStakes(st);
  }

  say(out.say, out.intent);

  if (out.finished) {
    S.game.state = st;
    S.game.summary = out.summary;
    setTimeout(() => showResults(out.summary), out.verdict ? 1600 : 400);
    return;
  }
  if (out.question && S.question && out.question.index !== S.question.index) {
    // Show the next question once the coach moves on, but keep the verdict
    // card (labelled with the previous question) until the student answers.
    S.question = out.question;
    setTimeout(() => {
      if (S.question === out.question) {
        renderQuestion(out.question);
        resetAnswer(true);
      }
    }, 2600);
  }
  if (!S.voiceOk) setMicIdle();
}

/** @param {any} q */
function renderQuestion(q) {
  if (!q) return;
  const [label, cls] = LEVELS[/** @type {1|2|3} */ (q.difficulty)] || LEVELS[2];
  $("#q-count").textContent = `Question ${q.index + 1} of ${q.total}`;
  const level = $("#q-level");
  level.textContent = label;
  level.className = `pill ${cls}`;
  $("#q-points").textContent = `${q.points} pts`;
  $("#q-attempt").hidden = true;
  $("#btn-hint").toggleAttribute("disabled", !q.has_hint);
  const card = $("#q-card");
  card.style.animation = "none";
  void card.offsetWidth;
  card.style.animation = "";
  $("#q-text").textContent = q.question;
  renderDots();
}

function renderDots() {
  const st = S.lastState;
  const q = S.question;
  if (!st) return;
  const dots = $("#q-dots");
  dots.innerHTML = "";
  for (let i = 0; i < st.total; i++) {
    const dot = document.createElement("i");
    const r = (st.results || [])[i];
    if (r && r.verdict !== "pending") dot.className = r.verdict;
    else if (q && i === q.index) dot.className = "current";
    dots.append(dot);
  }
}

/** @param {boolean} [keepVerdict] keep the previous question's verdict visible */
function resetAnswer(keepVerdict = false) {
  S.partial = "";
  $("#answer-text").innerHTML = `<span class="muted">Answer out loud when ${S.coach.name} finishes asking.</span>`;
  $("#answer-box").classList.remove("grading");
  if (keepVerdict && !$("#verdict").hidden) {
    $("#verdict").classList.add("previous");
    $("#verdict").dataset.label = `Question ${S.question.index} result`;
  } else {
    hideVerdict();
  }
  if (!S.coachTalking) setMicIdle();
}

function hideVerdict() {
  $("#verdict").hidden = true;
  $("#verdict").classList.remove("previous");
}

const VERDICT_LABEL = { correct: "✓ Correct", partial: "◐ Partly there", incorrect: "✗ Not quite", skipped: "⏭ Skipped" };

/**
 * Points still on the table for the current question (tries and hints cost).
 * @param {any} st
 */
function updateStakes(st) {
  const q = S.question;
  if (!q) return;
  const factor = st.attempt_factor ?? 1;
  $("#q-points").textContent = `${Math.round(q.points * factor * (st.hint_used ? 0.8 : 1))} pts`;
  const pill = $("#q-attempt");
  pill.hidden = (st.attempt ?? 1) <= 1;
  pill.textContent = `Try ${st.attempt} of ${st.max_attempts}`;
}

const CLUE_LABEL = { partial: "◐ Almost there", incorrect: "✗ Not yet", stuck: "💡 Let's work it out" };

/** @param {any} out */
function showClue(out) {
  const v = $("#verdict");
  v.hidden = false;
  v.classList.remove("previous");
  v.dataset.v = "clue";
  $("#verdict-chip").textContent = out.retry
    ? CLUE_LABEL[/** @type {keyof typeof CLUE_LABEL} */ (out.verdict)] || "💡 Keep going"
    : "💡 Hint";
  $("#verdict-points").textContent = !out.retry ? "" : out.attempts_left === 0 ? "Last try" : `${out.attempts_left + 1} tries left`;
  $("#verdict-clue").hidden = false;
  $("#verdict-clue").textContent = out.clue || out.say;
  $("#reveal").hidden = true;
}

/** @param {any} out */
function showVerdict(out) {
  const v = $("#verdict");
  v.hidden = false;
  v.classList.remove("previous");
  v.dataset.v = out.verdict;
  $("#verdict-chip").textContent = VERDICT_LABEL[/** @type {keyof typeof VERDICT_LABEL} */ (out.verdict)] || out.verdict;
  $("#verdict-points").textContent = out.points ? `+${out.points}` : "+0";
  $("#verdict-clue").hidden = true;
  const reveal = out.reveal;
  $("#reveal").hidden = !reveal;
  if (reveal) {
    $("#reveal-answer").textContent = reveal.answer;
    const list = $("#reveal-points");
    list.innerHTML = "";
    reveal.key_points.forEach((/** @type {string} */ p, /** @type {number} */ i) => {
      const li = document.createElement("li");
      li.textContent = p;
      if (reveal.points_hit.includes(i)) li.className = "hit";
      list.append(li);
    });
  }
}

let shownScore = 0;
/** @param {any} st @param {boolean} [instant] */
function updateHud(st, instant = false) {
  S.lastState = st;
  const target = st.score;
  const el = $("#hud-score");
  if (instant) {
    shownScore = target;
    el.textContent = String(target);
  } else {
    const from = shownScore;
    const t0 = performance.now();
    const tick = (/** @type {number} */ now) => {
      const k = Math.min(1, (now - t0) / 700);
      shownScore = Math.round(from + (target - from) * (1 - Math.pow(1 - k, 3)));
      el.textContent = String(shownScore);
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  $("#hud-streak-n").textContent = String(st.streak);
  $("#hud-mult").textContent = `×${(st.multiplier || 1).toFixed(1)}`;
  $("#hud-streak").dataset.level = String(st.streak >= 5 ? 3 : st.streak >= 3 ? 2 : st.streak >= 1 ? 1 : 0);
  renderDots();
}

/** @param {string} big @param {string} small @param {boolean} good */
function pop(big, small, good) {
  const el = document.createElement("div");
  el.className = `pop ${good ? "good" : ""}`;
  el.innerHTML = `${escapeHtml(big)}${small ? `<small>${escapeHtml(small)}</small>` : ""}`;
  $("#pop-layer").append(el);
  setTimeout(() => el.remove(), 1500);
}

// ── Results ──────────────────────────────────────────────────────────────

/** @param {any} summary */
async function showResults(summary) {
  if (!summary) return;
  setPhase("results");
  const game = S.game;
  $("#r-topic").textContent = `${S.pack.emoji} ${S.pack.title}${game.practice ? " · practice" : ""}`;
  $("#r-rank-icon").textContent = summary.rank.icon;
  $("#r-rank").textContent = summary.rank.title;
  countUp($("#r-score"), summary.score);
  $("#r-correct").textContent = `${summary.correct}/${summary.total}`;
  $("#r-streak").textContent = String(summary.best_streak);
  const s = summary.duration_s;
  $("#r-time").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  const badges = $("#r-badges");
  badges.innerHTML = "";
  summary.badges.forEach((/** @type {any} */ b, /** @type {number} */ i) => {
    const el = document.createElement("span");
    el.className = "badge";
    el.style.animationDelay = `${0.3 + i * 0.15}s`;
    el.textContent = `${b.icon} ${b.label}`;
    badges.append(el);
  });
  $("#btn-drill").hidden = !summary.missed.length;
  $("#r-placement").hidden = true;
  $("#r-report").innerHTML = '<div class="report-loading"><div class="ring small"></div> Writing your coaching notes…</div>';
  renderReview(summary.review);
  $("#r-board-topic").textContent = S.pack.title;
  if (summary.accuracy >= 0.6) {
    confetti({ count: 160 });
    S.avatar?.celebrate();
  }

  let report;
  try {
    report = await postJson(`api/game/${game.game_id}/report`, {});
  } catch (err) {
    $("#r-report").textContent = "Couldn't load the coaching notes.";
    return;
  }
  if (S.game !== game) return;
  renderReport(report.coaching);
  const pl = report.placement;
  if (pl) {
    $("#r-placement").hidden = false;
    $("#r-placement").textContent = `#${pl.topic_rank} on ${S.pack.title} · #${pl.global_rank} overall`;
  } else if (game.practice) {
    $("#r-placement").hidden = false;
    $("#r-placement").textContent = "Practice round: not counted on the leaderboard";
  }
  say(report.coaching.say, "wrapup");
  const rows = await fetchBoard(S.pack.slug, "all", 10);
  renderBoard($("#r-board"), rows, { me: game.player });
}

/** @param {any} c */
function renderReport(c) {
  const el = $("#r-report");
  const list = (/** @type {string[]} */ items) => `<ul>${items.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`;
  el.innerHTML = `<h4>📝 ${escapeHtml(c.headline || "Coaching notes")}</h4>${
    c.strengths?.length ? `<div class="label">You know</div>${list(c.strengths)}` : ""
  }${c.review?.length ? `<div class="label">Review next</div>${list(c.review)}` : ""}`;
}

/** @param {any[]} review */
function renderReview(review) {
  const ol = $("#r-review");
  ol.innerHTML = "";
  for (const r of review) {
    const li = document.createElement("li");
    li.className = r.verdict;
    const missed = r.key_points.filter((/** @type {string} */ _p, /** @type {number} */ i) => !r.points_hit.includes(i));
    li.innerHTML = `<span class="rp">${r.points ? `+${r.points}` : ""}</span><div class="rq">${escapeHtml(r.question)}</div>
      <div class="ra">${escapeHtml(r.answer)}</div>
      ${r.your_answer ? `<div class="ry">You: “${escapeHtml(r.your_answer)}”</div>` : ""}
      ${r.verdict !== "correct" && missed.length ? `<div class="ra">Missing: ${escapeHtml(missed.join("; "))}</div>` : ""}`;
    ol.append(li);
  }
}

/** @param {HTMLElement} el @param {number} target */
function countUp(el, target) {
  const t0 = performance.now();
  const tick = (/** @type {number} */ now) => {
    const k = Math.min(1, (now - t0) / 1400);
    el.textContent = String(Math.round(target * (1 - Math.pow(1 - k, 3))));
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ── Utils ────────────────────────────────────────────────────────────────

/** @param {string} url @param {any} body */
async function postJson(url, body) {
  const resp = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `Request failed (${resp.status})`);
  return data;
}

/** @param {string} s */
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] || c);
}

const sleep = (/** @type {number} */ ms) => new Promise((r) => setTimeout(r, ms));

let toastTimer = 0;
/** @param {string} msg @param {string} [kind] */
function toast(msg, kind = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => (el.hidden = true), 6000);
}

// Debug handle for the console and automated checks.
Object.assign(window, { viva: S, vivaSubmit: submitTurn });

boot();
