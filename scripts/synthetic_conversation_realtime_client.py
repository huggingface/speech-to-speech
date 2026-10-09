"""Synthetic realtime client(s): single or parallel, single-turn or multi-turn.

Opens one (or N) websocket connection(s) to /v1/realtime and cycles through
``--turns`` prompts per client at a fixed cadence (``--interval`` seconds
between turn starts). Each turn:

  1. Synthesize the prompt with macOS ``say`` (cached on disk after first run).
  2. Stream the prompt audio + trailing silence as input_audio_buffer.append.
  3. Wait for response.done.
  4. Sleep until --interval has elapsed since the turn started.

This single script subsumes two earlier ones:

* Parallel pool / capacity test:
    python scripts/synthetic_conversation_realtime_client.py --clients 3 --turns 1
  → 3 clients connect simultaneously, each sends one prompt. Surplus clients
    beyond pool size receive ``session_limit_reached`` and exit cleanly.

* Single-client soak / multi-turn:
    python scripts/synthetic_conversation_realtime_client.py --turns 60 --interval 10
  → 1 client, ~10 minute conversation, 60 sequential turns.

* Both at once (soak the pool):
    python scripts/synthetic_conversation_realtime_client.py --clients 2 --turns 60

* Soak a Hugging Face Inference Endpoint via its load balancer (10 min, 2 clients):
    export HF_TOKEN=hf_...
    python scripts/synthetic_conversation_realtime_client.py \
        --lb-url https://<your-lb>.us-east-1.aws.endpoints.huggingface.cloud \
        --clients 2 \
        --turns 60 \
        --interval 10 \
        --log-dir /tmp/hf_endpoint_soak

Each client uses a per-client prompt offset (coprime shift) so concurrent
clients send distinct prompts at each turn — making cross-session leaks
trivially detectable in the per-client transcript logs.

Outputs land under --log-dir:
  * prompts/prompt_NNN.wav        — shared cache of say-synthesized prompts
  * client_NNN/conversation.txt   — per-client transcript with timestamps
  * client_NNN/conversation.wav   — per-client assistant audio concatenated
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

logger = logging.getLogger("synthetic_client")

SAMPLE_RATE_HZ = 16000
CHUNK_MS = 20
BYTES_PER_SAMPLE = 2  # PCM16
CHUNK_BYTES = SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * CHUNK_MS // 1000  # 640
# Trailing silence after each prompt so server-side VAD detects speech_stopped
# and auto-commits. The local realtime service only supports server VAD.
TRAILING_SILENCE_MS = 1500
# Per-client prompt offset shift. 7 is coprime with len(PROMPTS)=60, so each
# client visits a unique permutation of the prompt list.
PROMPT_SHIFT_PER_CLIENT = 7

# 60 varied prompts — short enough to fit a 10-second turn budget, diverse
# enough that responses don't pattern-match. Cycled if --turns > len(PROMPTS).
PROMPTS: list[str] = [
    "What is the capital of France?",
    "Tell me a joke about robots.",
    "How does photosynthesis work?",
    "What is two plus two?",
    "Who painted the Mona Lisa?",
    "What is the largest ocean on Earth?",
    "Recommend a simple pasta recipe.",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "Tell me one fact about the planet Mars.",
    "What is the difference between weather and climate?",
    "How many continents are there?",
    "What is a haiku?",
    "What is the chemical formula for water?",
    "Who was the first president of the United States?",
    "What is gravity?",
    "Tell me a fun fact about dolphins.",
    "What is the tallest mountain on Earth?",
    "How do magnets work?",
    "What is the meaning of life in one sentence?",
    "Recommend a short book to read.",
    "What is the population of Tokyo, roughly?",
    "How does a refrigerator stay cold?",
    "Who invented the telephone?",
    "What is the smallest country in the world?",
    "Explain the theory of relativity simply.",
    "What is the difference between a virus and bacteria?",
    "Who wrote War and Peace?",
    "What is the deepest part of the ocean called?",
    "How does the human heart work?",
    "What is the most spoken language in the world?",
    "Tell me one fact about black holes.",
    "What is the longest river on Earth?",
    "How do you make a paper airplane?",
    "What is the chemical symbol for gold?",
    "Who composed the Fifth Symphony?",
    "What is the difference between an alligator and a crocodile?",
    "How many bones are in the human body?",
    "What is the smallest planet in our solar system?",
    "Suggest a beginner workout routine.",
    "What is the busiest airport in the world?",
    "How does Wi-Fi work in one sentence?",
    "What is the oldest known written language?",
    "Tell me about the Great Wall of China briefly.",
    "What is a neutron star?",
    "How does sound travel through space?",
    "What is the most popular sport globally?",
    "Who painted Starry Night?",
    "What happens when you mix baking soda and vinegar?",
    "How fast can a cheetah run?",
    "What is the difference between fiction and non-fiction?",
    "Why is the sky blue?",
    "What is the largest desert in the world?",
    "How many planets are in our solar system?",
    "Who discovered penicillin?",
    "What is a leap year?",
    "Tell me how rain forms.",
    "What is the longest-living animal?",
    "Say goodbye for now.",
]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def synthesize_with_say(text: str, out_path: Path) -> None:
    """Render *text* to a 16kHz mono PCM16 WAV using macOS ``say``."""
    subprocess.run(
        [
            "say",
            text,
            "-o",
            str(out_path),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
        ],
        check=True,
        capture_output=True,
    )


def load_pcm16_mono_16k(path: Path) -> bytes:
    """Load any WAV / AIFF and return 16kHz mono PCM16 little-endian bytes."""
    data, src_rate = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if src_rate != SAMPLE_RATE_HZ:
        mono = resample_poly(mono, SAMPLE_RATE_HZ, src_rate)
    mono = np.clip(mono, -1.0, 1.0)
    pcm16 = (mono * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def write_wav(path: Path, pcm16_bytes: bytes, rate: int = SAMPLE_RATE_HZ) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(BYTES_PER_SAMPLE)
        w.setframerate(rate)
        w.writeframes(pcm16_bytes)


# ---------------------------------------------------------------------------
# Per-turn ws flow
# ---------------------------------------------------------------------------


async def stream_prompt(ws, audio_pcm: bytes) -> None:
    """Send prompt chunks at real-time pacing, then trailing silence."""
    for i in range(0, len(audio_pcm), CHUNK_BYTES):
        chunk = audio_pcm[i : i + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
        await ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})
        )
        await asyncio.sleep(CHUNK_MS / 1000.0)
    silence_chunk = b"\x00" * CHUNK_BYTES
    silence_payload = base64.b64encode(silence_chunk).decode("ascii")
    for _ in range(TRAILING_SILENCE_MS // CHUNK_MS):
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": silence_payload}))
        await asyncio.sleep(CHUNK_MS / 1000.0)


async def consume_until_response_done(
    ws,
    response_audio_out: bytearray,
    response_timeout_s: float,
) -> dict:
    """Read server events until response.done (or timeout). Returns turn summary."""
    info: dict = {
        "transcript_in": "",
        "transcript_out": "",
        "error": None,
    }
    deadline = time.monotonic() + response_timeout_s
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            info["error"] = "response_timeout"
            return info
        event = json.loads(raw)
        t = event.get("type", "")

        if t == "conversation.item.input_audio_transcription.completed":
            info["transcript_in"] += event.get("transcript", "")
        elif t in ("response.audio.delta", "response.output_audio.delta"):
            delta = event.get("delta", "")
            if delta:
                response_audio_out.extend(base64.b64decode(delta))
        elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
            info["transcript_out"] += event.get("delta", "")
        elif t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            transcript = event.get("transcript")
            if transcript:
                info["transcript_out"] = transcript
        elif t == "response.done":
            return info
        elif t == "error":
            info["error"] = event.get("error", {}).get("message", "error event")
            return info
    info["error"] = "response_timeout"
    return info


def _truncate_transcript(s: str, max_len: int = 20) -> str:
    """Quote *s*; if longer than *max_len*, slice + ellipsis + total char count."""
    if len(s) <= max_len:
        return repr(s)
    return f"{(s[:max_len] + '...')!r} ({len(s)} chars)"


# ---------------------------------------------------------------------------
# Client driver (one per client)
# ---------------------------------------------------------------------------


async def _lb_allocate_session(
    lb_url: str,
    auth_headers: dict[str, str],
    http: httpx.AsyncClient,
) -> dict:
    """POST {lb_url}/session → returns dict with connect_url, session_id, session_token."""
    resp = await http.post(f"{lb_url}/session", headers=auth_headers, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


async def _lb_send_event(
    lb_url: str,
    session_id: str,
    session_token: str,
    event: str,
    http: httpx.AsyncClient,
) -> None:
    """POST {lb_url}/internal/sessions/{id}/event with {session_token, event}."""
    url = f"{lb_url}/internal/sessions/{session_id}/event"
    resp = await http.post(url, json={"session_token": session_token, "event": event}, timeout=30.0)
    resp.raise_for_status()


async def run_client(
    client_id: int,
    args: argparse.Namespace,
    ws_url: str,
    extra_headers: list[tuple[str, str]],
    audio_files: list[tuple[str, Path]],
) -> dict:
    """Run one client's multi-turn conversation. Returns per-client summary.

    If --lb-url is set, hits the load balancer first to allocate a slot on a
    compute node, then connects to the returned connect_url. Otherwise connects
    directly to ws_url. The LB callback events ("connected"/"disconnected") are
    sent around the websocket session so the LB doesn't reclaim the slot early.
    """
    summary: dict = {
        "client_id": client_id,
        "connected": False,
        "rejected": False,
        "completed": 0,
        "errors": 0,
        "error_msg": None,
    }

    log_dir = Path(args.log_dir) / f"client_{client_id:03d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript_log = log_dir / "conversation.txt"
    response_audio = bytearray()

    prefix = f"[c{client_id}]"
    auth_headers = {k: v for k, v in extra_headers}
    lb_session_id: str | None = None
    lb_session_token: str | None = None

    async with httpx.AsyncClient() as http:
        # Step 1: allocate a slot via the LB (if configured).
        if args.lb_url:
            try:
                alloc = await _lb_allocate_session(args.lb_url, auth_headers, http)
            except Exception as e:  # noqa: BLE001 — diagnostic path
                summary["error_msg"] = f"LB /session failed: {type(e).__name__}: {e}"
                logger.info(f"{prefix} {summary['error_msg']}")
                return summary
            connect_url = alloc["connect_url"]
            lb_session_id = alloc["session_id"]
            lb_session_token = alloc["session_token"]
            logger.info(
                f"{prefix} LB allocated session_id={lb_session_id} "
                f"ws_url={alloc.get('websocket_url', '?')}"
            )
        else:
            connect_url = ws_url

        try:
            async with websockets.connect(
                connect_url,
                max_size=2**24,
                additional_headers=extra_headers or None,
            ) as ws:
                summary["connected"] = True

                # Step 3: tell the LB the client actually connected (clears the
                # pending_timeout_s reaper). Best-effort — failure here logs but
                # doesn't abort the session.
                if lb_session_id and lb_session_token:
                    try:
                        await _lb_send_event(args.lb_url, lb_session_id, lb_session_token, "connected", http)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"{prefix} LB 'connected' callback failed: {e}")

                first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
                if first.get("type") == "error" and "session_limit_reached" in str(first):
                    summary["rejected"] = True
                    summary["error_msg"] = first.get("error", {}).get("message", "limit reached")
                    logger.warning(f"{prefix} REJECTED: {summary['error_msg']}")
                elif first.get("type") != "session.created":
                    summary["error_msg"] = f"unexpected first event: {first.get('type')}"
                    logger.error(f"{prefix} ERROR: {summary['error_msg']}")
                else:
                    sess = first.get("session") or {}
                    session_id = sess.get("id") or first.get("event_id") or "?"
                    logger.info(f"{prefix} connected, session={session_id}")

                    with transcript_log.open("w") as log_f:
                        log_f.write(f"# Synthetic conversation, client={client_id}, session={session_id}\n")
                        log_f.write(f"# Target turns={args.turns}, interval={args.interval}s\n\n")

                        for turn_idx in range(args.turns):
                            # Per-client prompt offset (coprime shift) so concurrent
                            # clients send distinct prompts at every turn.
                            prompt_idx = (turn_idx + client_id * PROMPT_SHIFT_PER_CLIENT) % len(audio_files)
                            text, wav_path = audio_files[prompt_idx]

                            turn_start = time.monotonic()
                            turn_audio = load_pcm16_mono_16k(wav_path)

                            logger.info(f"{prefix} turn {turn_idx + 1}/{args.turns} USER: {text!r}")
                            log_f.write(f"[turn {turn_idx + 1}/{args.turns}] USER: {text}\n")

                            try:
                                await stream_prompt(ws, turn_audio)
                            except websockets.exceptions.ConnectionClosed as e:
                                logger.info(
                                    f"{prefix} turn {turn_idx + 1}/{args.turns} "
                                    f"connection closed during send: {e}"
                                )
                                log_f.write(f"[turn {turn_idx + 1}/{args.turns}] CONNECTION_CLOSED: {e}\n")
                                summary["error_msg"] = f"connection_closed: {e}"
                                break

                            info = await consume_until_response_done(
                                ws, response_audio, args.response_timeout
                            )
                            turn_elapsed = time.monotonic() - turn_start

                            if info["error"]:
                                logger.info(
                                    f"{prefix} turn {turn_idx + 1}/{args.turns} ERROR: {info['error']}"
                                )
                                log_f.write(f"[turn {turn_idx + 1}/{args.turns}] ERROR: {info['error']}\n\n")
                                summary["errors"] += 1
                            else:
                                summary["completed"] += 1
                                logger.info(
                                    f"{prefix} turn {turn_idx + 1}/{args.turns} "
                                    f"ASSISTANT: {_truncate_transcript(info['transcript_out'])}"
                                )
                                log_f.write(f"[turn {turn_idx + 1}/{args.turns}] STT: {info['transcript_in']}\n")
                                log_f.write(
                                    f"[turn {turn_idx + 1}/{args.turns}] ASSISTANT: {info['transcript_out']}\n\n"
                                )
                            log_f.flush()

                            remaining = args.interval - turn_elapsed
                            if remaining > 0:
                                await asyncio.sleep(remaining)

        except Exception as e:  # noqa: BLE001 — diagnostic path
            summary["error_msg"] = f"{type(e).__name__}: {e}"
            logger.error(f"{prefix} EXCEPTION: {summary['error_msg']}")
        finally:
            # Step 5: release the LB slot, regardless of how the ws session ended.
            if lb_session_id and lb_session_token:
                try:
                    await _lb_send_event(
                        args.lb_url, lb_session_id, lb_session_token, "disconnected", http
                    )
                    logger.info(f"{prefix} LB notified: disconnected")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"{prefix} LB 'disconnected' callback failed: {e}")

    if response_audio:
        wav_out = log_dir / "conversation.wav"
        write_wav(wav_out, bytes(response_audio))
        logger.info(f"{prefix} wrote {len(response_audio)} bytes -> {wav_out}")
    return summary


# ---------------------------------------------------------------------------
# Top-level: synthesize prompts, spawn clients, summarize
# ---------------------------------------------------------------------------


async def run_all(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = log_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Synthesizing {len(PROMPTS)} prompts with `say` (cached in {prompts_dir})...")
    audio_files: list[tuple[str, Path]] = []
    for i, text in enumerate(PROMPTS):
        wav_path = prompts_dir / f"prompt_{i:03d}.wav"
        if not wav_path.exists():
            synthesize_with_say(text, wav_path)
        audio_files.append((text, wav_path))

    ws_url = args.url or f"ws://{args.host}:{args.port}/v1/realtime"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN env var is not set. Export it before running this script "
            "(e.g. `export HF_TOKEN=hf_...`)."
        )
    extra_headers: list[tuple[str, str]] = [("Authorization", f"Bearer {token}")]
    logger.info("Auth: Bearer token attached from HF_TOKEN env")
    if args.lb_url:
        target = f"LB {args.lb_url}"
    else:
        target = ws_url
    logger.info(
        f"Spawning {args.clients} client(s) against {target}, "
        f"{args.turns} turns each @ {args.interval:.1f}s interval"
    )

    summaries = await asyncio.gather(
        *(
            run_client(
                client_id=i,
                args=args,
                ws_url=ws_url,
                extra_headers=extra_headers,
                audio_files=audio_files,
            )
            for i in range(args.clients)
        )
    )

    logger.info("\n=== summary ===")
    for s in summaries:
        status = "rejected" if s["rejected"] else ("error" if s["error_msg"] else "ok")
        logger.info(
            f"  c{s['client_id']}: {status:8s} completed={s['completed']}/{args.turns} "
            f"errors={s['errors']} err={s['error_msg']}"
        )
    n_ok = sum(1 for s in summaries if s["connected"] and not s["rejected"] and not s["error_msg"])
    n_rej = sum(1 for s in summaries if s["rejected"])
    n_err = sum(1 for s in summaries if s["error_msg"] and not s["rejected"])
    total_turns = sum(s["completed"] for s in summaries)
    logger.info(f"=> {n_ok} successful clients, {n_rej} rejected, {n_err} errored")
    logger.info(f"=> {total_turns} total turns completed across pool")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if not shutil.which("say"):
        raise SystemExit("This script requires macOS `say` (not found in PATH).")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lb-url",
        default=None,
        help=(
            "Base URL of the load balancer (e.g. https://lb.example.com). When set, the script "
            "POSTs /session to allocate a slot on a compute node, connects to the returned "
            "connect_url, and sends connected/disconnected events around the session. Takes "
            "precedence over --url / --host / --port."
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help=(
            "Full ws:// or wss:// URL to a compute-node realtime endpoint, e.g. "
            "wss://endpoint.example.com/v1/realtime. Used when --lb-url is not set. "
            "Overrides --host/--port."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Used only when --url and --lb-url are unset.")
    parser.add_argument("--port", type=int, default=8765, help="Used only when --url and --lb-url are unset.")
    parser.add_argument("--clients", type=int, default=1, help="Number of parallel clients (default 1).")
    parser.add_argument("--turns", type=int, default=60, help="Turns per client (default 60).")
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between turn starts per client. Total runtime ≈ turns × interval. Default 10.",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=30.0,
        help="Per-turn wait for response.done before marking the turn errored.",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/synthetic_conversation",
        help="Top-level directory for prompt cache and per-client logs.",
    )
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients must be >= 1")
    if args.turns < 1:
        parser.error("--turns must be >= 1")
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
