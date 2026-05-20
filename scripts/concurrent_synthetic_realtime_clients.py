"""Drive N concurrent synthetic websocket clients against the realtime pool.

Reads a WAV file, opens N websockets to /v1/realtime in parallel, streams the audio
in real-time pacing as input_audio_buffer.append events, then collects the response
events. Use this to exercise the pool's per-pipeline isolation without needing N mics.

Example:
    # Server with a 2-slot pool:
    python -m speech_to_speech.s2s_pipeline --mode realtime --num_servers 2

    # Drive 2 concurrent synthetic clients:
    python scripts/concurrent_synthetic_clients.py --audio my_prompt.wav --n 2

    # Try N+1 to confirm the third gets rejected with session_limit_reached:
    python scripts/concurrent_synthetic_clients.py --audio my_prompt.wav --n 3

Per-client response audio is written to <log-dir>/client_<i>.wav.
"""
import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

SAMPLE_RATE_HZ = 16000
CHUNK_MS = 20
BYTES_PER_SAMPLE = 2  # PCM16
CHUNK_BYTES = SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * CHUNK_MS // 1000  # 640
# Trailing silence so server-side VAD can detect speech_stopped and auto-commit
# the buffer. The local realtime service only supports server VAD — no client commit.
TRAILING_SILENCE_MS = 1500


def load_pcm16_mono_16k(path: Path) -> bytes:
    """Load a WAV file and return 16kHz mono PCM16 little-endian bytes."""
    data, src_rate = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if src_rate != SAMPLE_RATE_HZ:
        # resample_poly with up=target, down=src gives target_rate
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


async def run_client(
    client_id: int,
    ws_url: str,
    audio_pcm: bytes,
    log_dir: Path,
    response_timeout_s: float,
) -> dict:
    """Drive one synthetic session: stream audio, collect response, log result."""
    result: dict = {
        "client_id": client_id,
        "connected": False,
        "rejected": False,
        "session_id": None,
        "transcript_out": "",
        "transcript_in": "",
        "audio_bytes_out": 0,
        "audio_bytes_in": 0,
        "error": None,
        "elapsed_s": 0.0,
    }
    started = time.monotonic()
    received_audio = bytearray()

    try:
        async with websockets.connect(ws_url, max_size=2**24) as ws:
            result["connected"] = True

            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            if first.get("type") == "error" and "session_limit_reached" in str(first):
                result["rejected"] = True
                result["error"] = first.get("error", {}).get("message", "limit reached")
                print(f"[client {client_id}] REJECTED: {result['error']}")
                return result

            if first.get("type") != "session.created":
                result["error"] = f"unexpected first event: {first.get('type')}"
                return result
            sess = first.get("session") or {}
            result["session_id"] = sess.get("id") or first.get("session_id") or first.get("event_id")
            if result["session_id"] is None:
                print(f"[client {client_id}] DEBUG session.created event keys: {list(first.keys())}")
            print(f"[client {client_id}] connected, session={result['session_id']}")

            silence_chunk = b"\x00" * CHUNK_BYTES
            n_silence = TRAILING_SILENCE_MS // CHUNK_MS

            async def send_audio() -> None:
                for i in range(0, len(audio_pcm), CHUNK_BYTES):
                    chunk = audio_pcm[i : i + CHUNK_BYTES]
                    if len(chunk) < CHUNK_BYTES:
                        chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))
                    payload = base64.b64encode(chunk).decode("ascii")
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))
                    result["audio_bytes_out"] += len(chunk)
                    await asyncio.sleep(CHUNK_MS / 1000.0)
                # Trailing silence so server VAD detects end-of-speech and auto-commits.
                silence_payload = base64.b64encode(silence_chunk).decode("ascii")
                for _ in range(n_silence):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": silence_payload}))
                    result["audio_bytes_out"] += CHUNK_BYTES
                    await asyncio.sleep(CHUNK_MS / 1000.0)
                print(f"[client {client_id}] audio sent ({TRAILING_SILENCE_MS}ms trailing silence), awaiting response")

            async def recv_events() -> None:
                deadline = time.monotonic() + response_timeout_s
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
                    except asyncio.TimeoutError:
                        break
                    event = json.loads(raw)
                    t = event.get("type", "")
                    if t == "response.audio.delta" or t == "response.output_audio.delta":
                        delta = event.get("delta", "")
                        if delta:
                            chunk = base64.b64decode(delta)
                            received_audio.extend(chunk)
                            result["audio_bytes_in"] += len(chunk)
                    elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                        result["transcript_out"] += event.get("delta", "")
                    elif t == "conversation.item.input_audio_transcription.completed":
                        result["transcript_in"] += event.get("transcript", "")
                    elif t == "response.done":
                        print(f"[client {client_id}] response.done")
                        return
                    elif t == "error":
                        result["error"] = event.get("error", {}).get("message", "error event")
                        print(f"[client {client_id}] error: {result['error']}")
                        return

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(recv_events())
            await asyncio.gather(send_task, recv_task)

    except Exception as e:  # noqa: BLE001 — diagnostic path
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[client {client_id}] EXCEPTION: {result['error']}")

    result["elapsed_s"] = time.monotonic() - started
    if received_audio:
        wav_path = log_dir / f"client_{client_id}.wav"
        write_wav(wav_path, bytes(received_audio))
        print(f"[client {client_id}] wrote {len(received_audio)} bytes -> {wav_path}")
    return result


async def main_async(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    audio_pcm = load_pcm16_mono_16k(Path(args.audio))
    duration_s = len(audio_pcm) / (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE)
    print(f"loaded {args.audio}: {duration_s:.2f}s @ 16kHz mono PCM16 ({len(audio_pcm)} bytes)")

    ws_url = f"ws://{args.host}:{args.port}/v1/realtime"
    print(f"spawning {args.n} concurrent clients against {ws_url}")

    tasks = [
        run_client(
            client_id=i,
            ws_url=ws_url,
            audio_pcm=audio_pcm,
            log_dir=log_dir,
            response_timeout_s=args.response_timeout,
        )
        for i in range(args.n)
    ]
    results = await asyncio.gather(*tasks)

    print("\n=== summary ===")
    for r in results:
        status = "rejected" if r["rejected"] else ("error" if r["error"] else "ok")
        print(
            f"  client {r['client_id']}: {status:8s} "
            f"out={r['audio_bytes_out']}B in={r['audio_bytes_in']}B "
            f"transcript_in={r['transcript_in']!r} transcript_out={r['transcript_out']!r} "
            f"elapsed={r['elapsed_s']:.2f}s "
            f"err={r['error']}"
        )
    n_ok = sum(1 for r in results if r["connected"] and not r["rejected"] and not r["error"])
    n_rej = sum(1 for r in results if r["rejected"])
    print(f"=> {n_ok} successful, {n_rej} rejected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--audio", required=True, help="Path to a WAV file to use as the user's prompt.")
    parser.add_argument("--n", type=int, default=2, help="Number of concurrent clients.")
    parser.add_argument(
        "--log-dir",
        default="/tmp/synthetic_clients",
        help="Directory to write per-client response WAVs.",
    )
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=30.0,
        help="How long to wait for response.done per client.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
