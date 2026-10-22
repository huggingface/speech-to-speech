"""Scripted spoken conversation with a speech-to-speech server backed by ``transformers serve``.

No microphone is needed: each user turn is synthesized by Gradium Phonon through
``transformers serve``'s ``/v1/audio/speech`` endpoint in a catalog voice, then
streamed to the speech-to-speech Realtime WebSocket at real-time pace. The
assistant's replies are recorded with their transcripts and client-side timings.

    python examples/transformers-serve/conversation_demo.py --out-dir demo-out

Outputs in ``--out-dir``:

* ``conversation.wav``: user and assistant turns, in order, at 16 kHz
* ``conversation.md``: what was said, what STT heard, and what the assistant replied
* ``turns.json``: the same, with per-turn timings
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000
TRAILING_SILENCE_MS = 1500
GAP_MS = 400

PROMPTS = [
    "Hi there! Who am I talking to?",
    "What is the capital of France?",
    "Please remember that I take my tea without sugar.",
    "How do I like my tea?",
    "Give me one quick tip for staying focused while working.",
]


def synthesize_user_turn(http: httpx.Client, args: argparse.Namespace, text: str, voice_path: str) -> bytes:
    """Phonon speech for *text* in the user's voice, as 16 kHz PCM16."""
    response = http.post(
        f"{args.transformers_url}/audio/speech",
        json={"model": args.tts_model, "input": text, "ref_audio": voice_path},
    )
    response.raise_for_status()
    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    audio = resample_poly(audio, SAMPLE_RATE, rate)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


async def send_audio(ws, pcm: bytes, marks: dict[str, float]) -> None:
    """Send *pcm* at real-time pace, followed by silence so server VAD ends the turn."""
    silence = b"\x00" * (SAMPLE_RATE * 2 * TRAILING_SILENCE_MS // 1000)
    for audio in (pcm, silence):
        for i in range(0, len(audio), CHUNK_BYTES):
            chunk = audio[i : i + CHUNK_BYTES].ljust(CHUNK_BYTES, b"\x00")
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()}))
            await asyncio.sleep(CHUNK_MS / 1000)
        # Timings are measured from the end of the user's audio, before the trailing silence
        marks.setdefault("speech_end", time.monotonic())


async def run_turn(ws, text: str, pcm: bytes, timeout_s: float) -> dict:
    turn = {"user": text, "heard": "", "assistant": "", "assistant_pcm": bytearray()}
    marks: dict[str, float] = {}
    sender = asyncio.create_task(send_audio(ws, pcm, marks))
    try:
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
            kind = event.get("type", "")
            now = time.monotonic()
            if kind == "input_audio_buffer.speech_stopped":
                marks.setdefault("vad_stop", now)
            elif kind == "conversation.item.input_audio_transcription.completed":
                marks.setdefault("transcript", now)
                turn["heard"] = event.get("transcript", "")
            elif kind in ("response.output_audio.delta", "response.audio.delta"):
                marks.setdefault("first_audio", now)
                turn["assistant_pcm"].extend(base64.b64decode(event.get("delta", "")))
            elif kind in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
                turn["assistant"] = event.get("transcript", "") or turn["assistant"]
            elif kind in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                turn["assistant"] += event.get("delta", "")
            elif kind == "response.done":
                marks["done"] = now
                break
            elif kind == "error":
                raise RuntimeError(event.get("error", {}).get("message", "error event"))
    finally:
        await sender
    speech_end = marks.pop("speech_end")
    turn["timings_s"] = {name: round(mark - speech_end, 3) for name, mark in marks.items()}
    turn["timings_s"]["user_audio"] = round(len(pcm) / (SAMPLE_RATE * 2), 3)
    turn["timings_s"]["assistant_audio"] = round(len(turn["assistant_pcm"]) / (SAMPLE_RATE * 2), 3)
    return turn


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)


async def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Phonon voices are latents, not recordings: `transformers serve` reads them from a path on its own machine
    voice_path = hf_hub_download("gradium/phonon-voices", f"voices/{args.user_voice}.safetensors")
    with httpx.Client(timeout=120) as http:
        user_audio = [synthesize_user_turn(http, args, text, voice_path) for text in PROMPTS[: args.turns]]

    turns = []
    async with websockets.connect(args.realtime_url, max_size=2**24) as ws:
        created = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if created.get("type") != "session.created":
            raise RuntimeError(f"unexpected first event: {created}")
        if args.assistant_voice:
            # The Realtime voice reaches `/v1/audio/speech` as its `voice`
            session = {"type": "realtime", "audio": {"output": {"voice": args.assistant_voice}}}
            await ws.send(json.dumps({"type": "session.update", "session": session}))
            while (event := json.loads(await asyncio.wait_for(ws.recv(), timeout=30)))["type"] != "session.updated":
                if event["type"] == "error":
                    raise RuntimeError(event.get("error", {}).get("message", "session.update failed"))
        for text, pcm in zip(PROMPTS, user_audio):
            print(f"USER       {text}", flush=True)
            turn = await run_turn(ws, text, pcm, args.response_timeout)
            timings = turn["timings_s"]
            print(f"  STT      {turn['heard'].strip()}", flush=True)
            print(f"  ASSIST   {turn['assistant'].strip()}", flush=True)
            print(f"  timings  {timings}", flush=True)
            turns.append((pcm, turn))

    gap = b"\x00" * (SAMPLE_RATE * 2 * GAP_MS // 1000)
    write_wav(out_dir / "conversation.wav", b"".join(pcm + gap + bytes(t["assistant_pcm"]) + gap for pcm, t in turns))
    records = [{key: value for key, value in t.items() if key != "assistant_pcm"} for _, t in turns]
    (out_dir / "turns.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# Conversation", ""]
    for i, record in enumerate(records, 1):
        lines += [
            f"**Turn {i}**",
            "",
            f"- User said: {record['user']}",
            f"- STT heard: {record['heard'].strip()}",
            f"- Assistant: {record['assistant'].strip()}",
            f"- First assistant audio {record['timings_s'].get('first_audio', 'n/a')} s after the user stopped",
            "",
        ]
    (out_dir / "conversation.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'conversation.wav'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--realtime-url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument("--transformers-url", default="http://localhost:8000/v1")
    parser.add_argument("--tts-model", default="gradium/phonon-31m-en")
    # Zoey, from gradium/phonon-voices
    parser.add_argument("--user-voice", default="NbpkqMVS3CJeq2j8")
    parser.add_argument(
        "--assistant-voice",
        help="Phonon catalog voice for the assistant, such as Harper. Without it, the server's voice is used.",
    )
    parser.add_argument("--turns", type=int, default=len(PROMPTS))
    parser.add_argument("--response-timeout", type=float, default=60.0)
    parser.add_argument("--out-dir", default="transformers-serve-demo")
    asyncio.run(main(parser.parse_args()))
