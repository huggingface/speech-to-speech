"""Measure consecutive automatic turns on one native Realtime connection.

Use the same short PCM16/16kHz WAV for every run when comparing deployments.
No audio commit or response.create messages are sent. With --wait-models, report
connection preparation separately and begin speech only after model readiness.
"""

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import websockets
from huggingface_hub import get_token


async def benchmark(args):
    with wave.open(args.audio) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected mono PCM16 WAV at 16000 Hz")
        pcm = wav.readframes(wav.getnframes())
    if not pcm:
        raise ValueError("Audio must not be empty")
    duration = len(pcm) / 32000
    results = []
    headers = {"Authorization": f"Bearer {get_token()}"}
    connected_at = time.monotonic()
    async with websockets.connect(args.url, additional_headers=headers, max_size=2**24) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": "Answer in English in exactly one short sentence.",
                        "audio": {
                            "input": {
                                "transcription": {"language": "en"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "create_response": True,
                                    "interrupt_response": True,
                                },
                            }
                        },
                    },
                }
            )
        )
        session_updated = False
        models_ready = not args.wait_models
        while not (session_updated and models_ready):
            event = json.loads(await asyncio.wait_for(ws.recv(), args.timeout))
            if event["type"] in ("error", "speech_to_speech.models.error"):
                raise RuntimeError(event["error"])
            if event["type"] == "session.updated":
                session_updated = True
            if event["type"] == "speech_to_speech.models.ready":
                models_ready = True
        preparation_seconds = time.monotonic() - connected_at

        for index in range(args.turns):
            started = time.monotonic()
            record = {
                "turn": index + 1,
                "audio_seconds": duration,
                "audio_chunks": 0,
                "connection_prepare_seconds": preparation_seconds,
            }

            async def upload():
                # Pace audio from one clock, avoiding accumulated sleep drift.
                data = pcm + bytes(3 * 32000)
                for offset in range(0, len(data), 1024):
                    delay = started + offset / 32000 - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(data[offset : offset + 1024]).decode(),
                            }
                        )
                    )

            sender = asyncio.create_task(upload())
            try:
                async with asyncio.timeout(args.timeout):
                    while True:
                        event = json.loads(await ws.recv())
                        elapsed = time.monotonic() - started
                        kind = event["type"]
                        if kind == "error":
                            record.setdefault("errors", []).append(event["error"])
                        elif kind == "input_audio_buffer.speech_stopped":
                            record["speech_stopped_received_s"] = elapsed
                        elif kind == "conversation.item.input_audio_transcription.completed":
                            record["transcript_received_s"] = elapsed
                            record["transcript"] = event["transcript"]
                        elif kind == "response.output_audio.delta":
                            record.setdefault("first_audio_received_s", elapsed)
                            record.setdefault("e2e_client_estimate_s", elapsed - duration)
                            record["audio_chunks"] += 1
                        elif kind == "response.done":
                            response = event["response"]
                            record["status"] = response["status"]
                            record["done_received_s"] = elapsed
                            record["response"] = response["output"]
                            metadata = response.get("metadata") or {}
                            raw = metadata.get("speech_to_speech.turn_latency")
                            record["server_timings"] = json.loads(raw) if raw else None
                            break
            finally:
                await sender
                results.append(record)
                Path(args.output).write_text(json.dumps(results, indent=2))
            print(json.dumps(record), flush=True)
            if record.get("status") != "completed" or not record["audio_chunks"]:
                raise RuntimeError("Automatic spoken turn did not complete successfully")
            await asyncio.sleep(args.gap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("audio")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--gap", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--wait-models", action="store_true")
    parser.add_argument("--output", default="/tmp/zerogpu-latency-benchmark.json")
    asyncio.run(benchmark(parser.parse_args()))
