"""Synthetic PCM through the real VAD/STT/ConvFill/TTS/WebSocket pipeline.

No browser playback is implied. Use a generated test WAV, never a private
recording. Latencies use concurrent receive timestamps, not a delayed drain.
"""

import argparse
import asyncio
import base64
import json
import os
import time
import wave
from pathlib import Path

import websockets


async def main(args):
    with wave.open(args.audio, "rb") as audio:
        assert audio.getframerate() == 24000 and audio.getsampwidth() == 2 and audio.getnchannels() == 1
        pcm = audio.readframes(audio.getnframes())
    headers = {"Authorization": "Bearer " + os.environ[args.auth_env]} if args.auth_env else {}
    results = {"actual_models": True, "browser_playback": False, "turns": []}
    for session in range(2):
        async with websockets.connect(
            args.url, additional_headers=headers, max_size=16 * 1024 * 1024, open_timeout=20, ping_interval=None
        ) as ws:
            assert json.loads(await ws.recv())["type"] == "session.created"
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24000},
                                    "turn_detection": {
                                        "type": "server_vad",
                                        "create_response": True,
                                        "interrupt_response": True,
                                    },
                                },
                                "output": {"voice": "Vivian"},
                            },
                        },
                    }
                )
            )
            while json.loads(await ws.recv())["type"] != "session.updated":
                pass
            for turn in range(2 if session == 0 else 1):
                record = {"session": session, "turn": turn, "audio_bytes": 0}

                async def receive():
                    while True:
                        event = json.loads(await asyncio.wait_for(ws.recv(), 60))
                        now = time.perf_counter()
                        kind = event.get("type", "")
                        if kind == "input_audio_buffer.speech_stopped":
                            record.setdefault("vad_stopped", now)
                        if kind == "conversation.item.input_audio_transcription.completed":
                            record["stt_completed"] = True
                        if kind == "response.output_audio.delta":
                            record.setdefault("first_audio", now)
                            record["audio_bytes"] += len(base64.b64decode(event["delta"]))
                        if kind == "response.output_audio_transcript.delta":
                            assert "<" not in event.get("delta", ""), "control token reached spoken transcript"
                        if kind == "error":
                            raise RuntimeError(
                                "Realtime error: " + str(event.get("error", {}).get("code") or "unknown")
                            )
                        if kind == "response.done":
                            record["status"] = event["response"]["status"]
                            assert record["status"] == "completed", record["status"]
                            return

                reader = asyncio.create_task(receive())
                signal = bytes(9600) + pcm + bytes(48000)
                for offset in range(0, len(signal), 4800):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(signal[offset : offset + 4800]).decode(),
                            }
                        )
                    )
                    await asyncio.sleep(0.1)
                await reader
                assert record.get("stt_completed") and record["audio_bytes"] > 0
                record["vad_event_to_first_audio_ms"] = round(
                    (record.pop("first_audio") - record.pop("vad_stopped")) * 1000, 1
                )
                results["turns"].append(record)
                Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(record), flush=True)
                # Allow dispatched playback to drain before the next test turn.
                await asyncio.sleep(min(record["audio_bytes"] / 48000 + 0.5, 20))
        await asyncio.sleep(2)
    results["reconnected"] = True
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:17860/v1/realtime")
    parser.add_argument("--auth-env")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="/tmp/convfill-realtime-validation.json")
    asyncio.run(main(parser.parse_args()))
