"""Barge in with synthetic microphone PCM against a real speech pipeline.

This checks transport cancellation, not what a human heard in a browser.
"""

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import websockets


async def run(args):
    with wave.open(args.audio, "rb") as audio:
        assert audio.getframerate() == 24000
        pcm = audio.readframes(audio.getnframes())
    records = []
    for case in ["first_audio", "substantive_text"]:
        record = {"case": case, "old_audio_after_barge_confirmation": 0}
        async with websockets.connect(args.url, ping_interval=None, max_size=8 * 1024 * 1024) as ws:
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
                                        "create_response": False,
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
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Explain why the sky is blue in three short sentences."}
                            ],
                        },
                    }
                )
            )
            await ws.send(json.dumps({"type": "response.create"}))
            sent = None
            confirmed = None
            sender = None
            transcript = ""
            first_audio = False

            async def interrupt():
                signal = pcm + bytes(48000)
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

            try:
                end = time.monotonic() + 45
                while time.monotonic() < end:
                    event = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    kind = event.get("type")
                    if kind == "response.output_audio_transcript.delta":
                        transcript += event.get("delta", "")
                    if kind == "response.output_audio.delta":
                        first_audio = True
                        if confirmed:
                            record["old_audio_after_barge_confirmation"] += 1
                    if sent is None and first_audio and (case == "first_audio" or "blue" in transcript.lower()):
                        sent = time.perf_counter()
                        sender = asyncio.create_task(interrupt())
                    if kind == "input_audio_buffer.speech_started" and sent:
                        confirmed = time.perf_counter()
                        record["audio_send_to_vad_barge_ms"] = round((confirmed - sent) * 1000, 1)
                    if kind == "error":
                        raise RuntimeError("Realtime error during barge-in")
                    if kind == "response.done":
                        record["response_status"] = event["response"]["status"]
                        assert sent, "Response ended before synthetic barge-in started"
                        assert record["response_status"] == "cancelled", record
                        record["_response_done_time"] = time.perf_counter()
                    # Upstream may emit response.done before speech_started.
                    if confirmed and "_response_done_time" in record:
                        record["barge_confirmation_to_response_done_ms"] = round(
                            (record.pop("_response_done_time") - confirmed) * 1000, 1
                        )
                        break
                else:
                    raise TimeoutError("Barge-in did not complete")
            finally:
                if sender:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
        records.append(record)
        Path(args.output).write_text(
            json.dumps({"actual_models": True, "browser_playback": False, "tests": records}, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
        await asyncio.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:17860/v1/realtime")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="/tmp/convfill-bargein-validation.json")
    asyncio.run(run(parser.parse_args()))
