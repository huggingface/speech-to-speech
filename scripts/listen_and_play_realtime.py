"""Standalone entry point for the packaged Realtime microphone/speaker client."""

import argparse
import asyncio

from speech_to_speech.api.openai_realtime.local_client import (
    RealtimeAudioClientConfig,
    listen_and_play_realtime,
)


def main() -> None:
    defaults = RealtimeAudioClientConfig()
    parser = argparse.ArgumentParser(description="Talk to an OpenAI-compatible Realtime speech pipeline.")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--api-key", default=defaults.api_key)
    parser.add_argument("--base-url", default=defaults.base_url)
    parser.add_argument("--websocket-base-url", default=defaults.websocket_base_url)
    parser.add_argument("--send-rate", type=int, default=defaults.send_rate)
    parser.add_argument("--recv-rate", type=int, default=defaults.recv_rate)
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--input-device", type=int, default=defaults.input_device)
    parser.add_argument("--output-device", type=int, default=defaults.output_device)
    parser.add_argument("--instructions", default=defaults.instructions)
    parser.add_argument(
        "--voice",
        default=defaults.voice,
        help="session.audio.output.voice (for example bm_fable, marin, or alloy).",
    )
    parser.add_argument("--print-json", action="store_true", default=defaults.print_json)
    parser.add_argument(
        "--block-mic-during-playback",
        action="store_true",
        default=defaults.block_mic_during_playback,
    )
    namespace = parser.parse_args()
    config = RealtimeAudioClientConfig(
        host=namespace.host,
        port=namespace.port,
        model=namespace.model,
        api_key=namespace.api_key,
        base_url=namespace.base_url,
        websocket_base_url=namespace.websocket_base_url,
        send_rate=namespace.send_rate,
        recv_rate=namespace.recv_rate,
        chunk_size=namespace.chunk_size,
        input_device=namespace.input_device,
        output_device=namespace.output_device,
        instructions=namespace.instructions,
        voice=namespace.voice,
        print_json=namespace.print_json,
        block_mic_during_playback=namespace.block_mic_during_playback,
    )
    try:
        asyncio.run(listen_and_play_realtime(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
