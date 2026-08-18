from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any, Literal

from speech_to_speech.api.openai_realtime.audio_client import (
    RealtimeAudioClientConfig,
    load_realtime_tool_module,
    run_realtime_audio_client,
)
from speech_to_speech.utils.cuda_env import sanitize_cuda_environment

Command = Literal["serve", "talk", "local"]

_LEGACY_MODE_COMMANDS: dict[str, Command] = {
    "realtime": "serve",
    "local": "local",
}


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech-to-speech",
        description="Run or connect to the Realtime speech-to-speech pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.add_parser("serve", add_help=False, help="Run the Realtime pipeline server.")
    subparsers.add_parser("talk", add_help=False, help="Connect microphone and speakers to a Realtime URL.")
    subparsers.add_parser("local", add_help=False, help="Run the server and audio client together over loopback.")
    return parser


def _extract_legacy_mode(command_args: list[str], parser: argparse.ArgumentParser) -> tuple[str | None, list[str]]:
    """Remove one legacy ``--mode`` option without parsing command-owned flags."""

    mode: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(command_args):
        argument = command_args[index]
        if argument == "--mode":
            if mode is not None:
                parser.error("--mode may only be specified once")
            if index + 1 == len(command_args) or command_args[index + 1].startswith("-"):
                parser.error("--mode requires a value: realtime or local")
            mode = command_args[index + 1]
            index += 2
            continue
        if argument.startswith("--mode="):
            if mode is not None:
                parser.error("--mode may only be specified once")
            mode = argument.partition("=")[2]
            if not mode:
                parser.error("--mode requires a value: realtime or local")
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return mode, remaining


def parse_command(argv: Sequence[str] | None = None) -> tuple[Command, list[str]]:
    """Split the top-level command from arguments owned by that command."""

    command_args = list(sys.argv[1:] if argv is None else argv)
    parser = _command_parser()
    if not command_args:
        parser.error("a command is required: serve, talk, or local")
    if command_args[0] in {"-h", "--help"}:
        parser.print_help()
        raise SystemExit(0)
    if command_args[0] not in {"serve", "talk", "local"}:
        legacy_mode, remaining = _extract_legacy_mode(command_args, parser)
        if legacy_mode is not None:
            legacy_command = _LEGACY_MODE_COMMANDS.get(legacy_mode)
            if legacy_command is None:
                parser.error(
                    f"--mode {legacy_mode!r} is no longer supported; only 'realtime' and 'local' remain "
                    "temporarily. Use 'speech-to-speech serve' or 'speech-to-speech local' instead."
                )
            print(
                f"Warning: '--mode {legacy_mode}' is deprecated and will stop working soon; "
                f"use 'speech-to-speech {legacy_command}' instead.",
                file=sys.stderr,
            )
            return legacy_command, remaining
    command = command_args[0]
    if command not in {"serve", "talk", "local"}:
        parser.error(f"unknown command {command!r}; choose serve, talk, or local")
    return command, command_args[1:]  # type: ignore[return-value]


def parse_talk_arguments(argv: Sequence[str]) -> RealtimeAudioClientConfig:
    """Parse the lightweight audio client command."""

    defaults = RealtimeAudioClientConfig()
    parser = argparse.ArgumentParser(
        prog="speech-to-speech talk",
        description="Connect microphone and speakers to an OpenAI-compatible Realtime endpoint.",
    )
    parser.add_argument(
        "--url",
        default=defaults.url,
        help="Full Realtime WebSocket endpoint, including /realtime.",
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument(
        "--api-key",
        default=defaults.api_key,
        help=(
            "Realtime API key. Defaults to OPENAI_API_KEY, or a harmless placeholder for an unauthenticated "
            "loopback endpoint."
        ),
    )
    parser.add_argument("--send-rate", type=int, default=defaults.send_rate)
    parser.add_argument("--recv-rate", type=int, default=defaults.recv_rate)
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--input-device", type=int, default=defaults.input_device)
    parser.add_argument("--output-device", type=int, default=defaults.output_device)
    parser.add_argument("--instructions", default=defaults.instructions)
    parser.add_argument(
        "--tool-module",
        help="Importable module defining TOOLS and async execute_tool(name, arguments).",
    )
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
    parser.add_argument(
        "--connection-retry-timeout",
        type=float,
        default=defaults.connection_retry_timeout_s,
        help="Seconds to wait for the Realtime endpoint to become available.",
    )
    namespace = parser.parse_args(list(argv))
    tools: list[dict[str, Any]] = []
    tool_executor = None
    tool_response_create = defaults.tool_response_create
    if namespace.tool_module:
        tools, tool_executor, tool_response_create = load_realtime_tool_module(namespace.tool_module)
    return RealtimeAudioClientConfig(
        url=namespace.url,
        model=namespace.model,
        api_key=namespace.api_key,
        send_rate=namespace.send_rate,
        recv_rate=namespace.recv_rate,
        chunk_size=namespace.chunk_size,
        input_device=namespace.input_device,
        output_device=namespace.output_device,
        instructions=namespace.instructions,
        voice=namespace.voice,
        print_json=namespace.print_json,
        block_mic_during_playback=namespace.block_mic_during_playback,
        connection_retry_timeout_s=namespace.connection_retry_timeout,
        tools=tools,
        tool_executor=tool_executor,
        tool_response_create=tool_response_create,
    )


def main() -> None:
    sanitize_cuda_environment()
    from speech_to_speech.utils.utils import load_dotenv_if_present

    load_dotenv_if_present()
    command, command_args = parse_command()
    if command == "talk":
        run_realtime_audio_client(parse_talk_arguments(command_args))
        return

    from speech_to_speech.s2s_pipeline import run_pipeline_command

    run_pipeline_command(command, command_args)


if __name__ == "__main__":
    main()
