"""Speech → multi-language translation text pipeline (s2mlt).

A slim sibling of ``s2s_pipeline.py`` for one fixed shape of deployment:

    audio in ─ WebSocket ─> VAD ─> Whisper streaming STT ─> TranslationNotifier
        ─> LLM (transformers | chat-completions) ─> TranslationOutputProcessor
        ─> translation events ─ WebSocket ─> client

There is no TTS stage and no audio output. Clients receive JSON events on the
same WebSocket that carries their audio in:

- ``input.transcription.delta`` / ``input.transcription.done`` - live and
  final transcription per segment;
- ``translation.delta`` / ``translation.done`` - streaming and final
  structured output: one translation per configured target language plus a
  corrected transcript;
- plus the VAD ``speech_started`` / ``speech_stopped`` events.

One VAD turn is one segment. A segment can be reopened (same ``turn_id``,
higher ``turn_revision``) when speech resumes within the merge window; clients
treat every event as an upsert of its segment, keeping the highest revision.

The input language is always auto-detected per decode (code-switching speech
must be transcribed as spoken, never forced into one language).

Run with CLI flags or a single JSON config file:

    s2mlt --llm_backend chat-completions --llm_base_url http://... \\
          --target_languages de fr
    s2mlt prod-config.json
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from queue import Queue
from threading import Event
from types import FrameType
from typing import Literal, Optional

from rich.console import Console
from transformers import HfArgumentParser

from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.connections.websocket_streamer import WebSocketStreamer
from speech_to_speech.LLM.translation import (
    LANGUAGE_NAMES,
    TranslationChatCompletionsHandler,
    TranslationOutputProcessor,
    build_translation_response_format,
)
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.queue_types import (
    AudioInItem,
    AudioOutItem,
    LMOutItem,
    STTOutItem,
    TextEventItem,
    TextPromptItem,
    VADOutItem,
)
from speech_to_speech.STT.translation_notifier import TranslationNotifier
from speech_to_speech.STT.whisper_streaming_stt_handler import WhisperStreamingSTTHandler
from speech_to_speech.utils.thread_manager import ThreadManager
from speech_to_speech.VAD.vad_handler import VADHandler

console = Console()
logger = logging.getLogger(__name__)


# ── Arguments ─────────────────────────────────────────────────────────


@dataclass
class S2MLTArguments:
    llm_backend: Literal["transformers", "chat-completions"] = field(
        default="chat-completions",
        metadata={"help": "The LLM backend to use. Either 'transformers' or 'chat-completions'."},
    )
    target_languages: list[str] = field(
        default_factory=lambda: ["de", "en"],
        metadata={
            "help": "Target language codes to translate every segment into (input language is always "
            "auto-detected). Default is 'de en'."
        },
    )
    ws_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host for the WebSocket server. Default is '0.0.0.0'."},
    )
    ws_port: int = field(
        default=8765,
        metadata={"help": "Port for the WebSocket server. Default is 8765."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Provide logging level. Example --log_level debug, default=info."},
    )


@dataclass
class S2MLTWhisperArguments:
    stt_model_name: str = field(
        default="openai/whisper-large-v3-turbo",
        metadata={"help": "Hugging Face model identifier for the Whisper streaming STT."},
    )
    stt_device: str = field(
        default="auto",
        metadata={"help": "Device for STT inference: 'auto', 'cuda', 'mps', or 'cpu'. Default is 'auto'."},
    )
    stt_torch_dtype: str = field(
        default="float16",
        metadata={"help": "Compute precision for STT: 'float16' or 'float32'. Default is 'float16'."},
    )
    stt_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Force a fixed transcription language. Leave unset (or 'auto') so code-switching "
            "speech is transcribed per sentence in the language actually spoken."
        },
    )
    stt_max_window_size: float = field(
        default=15.0,
        metadata={"help": "Maximum re-transcription window (s) before completed sentences are frozen."},
    )
    stt_sentence_buffer: float = field(
        default=2.0,
        metadata={"help": "Trailing seconds of sentences kept active when the streaming window slides."},
    )


@dataclass
class S2MLTLLMArguments:
    llm_model_name: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "Model name: a HF id for 'transformers', or the served model name for 'chat-completions'."},
    )
    llm_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL of the OpenAI-compatible server (chat-completions backend only)."},
    )
    llm_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key for the OpenAI-compatible server (chat-completions backend only)."},
    )
    llm_request_timeout_s: float = field(
        default=20.0,
        metadata={"help": "Per-request timeout (s) for the chat-completions backend. Default is 20."},
    )
    llm_device: str = field(
        default="cuda",
        metadata={"help": "Device for LLM inference (transformers backend only). Default is 'cuda'."},
    )
    llm_torch_dtype: str = field(
        default="float16",
        metadata={"help": "Torch dtype for the LLM (transformers backend only). Default is 'float16'."},
    )
    llm_max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum tokens generated per segment translation. Default is 1024."},
    )


@dataclass
class ParsedArguments:
    s2mlt: S2MLTArguments
    whisper: S2MLTWhisperArguments
    llm: S2MLTLLMArguments
    vad: VADHandlerArguments


def validate_target_languages(target_languages: list[str]) -> None:
    """Validate the fixed two-language output contract."""
    if len(target_languages) != 2:
        raise ValueError("--target_languages must list exactly two language codes")
    if len(set(target_languages)) != len(target_languages):
        raise ValueError("--target_languages must contain two distinct language codes")
    if "corrected" in target_languages:
        raise ValueError("'corrected' is reserved for the corrected-transcript output field")


def parse_arguments() -> ParsedArguments:
    parser = HfArgumentParser(
        (  # type: ignore[arg-type]
            S2MLTArguments,
            S2MLTWhisperArguments,
            S2MLTLLMArguments,
            VADHandlerArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        parsed = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        parsed = parser.parse_args_into_dataclasses()
    args = ParsedArguments(*parsed)

    validate_target_languages(args.s2mlt.target_languages)
    for code in args.s2mlt.target_languages:
        if code not in LANGUAGE_NAMES:
            logger.warning("Unknown target language code %r; it will be passed to the LLM verbatim", code)
    return args


def setup_logger(log_level: str) -> None:
    from speech_to_speech.pipeline.log_context import PipelineLogFilter

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(pipeline_prefix)s%(name)s - %(levelname)s - %(message)s",
    )
    pipeline_filter = PipelineLogFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(pipeline_filter)


# ── Pipeline construction ─────────────────────────────────────────────


class ContinuousListeningEvent(Event):
    """``should_listen`` that cannot be paused.

    The half-duplex pipeline clears ``should_listen`` at end of speech so the
    mic won't hear the bot's own TTS. A translator has no audio output and
    must keep transcribing while previous segments are still being translated,
    so ``clear()`` is a no-op; ``set()`` (client connect, session reset) works
    normally.
    """

    def clear(self) -> None:
        pass


def build_llm_handler(
    args: ParsedArguments,
    stop_event: Event,
    text_prompt_queue: Queue[TextPromptItem],
    lm_response_queue: Queue[LMOutItem],
) -> BaseHandler[LLMIn, LLMOut]:
    if args.s2mlt.llm_backend == "chat-completions":
        return TranslationChatCompletionsHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs={
                "model_name": args.llm.llm_model_name,
                "base_url": args.llm.llm_base_url,
                "api_key": args.llm.llm_api_key,
                "stream": True,
                "request_timeout_s": args.llm.llm_request_timeout_s,
                "response_format": build_translation_response_format(args.s2mlt.target_languages),
            },
        )

    if args.s2mlt.llm_backend == "transformers":
        from speech_to_speech.LLM.language_model import LanguageModelHandler

        return LanguageModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs={
                "model_name": args.llm.llm_model_name,
                "device": args.llm.llm_device,
                "torch_dtype": args.llm.llm_torch_dtype,
                "backend": "transformers",
                "gen_kwargs": {
                    "max_new_tokens": args.llm.llm_max_new_tokens,
                    "min_new_tokens": 0,
                    "do_sample": False,
                },
            },
        )

    raise ValueError("The LLM backend should be either 'transformers' or 'chat-completions'")


def build_pipeline(args: ParsedArguments, stop_event: Event) -> ThreadManager:
    should_listen = ContinuousListeningEvent()

    recv_audio_chunks_queue: Queue[AudioInItem] = Queue()
    send_audio_chunks_queue: Queue[AudioOutItem] = Queue()  # required by the streamer; never fed (no TTS)
    spoken_prompt_queue: Queue[VADOutItem] = Queue()
    stt_output_queue: Queue[STTOutItem] = Queue()
    text_prompt_queue: Queue[TextPromptItem] = Queue()
    lm_response_queue: Queue[LMOutItem] = Queue()
    text_output_queue: Queue[TextEventItem] = Queue()

    streamer = WebSocketStreamer(
        stop_event,
        input_queue=recv_audio_chunks_queue,
        output_queue=send_audio_chunks_queue,
        should_listen=should_listen,
        text_output_queue=text_output_queue,
        host=args.s2mlt.ws_host,
        port=args.s2mlt.ws_port,
    )

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs={
            **vars(args.vad),
            # Progressive audio release drives the live transcription deltas;
            # speculative_reopen_ms doubles as the segment merge window.
            "enable_realtime_transcription": True,
            "text_output_queue": text_output_queue,
        },
    )

    stt = WhisperStreamingSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=stt_output_queue,
        setup_kwargs={
            "model_name": args.whisper.stt_model_name,
            "device": args.whisper.stt_device,
            "torch_dtype": args.whisper.stt_torch_dtype,
            "language": args.whisper.stt_language,
            "live_transcription_update_interval": args.vad.realtime_processing_pause,
            "max_window_size": args.whisper.stt_max_window_size,
            "sentence_buffer": args.whisper.stt_sentence_buffer,
        },
    )

    notifier = TranslationNotifier(
        stop_event,
        queue_in=stt_output_queue,
        queue_out=text_prompt_queue,  # type: ignore[arg-type]
        setup_kwargs={
            "target_languages": args.s2mlt.target_languages,
            "text_output_queue": text_output_queue,
        },
    )

    lm = build_llm_handler(args, stop_event, text_prompt_queue, lm_response_queue)

    translation_processor = TranslationOutputProcessor(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=text_output_queue,  # type: ignore[arg-type]
        setup_kwargs={"target_languages": args.s2mlt.target_languages},
    )

    return ThreadManager([streamer, vad, stt, notifier, lm, translation_processor])


def main() -> None:
    args = parse_arguments()
    setup_logger(args.s2mlt.log_level)

    stop_event = Event()
    pipeline_manager = build_pipeline(args, stop_event)

    console.print(
        f"[green]s2mlt[/green]: translating auto-detected speech into "
        f"[bold]{', '.join(args.s2mlt.target_languages)}[/bold] "
        f"(llm_backend={args.s2mlt.llm_backend}, ws://{args.s2mlt.ws_host}:{args.s2mlt.ws_port})"
    )

    shutdown_requested = [False]

    def signal_handler(_sig: int, _frame: Optional[FrameType]) -> None:
        if not shutdown_requested[0]:
            shutdown_requested[0] = True
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            pipeline_manager.stop()
            console.print("[green]✓ Pipeline stopped successfully[/green]")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        pipeline_manager.start()
        pipeline_manager.wait()
    except KeyboardInterrupt:
        if not shutdown_requested[0]:
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            pipeline_manager.stop()
            console.print("[green]✓ Pipeline stopped successfully[/green]")


if __name__ == "__main__":
    main()
