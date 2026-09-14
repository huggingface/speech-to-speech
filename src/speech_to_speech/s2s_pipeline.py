import argparse
import json
import logging
import os
import signal
import sys
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from queue import Queue
from sys import platform
from threading import Event
from types import FrameType
from typing import Any, Literal, Optional, Sequence

import nltk
import torch
from rich.console import Console
from transformers import HfArgumentParser

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.arguments_classes.local_audio_arguments import LocalAudioArguments
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.realtime_server_arguments import (
    LocalRealtimeServerArguments,
    RealtimeServerArguments,
)
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.arguments_classes.vision_resolver_arguments import VisionResolverArguments
from speech_to_speech.arguments_classes.websocket_streamer_arguments import WebSocketStreamerArguments
from speech_to_speech.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.queue_types import (
    AudioInItem,
    AudioOutItem,
    LMOutItem,
    STTOutItem,
    TextEventItem,
    TextPromptItem,
    TTSInItem,
    VADOutItem,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import (
    set_log_transcripts,
    warn_if_log_transcripts_enabled,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.utils.thread_manager import ThreadManager
from speech_to_speech.VAD.vad_handler import VADHandler

# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

# caching allows ~50% compilation time reduction
# see https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.o2asbxsrp1ma
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

console = Console()
logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)  # quiet down numba logs

MLX_DEFAULT_LM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OPENAI_TTS_PLAYBACK_BUFFER_MS = 196.0


def _mac_preset_defaults(llm_backend: str) -> dict[str, Any]:
    """Return macOS parser defaults, leaving explicit arguments free to override them."""

    defaults: dict[str, Any] = {
        "stt": "parakeet-tdt",
        "llm_backend": "mlx-lm",
        "tts": "qwen3",
        "stt_device": "mps",
        "paraformer_stt_device": "mps",
        "facebook_mms_device": "mps",
        "qwen3_tts_device": "mps",
    }
    if llm_backend not in {"responses-api", "chat-completions"}:
        defaults["llm_device"] = "mps"
        if llm_backend == "mlx-lm":
            defaults["model_name"] = MLX_DEFAULT_LM_MODEL
    return defaults


@dataclass
class ParsedArguments:
    module_kwargs: ModuleArguments
    realtime_server_kwargs: RealtimeServerArguments
    local_audio_kwargs: LocalAudioArguments
    vad_handler_kwargs: VADHandlerArguments
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments
    language_model_handler_kwargs: LanguageModelHandlerArguments
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments
    vision_resolver_kwargs: VisionResolverArguments
    chat_tts_handler_kwargs: ChatTTSHandlerArguments
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments


def rename_args(args: Any, prefix: str) -> None:
    """
    Rename arguments by removing the prefix and prepares the gen_kwargs.
    """
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1 :]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs


def build_llm_proxy_config(
    module_kwargs: ModuleArguments,
    llm_backend: BackendSelection,
) -> Any:
    """Build proxy settings from the selected LLM's normalized configuration."""
    from speech_to_speech.api.openai_realtime.llm_proxy import LLMProxyConfig

    if not llm_backend.spec.capabilities.supports_llm_proxy:
        supported = ", ".join(name for name, spec in LLM_BACKENDS.items() if spec.capabilities.supports_llm_proxy)
        raise ValueError(
            f"The LLM proxy requires a backend with proxy support; choose one of: {supported}. "
            f"Got {llm_backend.name!r}."
        )
    config = llm_backend.config
    return LLMProxyConfig(
        enabled=module_kwargs.enable_llm_proxy,
        llm_backend=module_kwargs.llm_backend,
        upstream_base_url=config["base_url"],
        upstream_api_key=config["api_key"],
        model_name=config["model_name"],
        connect_timeout_s=module_kwargs.llm_proxy_connect_timeout_s,
    )


def _parse_selected_cli_configs(
    parser: HfArgumentParser,
    pipeline_args: list[str],
    selected_specs: Sequence[BackendSpec],
) -> tuple[Any, ...]:
    """Parse selected configs while accepting legacy options for inactive backends."""

    *parsed, remaining = parser.parse_args_into_dataclasses(
        args=pipeline_args,
        return_remaining_strings=True,
    )
    if not remaining:
        return tuple(parsed)

    selected_types = {spec.config_type for spec in selected_specs}
    inactive_types: list[type[Any]] = []
    for registry in (STT_BACKENDS, LLM_BACKENDS, TTS_BACKENDS):
        for spec in registry.values():
            if spec.config_type not in selected_types and spec.config_type not in inactive_types:
                inactive_types.append(spec.config_type)

    compatibility_parser = HfArgumentParser(
        tuple(inactive_types),  # type: ignore[arg-type]
        add_help=False,
        allow_abbrev=False,
        conflict_handler="resolve",
    )
    known_options = compatibility_parser._option_string_actions
    _, unknown = compatibility_parser.parse_known_args(remaining)
    if unknown:
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown}")

    ignored_options = sorted({token.split("=", 1)[0] for token in remaining if token.split("=", 1)[0] in known_options})
    logger.warning(
        "Ignoring options for inactive backends: %s",
        ", ".join(ignored_options),
    )
    return tuple(parsed)


def parse_arguments(
    argv: Sequence[str] | None = None,
    *,
    command: Literal["serve", "local"] = "serve",
) -> ParsedArguments:
    module_defaults = ModuleArguments()
    assert module_defaults.stt is not None
    assert module_defaults.llm_backend is not None
    assert module_defaults.tts is not None

    pipeline_args = list(sys.argv[1:] if argv is None else argv)
    _is_json = len(pipeline_args) == 1 and pipeline_args[0].endswith(".json")
    pipeline_json: dict[str, Any] | None = None
    if _is_json:
        with open(pipeline_args[0]) as _f:
            pipeline_json = json.load(_f)
        _mac_preset_enabled = bool(pipeline_json.get("mac_optimal_settings", False))
        _llm_name = pipeline_json.get("llm_backend") or (
            "mlx-lm" if _mac_preset_enabled else module_defaults.llm_backend
        )
        if _mac_preset_enabled:
            pipeline_json = {**_mac_preset_defaults(_llm_name), **pipeline_json}
        _stt_name = pipeline_json.get("stt") or module_defaults.stt
        _tts_name = pipeline_json.get("tts") or module_defaults.tts
    else:
        _pre = argparse.ArgumentParser(prog=f"speech-to-speech {command}", add_help=False)
        _pre.add_argument("--mac-optimal-settings", action="store_true")
        _pre.add_argument("--stt", choices=tuple(STT_BACKENDS))
        _pre.add_argument("--llm_backend", "--llm-backend", choices=tuple(LLM_BACKENDS))
        _pre.add_argument("--tts", choices=tuple(TTS_BACKENDS))
        _pre_args = _pre.parse_known_args(pipeline_args)[0]
        _mac_preset_enabled = _pre_args.mac_optimal_settings
        _stt_name = _pre_args.stt or module_defaults.stt
        _llm_name = _pre_args.llm_backend or ("mlx-lm" if _mac_preset_enabled else module_defaults.llm_backend)
        _tts_name = _pre_args.tts or module_defaults.tts

    selected_specs = []
    for registry, name in (
        (STT_BACKENDS, _stt_name),
        (LLM_BACKENDS, _llm_name),
        (TTS_BACKENDS, _tts_name),
    ):
        try:
            selected_specs.append(registry[name])
        except KeyError as exc:
            choices = ", ".join(registry)
            raise ValueError(f"Unsupported backend {name!r}; choose one of: {choices}.") from exc

    logger.debug(
        "Backend pre-parse: stt=%s, llm=%s, tts=%s",
        _stt_name,
        _llm_name,
        _tts_name,
    )

    server_arguments_class = RealtimeServerArguments if command == "serve" else LocalRealtimeServerArguments
    argument_classes: list[type[Any]] = [
        ModuleArguments,
        server_arguments_class,
    ]
    if command == "local":
        argument_classes.append(LocalAudioArguments)
    argument_classes.extend(
        [
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            ParaformerSTTHandlerArguments,
            FasterWhisperSTTHandlerArguments,
            MLXAudioWhisperSTTHandlerArguments,
            ParakeetTDTSTTHandlerArguments,
            _lm_class,
            VisionResolverArguments,
            ChatTTSHandlerArguments,
            FacebookMMSTTSHandlerArguments,
            PocketTTSHandlerArguments,
            KokoroTTSHandlerArguments,
            Qwen3TTSHandlerArguments,
        )
    )
    parser = HfArgumentParser(tuple(argument_classes), prog=f"speech-to-speech {command}")  # type: ignore[arg-type]
    mac_action = parser._option_string_actions.pop("--mac_optimal_settings")
    mac_action.option_strings = [option for option in mac_action.option_strings if option != "--mac_optimal_settings"]
    if _mac_preset_enabled:
        parser.set_defaults(**_mac_preset_defaults(_llm_name))

    if _is_json:
        assert pipeline_json is not None
        parsed = parser.parse_dict(pipeline_json, allow_extra_keys=True)
    else:
        parsed = _parse_selected_cli_configs(parser, pipeline_args, selected_specs)

    # Build a {type: instance} lookup so field assignment is order-independent.
    by_type: dict[type, Any] = {type(obj): obj for obj in parsed}
    logger.debug("Parsed %d argument classes: %s", len(by_type), [t.__name__ for t in by_type])
    if command == "serve":
        realtime_server_kwargs = by_type[RealtimeServerArguments]
    else:
        realtime_server_kwargs = RealtimeServerArguments(
            host="127.0.0.1",
            port=by_type[LocalRealtimeServerArguments].port,
        )

    module_kwargs = by_type[ModuleArguments]
    module_kwargs.stt = _stt_name
    module_kwargs.llm_backend = _llm_name
    module_kwargs.tts = _tts_name
    args = ParsedArguments(
        module_kwargs=module_kwargs,
        realtime_server_kwargs=realtime_server_kwargs,
        local_audio_kwargs=by_type.get(LocalAudioArguments, LocalAudioArguments()),
        vad_handler_kwargs=by_type[VADHandlerArguments],
        stt_backend=BackendSelection(
            selected_specs[0], selected_specs[0].normalize(by_type[selected_specs[0].config_type])
        ),
        llm_backend=BackendSelection(
            selected_specs[1], selected_specs[1].normalize(by_type[selected_specs[1].config_type])
        ),
        tts_backend=BackendSelection(
            selected_specs[2], selected_specs[2].normalize(by_type[selected_specs[2].config_type])
        ),
        vision_resolver_kwargs=by_type.get(VisionResolverArguments, VisionResolverArguments()),
        chat_tts_handler_kwargs=by_type[ChatTTSHandlerArguments],
        facebook_mms_tts_handler_kwargs=by_type[FacebookMMSTTSHandlerArguments],
        pocket_tts_handler_kwargs=by_type[PocketTTSHandlerArguments],
        kokoro_tts_handler_kwargs=by_type[KokoroTTSHandlerArguments],
        qwen3_tts_handler_kwargs=by_type[Qwen3TTSHandlerArguments],
    )
    return args


def setup_logger(log_level: str) -> None:
    global logger
    from speech_to_speech.pipeline.log_context import PipelineLogFilter

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(pipeline_prefix)s%(name)s - %(levelname)s - %(message)s",
    )
    # Attach the filter to every existing handler so each LogRecord gets a
    # `pipeline_prefix` attribute (matching the format string above).
    pipeline_filter = PipelineLogFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(pipeline_filter)

    logger = logging.getLogger(__name__)

    # torch compile logs
    if log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)


def check_mac_settings(module_kwargs: ModuleArguments) -> None:
    if platform == "darwin":
        if module_kwargs.device == "cuda":
            raise ValueError("Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'.")
        if module_kwargs.llm_backend != "mlx-lm":
            logger.warning(
                "For macOS users, it is recommended to use mlx-lm. You can activate it by passing --llm_backend mlx-lm."
            )
        if module_kwargs.tts not in ("pocket", "kokoro", "omnivoice", "qwen3"):
            logger.warning(
                "For macOS users, it is recommended to use qwen3 for TTS "
                "(pocket, kokoro, and omnivoice are also valid options)."
            )


def prepare_module_args(module_kwargs: ModuleArguments, llm_backend: BackendSelection) -> None:
    if module_kwargs.tts is None:
        module_kwargs.tts = "qwen3"
    if module_kwargs.stt == "none" and not llm_backend.spec.capabilities.supports_audio_input:
        supported = ", ".join(name for name, spec in LLM_BACKENDS.items() if spec.capabilities.supports_audio_input)
        raise ValueError(f"--stt none requires an audio-input LLM backend; choose one of: {supported}.")
    if module_kwargs.enable_llm_proxy and not llm_backend.spec.capabilities.supports_llm_proxy:
        supported = ", ".join(name for name, spec in LLM_BACKENDS.items() if spec.capabilities.supports_llm_proxy)
        raise ValueError(
            f"The LLM proxy requires a backend with proxy support; choose one of: {supported}. "
            f"Got {llm_backend.name!r}."
        )
    if platform == "darwin":
        check_mac_settings(module_kwargs)


def prepare_all_args(
    module_kwargs: ModuleArguments,
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
    vision_resolver_kwargs: VisionResolverArguments,
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
) -> None:
    prepare_module_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
        mlx_audio_whisper_stt_handler_kwargs,
        parakeet_tdt_stt_handler_kwargs,
        language_model_handler_kwargs,
        responses_api_language_model_handler_kwargs,
        chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs,
        pocket_tts_handler_kwargs,
        kokoro_tts_handler_kwargs,
        qwen3_tts_handler_kwargs,
    )

    rename_args(whisper_stt_handler_kwargs, "stt")
    rename_args(faster_whisper_stt_handler_kwargs, "faster_whisper_stt")
    rename_args(paraformer_stt_handler_kwargs, "paraformer_stt")
    rename_args(mlx_audio_whisper_stt_handler_kwargs, "mlx_audio_whisper")
    rename_args(parakeet_tdt_stt_handler_kwargs, "parakeet_tdt")
    rename_args(language_model_handler_kwargs, "llm")
    rename_args(responses_api_language_model_handler_kwargs, "responses_api")
    rename_args(vision_resolver_kwargs, "vision")
    rename_args(chat_tts_handler_kwargs, "chat_tts")
    rename_args(facebook_mms_tts_handler_kwargs, "facebook_mms")
    rename_args(pocket_tts_handler_kwargs, "pocket_tts")
    rename_args(kokoro_tts_handler_kwargs, "kokoro")
    rename_args(qwen3_tts_handler_kwargs, "qwen3_tts")


def initialize_queues_and_events() -> dict[str, Any]:
    return {
        "stop_event": Event(),
        "should_listen": Event(),
        "response_playing": Event(),
        "cancel_scope": CancelScope(),
        "recv_audio_chunks_queue": Queue[AudioInItem](),
        "send_audio_chunks_queue": Queue[AudioOutItem](),
        "spoken_prompt_queue": Queue[VADOutItem](),
        "stt_output_queue": Queue[STTOutItem](),
        "text_prompt_queue": Queue[TextPromptItem](),
        "lm_response_queue": Queue[LMOutItem](),
        "lm_processed_queue": Queue[TTSInItem](),  # NEW: LLM -> LM processor -> TTS
        "text_output_queue": Queue[TextEventItem](),  # NEW: for text messages to WebSocket
    }


def _build_handlers(
    *,
    stop_event: Event,
    should_listen: Event,
    recv_audio_chunks_queue: Queue[AudioInItem],
    spoken_prompt_queue: Queue[VADOutItem],
    stt_output_queue: Queue[STTOutItem],
    text_prompt_queue: Queue[TextPromptItem],
    lm_response_queue: Queue[LMOutItem],
    lm_processed_queue: Queue[TTSInItem],
    send_audio_chunks_queue: Queue[AudioOutItem],
    text_output_queue: Queue[TextEventItem],
    module_kwargs: ModuleArguments,
    vad_handler_kwargs: VADHandlerArguments,
    stt_backend: BackendSelection,
    llm_backend: BackendSelection,
    tts_backend: BackendSelection,
    speculative_turns: SpeculativeTurnTracker,
    cancel_scope: CancelScope,
    pipeline_index: int,
) -> list[Any]:
    """Build a handler chain: VAD → STT/AudioInput → LM → TTS."""
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs={
            **{
                config_field.name: deepcopy(getattr(vad_handler_kwargs, config_field.name))
                for config_field in fields(vad_handler_kwargs)
            },
            "text_output_queue": text_output_queue,
            "speculative_turns": speculative_turns,
        },
    )

    needs_notifier = not stt_backend.spec.capabilities.bypasses_transcription_notifier
    stt_queue_out: Queue[Any] = stt_output_queue if needs_notifier else text_prompt_queue
    stt_context = HandlerContext(
        stop_event=stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=stt_queue_out,
        text_output_queue=text_output_queue,
        should_listen=should_listen,
        cancel_scope=cancel_scope,
        speculative_turns=speculative_turns,
        pipeline_index=pipeline_index,
        sample_rate=vad_handler_kwargs.sample_rate,
        enable_live_transcription=module_kwargs.enable_live_transcription,
        live_transcription_update_interval=module_kwargs.live_transcription_update_interval,
    )
    stt_handler = create_backend_handler(stt_backend, stt_context)
    if stt_backend.spec.capabilities.streams_audio_chunks:
        vad.streaming_stt_sink = stt_handler
    speech_input_handlers = [stt_handler]
    if needs_notifier:
        transcription_notifier = TranscriptionNotifier(
            stop_event,
            queue_in=stt_output_queue,
            queue_out=text_prompt_queue,  # type: ignore[arg-type]
            setup_kwargs={
                "text_output_queue": text_output_queue,
                "should_listen": should_listen,
            },
        )
        speech_input_handlers.append(transcription_notifier)

    def handler_context(queue_in: Queue[Any], queue_out: Queue[Any]) -> HandlerContext:
        return HandlerContext(
            stop_event=stop_event,
            queue_in=queue_in,
            queue_out=queue_out,
            text_output_queue=text_output_queue,
            should_listen=should_listen,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
            pipeline_index=pipeline_index,
            sample_rate=vad_handler_kwargs.sample_rate,
            enable_live_transcription=module_kwargs.enable_live_transcription,
            live_transcription_update_interval=module_kwargs.live_transcription_update_interval,
        )

    lm_context = handler_context(text_prompt_queue, lm_response_queue)
    lm = create_backend_handler(
        llm_backend,
        lm_context,
    )

    lm_processor = LMOutputProcessor(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=lm_processed_queue,
        setup_kwargs={
            "speculative_turns": speculative_turns,
            "text_output_queue": text_output_queue,
        },
    )

    tts_context = handler_context(lm_processed_queue, send_audio_chunks_queue)
    tts = create_backend_handler(
        tts_backend,
        tts_context,
    )

    return [vad, *speech_input_handlers, lm, lm_processor, tts]


def _maybe_build_vision_resolver(vision_resolver_kwargs: VisionResolverArguments) -> Any | None:
    model_name = vars(vision_resolver_kwargs).get("model_name") or getattr(
        vision_resolver_kwargs, "vision_model_name", None
    )
    if not model_name:
        return None
    from speech_to_speech.LLM.vision_resolver import VisionResolver

    return VisionResolver(
        model_name=model_name,
        base_url=vars(vision_resolver_kwargs).get("base_url"),
        api_key=vars(vision_resolver_kwargs).get("api_key"),
        max_tokens=vars(vision_resolver_kwargs).get("max_tokens", 300),
        timeout_s=vars(vision_resolver_kwargs).get("timeout_s", 10.0),
    )


def _build_realtime_pipeline_unit(
    *,
    index: int,
    stop_event: Event,
    module_kwargs: ModuleArguments,
    vad_handler_kwargs: VADHandlerArguments,
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
    vision_resolver_kwargs: VisionResolverArguments,
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
) -> "PipelineUnit":
    """Build one isolated pipeline with its own state and queues.

    Returns a PipelineUnit that lives inside RealtimeServer's pool. Handler instances
    are returned in `unit.handlers`; the caller threads them via ThreadManager. No uvicorn
    is created here — the single server is owned by RealtimeServer.
    """
    from speech_to_speech.api.openai_realtime.service import RealtimeService

    # Per-unit copies isolate any setup-time mutation performed by third-party libraries.
    vad_kw = deepcopy(vad_handler_kwargs)
    whisper_kw = deepcopy(whisper_stt_handler_kwargs)
    faster_whisper_kw = deepcopy(faster_whisper_stt_handler_kwargs)
    paraformer_kw = deepcopy(paraformer_stt_handler_kwargs)
    mlx_audio_whisper_kw = deepcopy(mlx_audio_whisper_stt_handler_kwargs)
    parakeet_kw = deepcopy(parakeet_tdt_stt_handler_kwargs)
    lm_kw = deepcopy(language_model_handler_kwargs)
    responses_api_kw = deepcopy(responses_api_language_model_handler_kwargs)
    vision_kw = deepcopy(vision_resolver_kwargs)
    chat_tts_kw = deepcopy(chat_tts_handler_kwargs)
    facebook_mms_kw = deepcopy(facebook_mms_tts_handler_kwargs)
    pocket_tts_kw = deepcopy(pocket_tts_handler_kwargs)
    kokoro_tts_kw = deepcopy(kokoro_tts_handler_kwargs)
    qwen3_tts_kw = deepcopy(qwen3_tts_handler_kwargs)

    vision_resolver = _maybe_build_vision_resolver(vision_kw)
    if vision_resolver is not None:
        vars(lm_kw)["vision_resolver"] = vision_resolver
        vars(responses_api_kw)["vision_resolver"] = vision_resolver

    should_listen = Event()
    response_playing = Event()
    cancel_scope = CancelScope()
    speculative_turns = SpeculativeTurnTracker()
    recv_audio_chunks_queue: Queue[AudioInItem] = Queue()
    send_audio_chunks_queue: Queue[AudioOutItem] = Queue()
    spoken_prompt_queue: Queue[VADOutItem] = Queue()
    stt_output_queue: Queue[STTOutItem] = Queue()
    text_prompt_queue: Queue[TextPromptItem] = Queue()
    lm_response_queue: Queue[LMOutItem] = Queue()
    lm_processed_queue: Queue[TTSInItem] = Queue()
    text_output_queue: Queue[TextEventItem] = Queue()

    chat_size = llm_selection.config.get("chat_size", 10)
    default_instructions = llm_selection.config.get("init_chat_prompt")

    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
        chat_size=chat_size,
        speculative_turns=speculative_turns,
        default_instructions=default_instructions,
    )

    if module_kwargs.enable_live_transcription and not stt_selection.spec.capabilities.streams_audio_chunks:
        vad_kw.enable_realtime_transcription = True
        vad_kw.realtime_processing_pause = module_kwargs.live_transcription_update_interval

    handlers = _build_handlers(
        stop_event=stop_event,
        should_listen=should_listen,
        recv_audio_chunks_queue=recv_audio_chunks_queue,
        spoken_prompt_queue=spoken_prompt_queue,
        stt_output_queue=stt_output_queue,
        text_prompt_queue=text_prompt_queue,
        lm_response_queue=lm_response_queue,
        lm_processed_queue=lm_processed_queue,
        send_audio_chunks_queue=send_audio_chunks_queue,
        text_output_queue=text_output_queue,
        module_kwargs=module_kwargs,
        vad_handler_kwargs=vad_kw,
        stt_backend=stt_selection,
        llm_backend=llm_selection,
        tts_backend=tts_selection,
        speculative_turns=speculative_turns,
        cancel_scope=cancel_scope,
        pipeline_index=index,
    )
    for h in handlers:
        h.pipeline_index = index

    return PipelineUnit(
        index=index,
        service=service,
        cancel_scope=cancel_scope,
        should_listen=should_listen,
        response_playing=response_playing,
        input_queue=recv_audio_chunks_queue,
        output_queue=send_audio_chunks_queue,
        text_output_queue=text_output_queue,
        text_prompt_queue=text_prompt_queue,
        handlers=handlers,
    )


def build_pipeline(
    module_kwargs: ModuleArguments,
    socket_receiver_kwargs: SocketReceiverArguments,
    socket_sender_kwargs: SocketSenderArguments,
    websocket_streamer_kwargs: WebSocketStreamerArguments,
    vad_handler_kwargs: VADHandlerArguments,
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
    vision_resolver_kwargs: VisionResolverArguments,
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
    queues_and_events: dict[str, Any],
) -> ThreadManager:
    """Build a pool of pipeline units behind one server."""
    from speech_to_speech.api.openai_realtime.server import RealtimeServer

    comms_handlers: list[Any] = []
    if module_kwargs.mode == "local":
        from speech_to_speech.connections.local_audio_streamer import LocalAudioStreamer

        local_audio_streamer = LocalAudioStreamer(
            input_queue=recv_audio_chunks_queue,
            output_queue=send_audio_chunks_queue,
            should_listen=should_listen,
        )
        comms_handlers = [local_audio_streamer]
        should_listen.set()
    elif module_kwargs.mode == "websocket":
        from speech_to_speech.connections.websocket_streamer import WebSocketStreamer

        text_output_queue = queues_and_events["text_output_queue"]
        websocket_streamer = WebSocketStreamer(
            stop_event,
            input_queue=recv_audio_chunks_queue,
            output_queue=send_audio_chunks_queue,
            should_listen=should_listen,
            text_output_queue=text_output_queue,
            host=websocket_streamer_kwargs.ws_host,
            port=websocket_streamer_kwargs.ws_port,
        )
        comms_handlers = [websocket_streamer]
    elif module_kwargs.mode == "realtime":
        from speech_to_speech.api.openai_realtime.server import RealtimeServer

        pool_size = max(1, module_kwargs.num_pipelines)
        pool = [
            _build_realtime_pipeline_unit(
                index=i,
                stop_event=stop_event,
                module_kwargs=module_kwargs,
                vad_handler_kwargs=vad_handler_kwargs,
                whisper_stt_handler_kwargs=whisper_stt_handler_kwargs,
                faster_whisper_stt_handler_kwargs=faster_whisper_stt_handler_kwargs,
                paraformer_stt_handler_kwargs=paraformer_stt_handler_kwargs,
                mlx_audio_whisper_stt_handler_kwargs=mlx_audio_whisper_stt_handler_kwargs,
                parakeet_tdt_stt_handler_kwargs=parakeet_tdt_stt_handler_kwargs,
                language_model_handler_kwargs=language_model_handler_kwargs,
                responses_api_language_model_handler_kwargs=responses_api_language_model_handler_kwargs,
                vision_resolver_kwargs=vision_resolver_kwargs,
                chat_tts_handler_kwargs=chat_tts_handler_kwargs,
                facebook_mms_tts_handler_kwargs=facebook_mms_tts_handler_kwargs,
                pocket_tts_handler_kwargs=pocket_tts_handler_kwargs,
                kokoro_tts_handler_kwargs=kokoro_tts_handler_kwargs,
                qwen3_tts_handler_kwargs=qwen3_tts_handler_kwargs,
            )
            for i in range(pool_size)
        ]

        llm_proxy_config = build_llm_proxy_config(module_kwargs, responses_api_language_model_handler_kwargs)

        realtime_server = RealtimeServer(
            stop_event=stop_event,
            module_kwargs=module_kwargs,
            vad_handler_kwargs=args.vad_handler_kwargs,
            stt_backend=args.stt_backend,
            llm_backend=args.llm_backend,
            tts_backend=args.tts_backend,
        )
        for index in range(module_kwargs.num_pipelines)
    ]

        all_handlers: list[Any] = [realtime_server]
        for unit in pool:
            all_handlers.extend(unit.handlers)
        return ThreadManager(all_handlers)
    else:
        from speech_to_speech.connections.socket_receiver import SocketReceiver
        from speech_to_speech.connections.socket_sender import SocketSender

        comms_handlers = [
            SocketReceiver(
                stop_event,
                recv_audio_chunks_queue,
                should_listen,
                host=socket_receiver_kwargs.recv_host,
                port=socket_receiver_kwargs.recv_port,
                chunk_size=socket_receiver_kwargs.chunk_size,
            ),
            SocketSender(
                stop_event,
                send_audio_chunks_queue,
                should_listen,
                host=socket_sender_kwargs.send_host,
                port=socket_sender_kwargs.send_port,
            ),
        ]

    # Set VAD realtime transcription parameters from module_kwargs
    if module_kwargs.enable_live_transcription:
        vad_handler_kwargs.enable_realtime_transcription = True
        vad_handler_kwargs.realtime_processing_pause = module_kwargs.live_transcription_update_interval

    vision_resolver = _maybe_build_vision_resolver(vision_resolver_kwargs)
    if vision_resolver is not None:
        vars(language_model_handler_kwargs)["vision_resolver"] = vision_resolver
        vars(responses_api_language_model_handler_kwargs)["vision_resolver"] = vision_resolver

    if module_kwargs.llm_backend in ("responses-api", "chat-completions"):
        _lm_vars = vars(responses_api_language_model_handler_kwargs)
    else:
        _lm_vars = vars(language_model_handler_kwargs)
    transcription_notifier_setup: dict[str, Any] = {
        "text_output_queue": text_output_queue,
        "should_listen": should_listen,
        "runtime_config": RuntimeConfig(
            chat=Chat(_lm_vars.get("chat_size", 30)),
            session=RealtimeSessionCreateRequest(
                type="realtime",
                instructions=_lm_vars.get("init_chat_prompt"),
            ),
        ),
    }

    pipeline_handlers = _build_pipeline_handlers(
        stop_event=stop_event,
        pool=pool,
        host=host or args.realtime_server_kwargs.host,
        port=args.realtime_server_kwargs.port,
        llm_proxy_config=(
            build_llm_proxy_config(module_kwargs, args.llm_backend) if module_kwargs.enable_llm_proxy else None
        ),
    )

    handlers: list[Any] = []
    for unit in pool:
        handlers.extend(unit.handlers)
    handlers.append(server)
    return ThreadManager(handlers)


def build_local_pipeline(args: ParsedArguments, stop_event: Event) -> ThreadManager:
    """Compose the canonical server and audio client over a forced loopback URL."""

    from speech_to_speech.api.openai_realtime.audio_client import (
        RealtimeAudioClient,
        RealtimeAudioClientConfig,
        load_realtime_tool_module,
    )

    local_audio = args.local_audio_kwargs
    playback_buffer_ms = local_audio.local_audio_playback_buffer_ms
    if playback_buffer_ms is None:
        playback_buffer_ms = OPENAI_TTS_PLAYBACK_BUFFER_MS if args.tts_backend.name == "openai" else 0.0
    tools: list[dict[str, Any]] = []
    tool_executor = None
    tool_response_create = True
    if local_audio.local_audio_tool_module:
        tools, tool_executor, tool_response_create = load_realtime_tool_module(local_audio.local_audio_tool_module)
    server_manager = build_pipeline(args, stop_event, host="127.0.0.1")
    client = RealtimeAudioClient(
        stop_event,
        RealtimeAudioClientConfig(
            url=f"ws://127.0.0.1:{args.realtime_server_kwargs.port}/v1/realtime",
            api_key="local",
            chunk_size=local_audio.local_audio_chunk_size,
            playback_buffer_ms=playback_buffer_ms,
            input_device=local_audio.local_audio_input_device,
            output_device=local_audio.local_audio_output_device,
            print_json=local_audio.local_audio_print_json,
            block_mic_during_playback=local_audio.local_audio_block_mic_during_playback,
            tools=tools,
            tool_executor=tool_executor,
            tool_response_create=tool_response_create,
        ),
    )
    return ThreadManager([*server_manager.handlers, client])


def run_pipeline_command(command: Literal["serve", "local"], argv: Sequence[str]) -> None:
    """Run the server alone or compose it with the loopback audio client."""

    args = parse_arguments(argv, command=command)

    setup_logger(args.module_kwargs.log_level)
    # Set the transcript gate and warn before any conversation is processed, so an operator
    # sees the notice ahead of the first turn rather than after content is already logged.
    set_log_transcripts(args.module_kwargs.log_transcripts)
    warn_if_log_transcripts_enabled()

    if args.module_kwargs.num_pipelines < 1:
        raise ValueError(f"--num_pipelines must be >= 1, got {args.module_kwargs.num_pipelines}")

    prepare_all_args(
        args.module_kwargs,
        args.whisper_stt_handler_kwargs,
        args.paraformer_stt_handler_kwargs,
        args.faster_whisper_stt_handler_kwargs,
        args.mlx_audio_whisper_stt_handler_kwargs,
        args.parakeet_tdt_stt_handler_kwargs,
        args.language_model_handler_kwargs,
        args.responses_api_language_model_handler_kwargs,
        args.vision_resolver_kwargs,
        args.chat_tts_handler_kwargs,
        args.facebook_mms_tts_handler_kwargs,
        args.pocket_tts_handler_kwargs,
        args.kokoro_tts_handler_kwargs,
        args.qwen3_tts_handler_kwargs,
    )

    # Validate after prepare_all_args(): --local_mac_optimal_settings mutates
    # module_kwargs.mode to "local", so checking before would let
    # --local_mac_optimal_settings --num_pipelines 2 sneak past this guard.
    if args.module_kwargs.num_pipelines > 1 and args.module_kwargs.mode != "realtime":
        raise ValueError(
            f"--num_pipelines > 1 is only supported with --mode realtime "
            f"(got mode={args.module_kwargs.mode!r}, num_pipelines={args.module_kwargs.num_pipelines})"
        )

    # On Apple Silicon, all MLX inference serializes through a global lock (utils/mlx_lock.py).
    # The progressive STT path uses a short timeout and drops work under contention, producing
    # a flood of warnings without affecting final transcripts. With a pool, pre-emptively turn
    # it off so logs stay readable; the final STT path is unaffected. Non-darwin platforms
    # don't share this lock, so leave their live transcription alone.
    if args.module_kwargs.num_pipelines > 1 and platform == "darwin" and args.module_kwargs.enable_live_transcription:
        logger.info(
            "MLX contention: --num_pipelines=%d > 1 on Apple Silicon → disabling live transcription "
            "(progressive STT contends on the global MLX lock)",
            args.module_kwargs.num_pipelines,
        )
        args.module_kwargs.enable_live_transcription = False

    queues_and_events = initialize_queues_and_events()

    pipeline_manager = build_pipeline(
        args.module_kwargs,
        args.socket_receiver_kwargs,
        args.socket_sender_kwargs,
        args.websocket_streamer_kwargs,
        args.vad_handler_kwargs,
        args.whisper_stt_handler_kwargs,
        args.faster_whisper_stt_handler_kwargs,
        args.paraformer_stt_handler_kwargs,
        args.mlx_audio_whisper_stt_handler_kwargs,
        args.parakeet_tdt_stt_handler_kwargs,
        args.language_model_handler_kwargs,
        args.responses_api_language_model_handler_kwargs,
        args.vision_resolver_kwargs,
        args.chat_tts_handler_kwargs,
        args.facebook_mms_tts_handler_kwargs,
        args.pocket_tts_handler_kwargs,
        args.kokoro_tts_handler_kwargs,
        args.qwen3_tts_handler_kwargs,
        queues_and_events,
    )

    # Set up graceful shutdown handler
    shutdown_requested = [False]  # Use list for nonlocal mutation

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


def main() -> None:
    """Compatibility entry point for direct module execution."""

    from speech_to_speech.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
