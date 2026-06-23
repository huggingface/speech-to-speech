import argparse
import json
import logging
import os
import signal
import sys
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from sys import platform
from threading import Event
from types import FrameType
from typing import Any, Optional

import nltk
import torch
from openai.types.realtime import RealtimeSessionCreateRequest
from rich.console import Console
from transformers import HfArgumentParser

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.arguments_classes.chat_completions_language_model_arguments import (
    ChatCompletionsLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.chat_tts_arguments import ChatTTSHandlerArguments
from speech_to_speech.arguments_classes.facebookmms_tts_arguments import FacebookMMSTTSHandlerArguments
from speech_to_speech.arguments_classes.faster_whisper_stt_arguments import (
    FasterWhisperSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
from speech_to_speech.arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from speech_to_speech.arguments_classes.mlx_audio_whisper_arguments import (
    MLXAudioWhisperSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.paraformer_stt_arguments import ParaformerSTTHandlerArguments
from speech_to_speech.arguments_classes.parakeet_tdt_arguments import (
    ParakeetTDTSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.pocket_tts_arguments import PocketTTSHandlerArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from speech_to_speech.arguments_classes.socket_sender_arguments import SocketSenderArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.arguments_classes.websocket_streamer_arguments import WebSocketStreamerArguments
from speech_to_speech.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut, STTIn, STTOut, TTSIn, TTSOut
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


@dataclass
class ParsedArguments:
    module_kwargs: ModuleArguments
    socket_receiver_kwargs: SocketReceiverArguments
    socket_sender_kwargs: SocketSenderArguments
    websocket_streamer_kwargs: WebSocketStreamerArguments
    vad_handler_kwargs: VADHandlerArguments
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments
    language_model_handler_kwargs: LanguageModelHandlerArguments
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments
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


def parse_arguments() -> ParsedArguments:
    # Pre-parse to determine which LM backend is selected, so only one of the
    # mutually exclusive LM argument classes is registered with HfArgumentParser
    # (avoids duplicate field names from the shared LanguageModelBaseArguments base).
    # chat-completions reuses the responses-api connection fields via subclassing.
    _backend_lm_class = {
        "responses-api": ResponsesApiLanguageModelHandlerArguments,
        "chat-completions": ChatCompletionsLanguageModelHandlerArguments,
    }
    _is_json = len(sys.argv) == 2 and sys.argv[1].endswith(".json")
    if _is_json:
        with open(sys.argv[1]) as _f:
            _backend = json.load(_f).get("llm_backend")
    else:
        _pre = argparse.ArgumentParser(add_help=False)
        _pre.add_argument("--llm_backend", default="responses-api")
        _backend = _pre.parse_known_args()[0].llm_backend

    _lm_class = _backend_lm_class.get(_backend, LanguageModelHandlerArguments)
    logger.debug("LLM backend pre-parse: backend=%s, registering %s", _backend, _lm_class.__name__)

    parser = HfArgumentParser(
        (  # type: ignore[arg-type]
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            WebSocketStreamerArguments,
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            ParaformerSTTHandlerArguments,
            FasterWhisperSTTHandlerArguments,
            MLXAudioWhisperSTTHandlerArguments,
            ParakeetTDTSTTHandlerArguments,
            _lm_class,
            ChatTTSHandlerArguments,
            FacebookMMSTTSHandlerArguments,
            PocketTTSHandlerArguments,
            KokoroTTSHandlerArguments,
            Qwen3TTSHandlerArguments,
        )
    )

    if _is_json:
        parsed = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        parsed = parser.parse_args_into_dataclasses()

    # Build a {type: instance} lookup so field assignment is order-independent.
    by_type: dict[type, Any] = {type(obj): obj for obj in parsed}
    logger.debug("Parsed %d argument classes: %s", len(by_type), [t.__name__ for t in by_type])

    return ParsedArguments(
        module_kwargs=by_type[ModuleArguments],
        socket_receiver_kwargs=by_type[SocketReceiverArguments],
        socket_sender_kwargs=by_type[SocketSenderArguments],
        websocket_streamer_kwargs=by_type[WebSocketStreamerArguments],
        vad_handler_kwargs=by_type[VADHandlerArguments],
        whisper_stt_handler_kwargs=by_type[WhisperSTTHandlerArguments],
        paraformer_stt_handler_kwargs=by_type[ParaformerSTTHandlerArguments],
        faster_whisper_stt_handler_kwargs=by_type[FasterWhisperSTTHandlerArguments],
        mlx_audio_whisper_stt_handler_kwargs=by_type[MLXAudioWhisperSTTHandlerArguments],
        parakeet_tdt_stt_handler_kwargs=by_type[ParakeetTDTSTTHandlerArguments],
        language_model_handler_kwargs=by_type.get(LanguageModelHandlerArguments, LanguageModelHandlerArguments()),
        # The OpenAI-compatible slot holds whichever class was registered:
        # ChatCompletions... (a subclass) for chat-completions, else ResponsesApi....
        responses_api_language_model_handler_kwargs=by_type.get(
            ChatCompletionsLanguageModelHandlerArguments,
            by_type.get(ResponsesApiLanguageModelHandlerArguments, ResponsesApiLanguageModelHandlerArguments()),
        ),
        chat_tts_handler_kwargs=by_type[ChatTTSHandlerArguments],
        facebook_mms_tts_handler_kwargs=by_type[FacebookMMSTTSHandlerArguments],
        pocket_tts_handler_kwargs=by_type[PocketTTSHandlerArguments],
        kokoro_tts_handler_kwargs=by_type[KokoroTTSHandlerArguments],
        qwen3_tts_handler_kwargs=by_type[Qwen3TTSHandlerArguments],
    )


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


MLX_DEFAULT_LM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-bf16"
TRANSFORMERS_DEFAULT_LM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def optimal_mac_settings(mac_optimal_settings: bool, *handler_kwargs: Any) -> None:
    if mac_optimal_settings:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "device"):
                kwargs.device = "mps"
            if hasattr(kwargs, "mode"):
                kwargs.mode = "local"
            if hasattr(kwargs, "stt"):
                kwargs.stt = "parakeet-tdt"
            if hasattr(kwargs, "llm_backend"):
                kwargs.llm_backend = "mlx-lm"
            if hasattr(kwargs, "tts"):
                kwargs.tts = "qwen3"
            if hasattr(kwargs, "model_name"):
                if kwargs.model_name == TRANSFORMERS_DEFAULT_LM_MODEL:
                    kwargs.model_name = MLX_DEFAULT_LM_MODEL


def check_mac_settings(module_kwargs: ModuleArguments) -> None:
    if platform == "darwin":
        if module_kwargs.device == "cuda":
            raise ValueError("Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'.")
        if module_kwargs.llm_backend != "mlx-lm":
            logger.warning(
                "For macOS users, it is recommended to use mlx-lm. You can activate it by passing --llm_backend mlx-lm."
            )
        if module_kwargs.tts not in ("pocket", "kokoro", "qwen3"):
            logger.warning(
                "For macOS users, it is recommended to use qwen3 for TTS (pocket and kokoro are also valid options)."
            )


def overwrite_device_argument(common_device: Optional[str], *handler_kwargs: Any) -> None:
    if common_device:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "llm_device"):
                kwargs.llm_device = common_device
            if hasattr(kwargs, "tts_device"):
                kwargs.tts_device = common_device
            if hasattr(kwargs, "stt_device"):
                kwargs.stt_device = common_device
            if hasattr(kwargs, "paraformer_stt_device"):
                kwargs.paraformer_stt_device = common_device
            if hasattr(kwargs, "facebook_mms_device"):
                kwargs.facebook_mms_device = common_device
            if hasattr(kwargs, "qwen3_tts_device"):
                kwargs.qwen3_tts_device = common_device


def prepare_module_args(module_kwargs: ModuleArguments, *handler_kwargs: Any) -> None:
    optimal_mac_settings(module_kwargs.local_mac_optimal_settings, module_kwargs, *handler_kwargs)
    if module_kwargs.tts is None:
        module_kwargs.tts = "qwen3"
    if platform == "darwin":
        check_mac_settings(module_kwargs)
    overwrite_device_argument(module_kwargs.device, *handler_kwargs)


def prepare_all_args(
    module_kwargs: ModuleArguments,
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
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


def _build_pipeline_handlers(
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
    text_output_queue: Optional[Queue[TextEventItem]],
    transcription_notifier_setup: dict[str, Any],
    module_kwargs: ModuleArguments,
    vad_handler_kwargs: VADHandlerArguments,
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
    speculative_turns: SpeculativeTurnTracker | None = None,
) -> list[Any]:
    """Build the shared handler chain: VAD → STT → TranscriptionNotifier → LM → LMOutputProcessor → TTS.

    Callers own the queues, events, and any per-mode mutations to handler kwargs (cancel_scope,
    text_output_queue, live transcription flags). `transcription_notifier_setup` differs by mode:
    realtime mode passes a service-driven setup; non-realtime modes inject a RuntimeConfig.
    """
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )

    transcription_notifier = TranscriptionNotifier(
        stop_event,
        queue_in=stt_output_queue,
        queue_out=text_prompt_queue,  # type: ignore[arg-type]
        setup_kwargs=transcription_notifier_setup,
    )

    stt = get_stt_handler(
        module_kwargs,
        stop_event,
        spoken_prompt_queue,
        stt_output_queue,
        speculative_turns,
        whisper_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
        mlx_audio_whisper_stt_handler_kwargs,
        parakeet_tdt_stt_handler_kwargs,
    )

    lm = get_llm_handler(
        module_kwargs,
        stop_event,
        text_prompt_queue,
        lm_response_queue,
        language_model_handler_kwargs,
        responses_api_language_model_handler_kwargs,
    )

    lm_processor = LMOutputProcessor(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=lm_processed_queue,
        setup_kwargs={"text_output_queue": text_output_queue, "speculative_turns": speculative_turns},
    )

    tts = get_tts_handler(
        module_kwargs,
        stop_event,
        lm_processed_queue,
        send_audio_chunks_queue,
        should_listen,
        chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs,
        pocket_tts_handler_kwargs,
        kokoro_tts_handler_kwargs,
        qwen3_tts_handler_kwargs,
    )

    return [vad, stt, transcription_notifier, lm, lm_processor, tts]


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
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
) -> "PipelineUnit":
    """Build one isolated realtime pipeline (own queues, events, service, handlers).

    Returns a PipelineUnit that lives inside RealtimeServer's pool. Handler instances
    are returned in `unit.handlers`; the caller threads them via ThreadManager. No uvicorn
    is created here — the single server is owned by RealtimeServer.
    """
    from speech_to_speech.api.openai_realtime.service import RealtimeService

    # Per-unit deep copies so mutations (cancel_scope, text_output_queue) don't bleed across pipelines.
    vad_kw = deepcopy(vad_handler_kwargs)
    whisper_kw = deepcopy(whisper_stt_handler_kwargs)
    faster_whisper_kw = deepcopy(faster_whisper_stt_handler_kwargs)
    paraformer_kw = deepcopy(paraformer_stt_handler_kwargs)
    mlx_audio_whisper_kw = deepcopy(mlx_audio_whisper_stt_handler_kwargs)
    parakeet_kw = deepcopy(parakeet_tdt_stt_handler_kwargs)
    lm_kw = deepcopy(language_model_handler_kwargs)
    responses_api_kw = deepcopy(responses_api_language_model_handler_kwargs)
    chat_tts_kw = deepcopy(chat_tts_handler_kwargs)
    facebook_mms_kw = deepcopy(facebook_mms_tts_handler_kwargs)
    pocket_tts_kw = deepcopy(pocket_tts_handler_kwargs)
    kokoro_tts_kw = deepcopy(kokoro_tts_handler_kwargs)
    qwen3_tts_kw = deepcopy(qwen3_tts_handler_kwargs)

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

    vars(vad_kw)["text_output_queue"] = text_output_queue
    vars(vad_kw)["speculative_turns"] = speculative_turns
    for kw in (
        lm_kw,
        responses_api_kw,
        kokoro_tts_kw,
        qwen3_tts_kw,
        pocket_tts_kw,
        chat_tts_kw,
        facebook_mms_kw,
    ):
        vars(kw)["cancel_scope"] = cancel_scope
        vars(kw)["speculative_turns"] = speculative_turns

    if module_kwargs.llm_backend in ("responses-api", "chat-completions"):
        chat_size = vars(responses_api_kw).get("chat_size", 10)
    else:
        chat_size = vars(lm_kw).get("chat_size", 10)

    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
        chat_size=chat_size,
        speculative_turns=speculative_turns,
    )

    if module_kwargs.enable_live_transcription:
        vad_kw.enable_realtime_transcription = True
        vad_kw.realtime_processing_pause = module_kwargs.live_transcription_update_interval

    handlers = _build_pipeline_handlers(
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
        transcription_notifier_setup={
            "text_output_queue": text_output_queue,
            "should_listen": should_listen,
        },
        module_kwargs=module_kwargs,
        vad_handler_kwargs=vad_kw,
        whisper_stt_handler_kwargs=whisper_kw,
        faster_whisper_stt_handler_kwargs=faster_whisper_kw,
        paraformer_stt_handler_kwargs=paraformer_kw,
        mlx_audio_whisper_stt_handler_kwargs=mlx_audio_whisper_kw,
        parakeet_tdt_stt_handler_kwargs=parakeet_kw,
        language_model_handler_kwargs=lm_kw,
        responses_api_language_model_handler_kwargs=responses_api_kw,
        chat_tts_handler_kwargs=chat_tts_kw,
        facebook_mms_tts_handler_kwargs=facebook_mms_kw,
        pocket_tts_handler_kwargs=pocket_tts_kw,
        kokoro_tts_handler_kwargs=kokoro_tts_kw,
        qwen3_tts_handler_kwargs=qwen3_tts_kw,
        speculative_turns=speculative_turns,
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
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
    queues_and_events: dict[str, Any],
) -> ThreadManager:
    stop_event = queues_and_events["stop_event"]
    should_listen = queues_and_events["should_listen"]
    recv_audio_chunks_queue = queues_and_events["recv_audio_chunks_queue"]
    send_audio_chunks_queue = queues_and_events["send_audio_chunks_queue"]
    spoken_prompt_queue = queues_and_events["spoken_prompt_queue"]
    stt_output_queue = queues_and_events["stt_output_queue"]
    text_prompt_queue = queues_and_events["text_prompt_queue"]
    lm_response_queue = queues_and_events["lm_response_queue"]
    lm_processed_queue = queues_and_events["lm_processed_queue"]
    text_output_queue = (
        None  # Only set for websocket/realtime modes; kept None otherwise to avoid unbounded queue growth
    )

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
                chat_tts_handler_kwargs=chat_tts_handler_kwargs,
                facebook_mms_tts_handler_kwargs=facebook_mms_tts_handler_kwargs,
                pocket_tts_handler_kwargs=pocket_tts_handler_kwargs,
                kokoro_tts_handler_kwargs=kokoro_tts_handler_kwargs,
                qwen3_tts_handler_kwargs=qwen3_tts_handler_kwargs,
            )
            for i in range(pool_size)
        ]

        realtime_server = RealtimeServer(
            stop_event=stop_event,
            pool=pool,
            host=websocket_streamer_kwargs.ws_host,
            port=websocket_streamer_kwargs.ws_port,
        )

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
        should_listen=should_listen,
        recv_audio_chunks_queue=recv_audio_chunks_queue,
        spoken_prompt_queue=spoken_prompt_queue,
        stt_output_queue=stt_output_queue,
        text_prompt_queue=text_prompt_queue,
        lm_response_queue=lm_response_queue,
        lm_processed_queue=lm_processed_queue,
        send_audio_chunks_queue=send_audio_chunks_queue,
        text_output_queue=text_output_queue,
        transcription_notifier_setup=transcription_notifier_setup,
        module_kwargs=module_kwargs,
        vad_handler_kwargs=vad_handler_kwargs,
        whisper_stt_handler_kwargs=whisper_stt_handler_kwargs,
        faster_whisper_stt_handler_kwargs=faster_whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs=paraformer_stt_handler_kwargs,
        mlx_audio_whisper_stt_handler_kwargs=mlx_audio_whisper_stt_handler_kwargs,
        parakeet_tdt_stt_handler_kwargs=parakeet_tdt_stt_handler_kwargs,
        language_model_handler_kwargs=language_model_handler_kwargs,
        responses_api_language_model_handler_kwargs=responses_api_language_model_handler_kwargs,
        chat_tts_handler_kwargs=chat_tts_handler_kwargs,
        facebook_mms_tts_handler_kwargs=facebook_mms_tts_handler_kwargs,
        pocket_tts_handler_kwargs=pocket_tts_handler_kwargs,
        kokoro_tts_handler_kwargs=kokoro_tts_handler_kwargs,
        qwen3_tts_handler_kwargs=qwen3_tts_handler_kwargs,
    )

    return ThreadManager([*comms_handlers, *pipeline_handlers])


def get_stt_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    spoken_prompt_queue: Queue[VADOutItem],
    text_prompt_queue: Queue[STTOutItem],
    speculative_turns: SpeculativeTurnTracker | None,
    whisper_stt_handler_kwargs: WhisperSTTHandlerArguments,
    faster_whisper_stt_handler_kwargs: FasterWhisperSTTHandlerArguments,
    paraformer_stt_handler_kwargs: ParaformerSTTHandlerArguments,
    mlx_audio_whisper_stt_handler_kwargs: MLXAudioWhisperSTTHandlerArguments,
    parakeet_tdt_stt_handler_kwargs: ParakeetTDTSTTHandlerArguments,
) -> BaseHandler[STTIn, STTOut]:
    from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

    def with_speculative_turns(handler: BaseSTTHandler) -> BaseSTTHandler:
        if speculative_turns is not None:
            handler.speculative_turns = speculative_turns
        return handler

    if module_kwargs.stt == "whisper":
        from speech_to_speech.STT.whisper_stt_handler import WhisperSTTHandler

        return with_speculative_turns(
            WhisperSTTHandler(
                stop_event,
                queue_in=spoken_prompt_queue,
                queue_out=text_prompt_queue,
                setup_kwargs=vars(whisper_stt_handler_kwargs),
            )
        )
    elif module_kwargs.stt == "whisper-mlx":
        from speech_to_speech.STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler

        return with_speculative_turns(
            LightningWhisperSTTHandler(
                stop_event,
                queue_in=spoken_prompt_queue,
                queue_out=text_prompt_queue,
                setup_kwargs=vars(whisper_stt_handler_kwargs),
            )
        )
    elif module_kwargs.stt == "mlx-audio-whisper":
        from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler

        # Merge MLX Audio Whisper kwargs with shared language parameter from Whisper kwargs
        setup_kwargs = {**vars(mlx_audio_whisper_stt_handler_kwargs), "language": whisper_stt_handler_kwargs.language}
        return with_speculative_turns(
            MLXAudioWhisperSTTHandler(
                stop_event,
                queue_in=spoken_prompt_queue,
                queue_out=text_prompt_queue,
                setup_kwargs=setup_kwargs,
            )
        )
    elif module_kwargs.stt == "paraformer":
        from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler

        return with_speculative_turns(
            ParaformerSTTHandler(
                stop_event,
                queue_in=spoken_prompt_queue,
                queue_out=text_prompt_queue,
                setup_kwargs=vars(paraformer_stt_handler_kwargs),
            )
        )
    elif module_kwargs.stt == "faster-whisper":
        from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler

        return with_speculative_turns(
            FasterWhisperSTTHandler(
                stop_event,
                queue_in=spoken_prompt_queue,
                queue_out=text_prompt_queue,
                setup_kwargs=vars(faster_whisper_stt_handler_kwargs),
            )
        )
    elif module_kwargs.stt == "parakeet-tdt":
        from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler

        # Add live transcription parameters to setup_kwargs
        setup_kwargs = {
            **vars(parakeet_tdt_stt_handler_kwargs),
            "enable_live_transcription": module_kwargs.enable_live_transcription,
            "live_transcription_update_interval": module_kwargs.live_transcription_update_interval,
        }

        return with_speculative_turns(
            ParakeetTDTSTTHandler(
                stop_event,
                queue_in=spoken_prompt_queue,
                queue_out=text_prompt_queue,
                setup_kwargs=setup_kwargs,
            )
        )
    else:
        raise ValueError(
            "The STT should be either whisper, whisper-mlx, mlx-audio-whisper, faster-whisper, parakeet-tdt, or paraformer."
        )


def get_llm_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    text_prompt_queue: Queue[TextPromptItem],
    lm_response_queue: Queue[LMOutItem],
    language_model_handler_kwargs: LanguageModelHandlerArguments,
    responses_api_language_model_handler_kwargs: ResponsesApiLanguageModelHandlerArguments,
) -> BaseHandler[LLMIn, LLMOut]:
    if module_kwargs.llm_backend == "responses-api":
        from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler

        return ResponsesApiModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(responses_api_language_model_handler_kwargs),
        )

    if module_kwargs.llm_backend == "chat-completions":
        from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

        # Reuses the responses-api argument class (identical fields: model_name,
        # base_url, api_key, stream, disable_thinking, ...).
        return ChatCompletionsApiModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=vars(responses_api_language_model_handler_kwargs),
        )

    if module_kwargs.llm_backend in ("transformers", "mlx-lm"):
        lm_kwargs = vars(language_model_handler_kwargs)
        is_vlm = lm_kwargs.pop("is_vlm", False)
        if module_kwargs.llm_backend == "mlx-lm":
            lm_kwargs["backend"] = "mlx"

        if is_vlm:
            from speech_to_speech.LLM.language_model import VisionLanguageModelHandler

            return VisionLanguageModelHandler(
                stop_event,
                queue_in=text_prompt_queue,
                queue_out=lm_response_queue,
                setup_kwargs=lm_kwargs,
            )
        from speech_to_speech.LLM.language_model import LanguageModelHandler

        return LanguageModelHandler(
            stop_event,
            queue_in=text_prompt_queue,
            queue_out=lm_response_queue,
            setup_kwargs=lm_kwargs,
        )

    raise ValueError("The LLM should be either transformers, mlx-lm, responses-api or chat-completions")


def get_tts_handler(
    module_kwargs: ModuleArguments,
    stop_event: Event,
    lm_response_queue: Queue[TTSInItem],
    send_audio_chunks_queue: Queue[AudioOutItem],
    should_listen: Event,
    chat_tts_handler_kwargs: ChatTTSHandlerArguments,
    facebook_mms_tts_handler_kwargs: FacebookMMSTTSHandlerArguments,
    pocket_tts_handler_kwargs: PocketTTSHandlerArguments,
    kokoro_tts_handler_kwargs: KokoroTTSHandlerArguments,
    qwen3_tts_handler_kwargs: Qwen3TTSHandlerArguments,
) -> BaseHandler[TTSIn, TTSOut]:
    if module_kwargs.tts == "chatTTS":
        try:
            from speech_to_speech.TTS.chatTTS_handler import ChatTTSHandler
        except (ImportError, RuntimeError) as e:
            logger.error('Error importing ChatTTSHandler. Install it with `pip install "speech-to-speech[chattts]"`.')
            raise e
        return ChatTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(chat_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "facebookMMS":
        from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler

        return FacebookMMSTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(facebook_mms_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "pocket":
        try:
            from speech_to_speech.TTS.pocket_tts_handler import PocketTTSHandler
        except ImportError as e:
            raise ImportError(
                'Pocket TTS is optional. Install it with `pip install "speech-to-speech[pocket]"`.'
            ) from e

        return PocketTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(pocket_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "kokoro":
        try:
            from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
        except ImportError as e:
            raise ImportError('Kokoro is optional. Install it with `pip install "speech-to-speech[kokoro]"`.') from e

        return KokoroTTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(kokoro_tts_handler_kwargs),
        )
    elif module_kwargs.tts == "qwen3":
        from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

        return Qwen3TTSHandler(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=send_audio_chunks_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(qwen3_tts_handler_kwargs),
        )
    else:
        raise ValueError("The TTS should be either chatTTS, facebookMMS, pocket, kokoro, or qwen3")


def main() -> None:
    args = parse_arguments()

    setup_logger(args.module_kwargs.log_level)

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


if __name__ == "__main__":
    main()
