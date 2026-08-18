from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from importlib import import_module
from queue import Queue
from threading import Event
from typing import Any, Literal

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
from speech_to_speech.arguments_classes.omnivoice_tts_arguments import OmniVoiceTTSHandlerArguments
from speech_to_speech.arguments_classes.openai_stt_arguments import OpenAICompatibleSTTHandlerArguments
from speech_to_speech.arguments_classes.openai_tts_arguments import OpenAICompatibleTTSHandlerArguments
from speech_to_speech.arguments_classes.paraformer_stt_arguments import ParaformerSTTHandlerArguments
from speech_to_speech.arguments_classes.parakeet_tdt_arguments import (
    ParakeetTDTSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.pocket_tts_arguments import PocketTTSHandlerArguments
from speech_to_speech.arguments_classes.qwen3_asr_stt_arguments import Qwen3ASRSTTHandlerArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.supertonic_tts_arguments import SupertonicTTSHandlerArguments
from speech_to_speech.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

BackendKind = Literal["stt", "llm", "tts"]
BackendConfig = dict[str, Any]

logger = logging.getLogger(__name__)


@dataclass
class EmptyBackendArguments:
    """Argument placeholder for a backend with no backend-specific options."""


@dataclass(frozen=True)
class BackendCapabilities:
    """Construction details that affect stage composition, not backend selection."""

    bypasses_transcription_notifier: bool = False
    supports_audio_input: bool = False
    supports_llm_proxy: bool = False


@dataclass(frozen=True)
class HandlerContext:
    """Pipeline-local state made available to every registered factory."""

    stop_event: Event
    queue_in: Queue[Any]
    queue_out: Queue[Any]
    text_output_queue: Queue[Any]
    should_listen: Event
    cancel_scope: CancelScope
    speculative_turns: SpeculativeTurnTracker
    pipeline_index: int
    sample_rate: int
    enable_live_transcription: bool
    live_transcription_update_interval: float


HandlerFactory = Callable[[HandlerContext, Mapping[str, Any]], Any]
ConfigNormalizer = Callable[[Any], BackendConfig]


@dataclass(frozen=True)
class BackendSpec:
    """Static construction metadata for one built-in backend."""

    name: str
    kind: BackendKind
    config_type: type[Any]
    create_handler: HandlerFactory
    config_prefix: str | None = None
    normalize_config: ConfigNormalizer | None = None
    required_extra: str | None = None
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    def normalize(self, config: Any) -> BackendConfig:
        if not isinstance(config, self.config_type):
            raise TypeError(f"Backend {self.name!r} expects {self.config_type.__name__}, got {type(config).__name__}.")
        normalizer = self.normalize_config
        if normalizer is not None:
            return normalizer(config)
        return normalize_dataclass_config(config, self.config_prefix)


@dataclass(frozen=True)
class BackendSelection:
    """A selected registry entry and its normalized configuration."""

    spec: BackendSpec
    config: BackendConfig

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def kind(self) -> BackendKind:
        return self.spec.kind

    def copy_for_pipeline(self) -> BackendSelection:
        return BackendSelection(self.spec, deepcopy(self.config))


def normalize_dataclass_config(config: Any, prefix: str | None = None) -> BackendConfig:
    """Return handler kwargs without mutating the parsed argument dataclass."""

    normalized: BackendConfig = {}
    generation: BackendConfig = {}
    marker = f"{prefix}_" if prefix else None

    for config_field in fields(config):
        name = config_field.name
        value = deepcopy(getattr(config, name))
        if marker and name.startswith(marker):
            name = name[len(marker) :]
        if name == "gen_kwargs" and isinstance(value, Mapping):
            generation.update(value)
        elif name.startswith("gen_"):
            generation[name[len("gen_") :]] = value
        else:
            normalized[name] = value

    normalized["gen_kwargs"] = generation
    return normalized


def _normalize_facebook_mms_config(config: Any) -> BackendConfig:
    normalized = normalize_dataclass_config(config, "facebook_mms")
    normalized["language"] = normalized.pop("tts_language")
    return normalized


def build_backend_registry(kind: BackendKind, specs: Iterable[BackendSpec]) -> dict[str, BackendSpec]:
    """Build an explicit registry and reject ambiguous entries eagerly."""

    registry: dict[str, BackendSpec] = {}
    for spec in specs:
        if spec.kind != kind:
            raise ValueError(f"Backend {spec.name!r} has kind {spec.kind!r}; expected {kind!r}.")
        if spec.name in registry:
            raise ValueError(f"Duplicate {kind} backend name: {spec.name!r}.")
        registry[spec.name] = spec
    return registry


def select_backend(registry: Mapping[str, BackendSpec], name: str, config: Any) -> BackendSelection:
    try:
        spec = registry[name]
    except KeyError as exc:
        choices = ", ".join(registry)
        raise ValueError(f"Unsupported backend {name!r}; choose one of: {choices}.") from exc
    return BackendSelection(spec, spec.normalize(config))


def _optional_dependency_error(selection: BackendSelection, exc: BaseException) -> ImportError | None:
    extra = selection.spec.required_extra
    if extra is None:
        return None
    return ImportError(
        f"Backend {selection.name!r} ({selection.kind}) requires optional dependencies. "
        f'Install them with `pip install "speech-to-speech[{extra}]"`. '
        f"Original import error: {exc}"
    )


def create_backend_handler(selection: BackendSelection, context: HandlerContext) -> Any:
    """Construct a handler and turn lazy dependency failures into actionable errors."""

    try:
        return selection.spec.create_handler(context, selection.config)
    except ImportError as exc:
        dependency_error = _optional_dependency_error(selection, exc)
        if dependency_error is None:
            raise
        raise dependency_error from exc


def _load_handler(module_name: str, class_name: str) -> type[Any]:
    try:
        module = import_module(module_name)
    except RuntimeError as exc:
        # Some optional packages fail at import time with RuntimeError rather
        # than ImportError when a native dependency is unavailable.
        raise ImportError(f"Could not import backend module {module_name!r}: {exc}") from exc
    return getattr(module, class_name)


def _simple_handler_factory(
    module_name: str,
    class_name: str,
    *,
    setup_should_listen: bool = False,
    attach_speculative_turns: bool = False,
    context_kwargs: bool = False,
) -> HandlerFactory:
    def create(context: HandlerContext, config: Mapping[str, Any]) -> Any:
        handler_class = _load_handler(module_name, class_name)
        setup_kwargs = dict(config)
        if context_kwargs:
            setup_kwargs.update(
                cancel_scope=context.cancel_scope,
                speculative_turns=context.speculative_turns,
            )
        handler = handler_class(
            context.stop_event,
            queue_in=context.queue_in,
            queue_out=context.queue_out,
            setup_args=(context.should_listen,) if setup_should_listen else (),
            setup_kwargs=setup_kwargs,
        )
        if attach_speculative_turns:
            handler.speculative_turns = context.speculative_turns
        return handler

    return create


def _create_audio_input(context: HandlerContext, _config: Mapping[str, Any]) -> Any:
    handler_class = _load_handler("speech_to_speech.LLM.audio_input_notifier", "AudioInputNotifier")
    return handler_class(
        context.stop_event,
        queue_in=context.queue_in,
        queue_out=context.queue_out,
        setup_kwargs={
            "sample_rate": context.sample_rate,
            "speculative_turns": context.speculative_turns,
            "text_output_queue": context.text_output_queue,
        },
    )


def _create_parakeet(context: HandlerContext, config: Mapping[str, Any]) -> Any:
    handler_class = _load_handler("speech_to_speech.STT.parakeet_tdt_handler", "ParakeetTDTSTTHandler")
    setup_kwargs = {
        **config,
        "enable_live_transcription": context.enable_live_transcription,
        "live_transcription_update_interval": context.live_transcription_update_interval,
    }
    handler = handler_class(
        context.stop_event,
        queue_in=context.queue_in,
        queue_out=context.queue_out,
        setup_kwargs=setup_kwargs,
    )
    handler.speculative_turns = context.speculative_turns
    return handler


def _create_openai_stt(context: HandlerContext, config: Mapping[str, Any]) -> Any:
    from speech_to_speech.STT.endpoint_admission import (
        EndpointAdmissionRegistry,
        EndpointAdmissionSettings,
    )

    handler_class = _load_handler(
        "speech_to_speech.STT.openai_compatible_handler",
        "OpenAICompatibleSTTHandler",
    )
    setup_kwargs = dict(config)
    configured_api_key = setup_kwargs.get("api_key")
    base_url = setup_kwargs["base_url"].rstrip("/")
    resolved_api_key = configured_api_key
    if resolved_api_key is None and base_url == "https://api.openai.com/v1":
        resolved_api_key = os.getenv("OPENAI_API_KEY")
    lease = EndpointAdmissionRegistry.acquire(
        base_url=setup_kwargs["base_url"],
        api_key=resolved_api_key,
        settings=EndpointAdmissionSettings(
            max_concurrency=setup_kwargs["max_concurrency"],
            max_queue_size=setup_kwargs["max_queue_size"],
            progressive_min_interval_s=setup_kwargs["progressive_min_interval"],
        ),
    )
    setup_kwargs.update(
        admission_lease=lease,
        speculative_turns=context.speculative_turns,
    )
    try:
        return handler_class(
            context.stop_event,
            queue_in=context.queue_in,
            queue_out=context.queue_out,
            setup_kwargs=setup_kwargs,
        )
    except Exception:
        lease.release()
        raise


def _create_openai_tts(context: HandlerContext, config: Mapping[str, Any]) -> Any:
    handler_class = _load_handler(
        "speech_to_speech.TTS.openai_compatible_handler",
        "OpenAICompatibleTTSHandler",
    )
    return handler_class(
        context.stop_event,
        queue_in=context.queue_in,
        queue_out=context.queue_out,
        setup_args=(context.should_listen,),
        setup_kwargs={
            **config,
            "cancel_scope": context.cancel_scope,
            "speculative_turns": context.speculative_turns,
        },
    )


def _create_local_llm(backend: Literal["transformers", "mlx-lm"]) -> HandlerFactory:
    def create(context: HandlerContext, config: Mapping[str, Any]) -> Any:
        setup_kwargs = dict(config)
        is_vlm = setup_kwargs.pop("is_vlm", False)
        if backend == "mlx-lm":
            setup_kwargs["backend"] = "mlx"
        setup_kwargs.update(
            cancel_scope=context.cancel_scope,
            speculative_turns=context.speculative_turns,
        )
        class_name = "VisionLanguageModelHandler" if is_vlm else "LanguageModelHandler"
        handler_class = _load_handler("speech_to_speech.LLM.language_model", class_name)
        return handler_class(
            context.stop_event,
            queue_in=context.queue_in,
            queue_out=context.queue_out,
            setup_kwargs=setup_kwargs,
        )

    return create


STT_BACKENDS = build_backend_registry(
    "stt",
    [
        BackendSpec(
            "none",
            "stt",
            EmptyBackendArguments,
            _create_audio_input,
            capabilities=BackendCapabilities(bypasses_transcription_notifier=True),
        ),
        BackendSpec(
            "whisper",
            "stt",
            WhisperSTTHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.STT.whisper_stt_handler",
                "WhisperSTTHandler",
                attach_speculative_turns=True,
            ),
            config_prefix="stt",
        ),
        BackendSpec(
            "whisper-mlx",
            "stt",
            WhisperSTTHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.STT.lightning_whisper_mlx_handler",
                "LightningWhisperSTTHandler",
                attach_speculative_turns=True,
            ),
            config_prefix="stt",
            required_extra="whisper-mlx",
        ),
        BackendSpec(
            "mlx-audio-whisper",
            "stt",
            MLXAudioWhisperSTTHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.STT.mlx_audio_whisper_handler",
                "MLXAudioWhisperSTTHandler",
                attach_speculative_turns=True,
            ),
            config_prefix="mlx_audio_whisper",
        ),
        BackendSpec(
            "faster-whisper",
            "stt",
            FasterWhisperSTTHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.STT.faster_whisper_handler",
                "FasterWhisperSTTHandler",
                attach_speculative_turns=True,
            ),
            config_prefix="faster_whisper_stt",
            required_extra="faster-whisper",
        ),
        BackendSpec(
            "parakeet-tdt",
            "stt",
            ParakeetTDTSTTHandlerArguments,
            _create_parakeet,
            config_prefix="parakeet_tdt",
        ),
        BackendSpec(
            "paraformer",
            "stt",
            ParaformerSTTHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.STT.paraformer_handler",
                "ParaformerSTTHandler",
                attach_speculative_turns=True,
            ),
            config_prefix="paraformer_stt",
            required_extra="paraformer",
        ),
        BackendSpec(
            "qwen3-asr",
            "stt",
            Qwen3ASRSTTHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.STT.qwen3_asr_handler",
                "Qwen3ASRSTTHandler",
                attach_speculative_turns=True,
            ),
            config_prefix="qwen3_asr",
        ),
        BackendSpec(
            "openai",
            "stt",
            OpenAICompatibleSTTHandlerArguments,
            _create_openai_stt,
            config_prefix="openai_stt",
        ),
    ],
)

LLM_BACKENDS = build_backend_registry(
    "llm",
    [
        BackendSpec(
            "transformers",
            "llm",
            LanguageModelHandlerArguments,
            _create_local_llm("transformers"),
            config_prefix="llm",
        ),
        BackendSpec(
            "mlx-lm",
            "llm",
            LanguageModelHandlerArguments,
            _create_local_llm("mlx-lm"),
            config_prefix="llm",
            required_extra="mlx-lm",
        ),
        BackendSpec(
            "responses-api",
            "llm",
            ResponsesApiLanguageModelHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.LLM.responses_api_language_model",
                "ResponsesApiModelHandler",
                context_kwargs=True,
            ),
            config_prefix="responses_api",
            capabilities=BackendCapabilities(supports_llm_proxy=True),
        ),
        BackendSpec(
            "chat-completions",
            "llm",
            ChatCompletionsLanguageModelHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.LLM.chat_completions_language_model",
                "ChatCompletionsApiModelHandler",
                context_kwargs=True,
            ),
            config_prefix="responses_api",
            capabilities=BackendCapabilities(supports_audio_input=True, supports_llm_proxy=True),
        ),
    ],
)

TTS_BACKENDS = build_backend_registry(
    "tts",
    [
        BackendSpec(
            "chatTTS",
            "tts",
            ChatTTSHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.TTS.chatTTS_handler",
                "ChatTTSHandler",
                setup_should_listen=True,
                context_kwargs=True,
            ),
            config_prefix="chat_tts",
            required_extra="chattts",
        ),
        BackendSpec(
            "facebookMMS",
            "tts",
            FacebookMMSTTSHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.TTS.facebookmms_handler",
                "FacebookMMSTTSHandler",
                setup_should_listen=True,
                context_kwargs=True,
            ),
            normalize_config=_normalize_facebook_mms_config,
        ),
        BackendSpec(
            "omnivoice",
            "tts",
            OmniVoiceTTSHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.TTS.omnivoice_handler",
                "OmniVoiceTTSHandler",
                setup_should_listen=True,
                context_kwargs=True,
            ),
            config_prefix="omnivoice",
            required_extra="omnivoice",
        ),
        BackendSpec(
            "pocket",
            "tts",
            PocketTTSHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.TTS.pocket_tts_handler",
                "PocketTTSHandler",
                setup_should_listen=True,
                context_kwargs=True,
            ),
            config_prefix="pocket_tts",
            required_extra="pocket",
        ),
        BackendSpec(
            "kokoro",
            "tts",
            KokoroTTSHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.TTS.kokoro_handler",
                "KokoroTTSHandler",
                setup_should_listen=True,
                context_kwargs=True,
            ),
            config_prefix="kokoro",
            required_extra="kokoro",
        ),
        BackendSpec(
            "qwen3",
            "tts",
            Qwen3TTSHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.TTS.qwen3_tts_handler",
                "Qwen3TTSHandler",
                setup_should_listen=True,
                context_kwargs=True,
            ),
            config_prefix="qwen3_tts",
        ),
        BackendSpec(
            "openai",
            "tts",
            OpenAICompatibleTTSHandlerArguments,
            _create_openai_tts,
            config_prefix="openai_tts",
        ),
        BackendSpec(
            "supertonic",
            "tts",
            SupertonicTTSHandlerArguments,
            _simple_handler_factory(
                "speech_to_speech.TTS.supertonic_tts_handler",
                "SupertonicTTSHandler",
                setup_should_listen=True,
                context_kwargs=True,
            ),
            config_prefix="supertonic_tts",
            required_extra="supertonic",
        ),
    ],
)
