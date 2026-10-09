"""Offline configuration APIs. Existing startup interfaces are unchanged."""

from __future__ import annotations

import math
import os
import re
import types
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Union, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from speech_to_speech.backend_registry import BackendSelection

_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_STAGES = ("vad", "stt", "llm", "tts")
_MODULE_PROCESS = {"log_level", "log_transcripts"}
_MODULE_SELECTORS = {"stt", "llm_backend", "tts"}
_MODULE_PIPELINE = {
    "diarization",
    "diarization_model_name",
    "diarization_revision",
    "diarization_device",
    "diarization_dtype",
    "diarization_streaming_mode",
    "diarization_threshold",
    "detect_llm_output_language",
    "device",
    "mac_optimal_settings",
    "enable_live_transcription",
    "live_transcription_update_interval",
    "enable_llm_proxy",
    "llm_proxy_connect_timeout_s",
    "enable_visemes",
    "stv_model_name",
    "stv_device",
}
_PATH_FIELDS = {
    "vad_firered_model_dir",
    "smart_turn_model_path",
    "omnivoice_ref_audio",
    "omnivoice_voice_clone_prompt",
    "omnivoice_ref_voices_dir",
    "qwen3_tts_gguf_talker_path",
    "qwen3_tts_gguf_codec_path",
    "qwen3_tts_ref_cache_dir",
    "qwen3_tts_ref_audio",
    "qwen3_tts_ref_spk",
    "qwen3_tts_ref_rvq",
}


class ConfigurationError(ValueError):
    """A sanitized configuration diagnostic that does not echo input values."""


@dataclass
class ConfigDocument:
    """Unresolved file data and source locations. Data can contain secrets."""

    path: Path
    data: dict[str, Any] = field(repr=False)
    locations: dict[tuple[str | int, ...], tuple[int, int]] = field(default_factory=dict, repr=False)


@dataclass
class ResolvedStage:
    block_id: str = field(repr=False)
    kind: str
    backend: str
    settings: dict[str, Any] = field(repr=False)
    selection: BackendSelection | None = field(default=None, repr=False)


@dataclass
class ResolvedPipeline:
    num_pipelines: int
    stages: dict[str, ResolvedStage] = field(repr=False)
    options: dict[str, Any] = field(repr=False)


@dataclass
class ResolvedConfig:
    runtime: dict[str, Any] = field(repr=False)
    pipelines: dict[str, ResolvedPipeline] = field(repr=False)
    client: dict[str, Any] | None = field(default=None, repr=False)
    sources: dict[tuple[str | int, ...], str] = field(default_factory=dict, repr=False)


def _error(document: ConfigDocument, path: tuple[str | int, ...], message: str) -> NoReturn:
    mark = document.locations.get(path)
    location = str(document.path)
    if mark:
        location += f":{mark[0]}:{mark[1]}"
    # Only schema field names and validated identifiers can appear in diagnostics.
    safe_path = ".".join(str(part) for part in path if isinstance(part, int) or _ID.fullmatch(part))
    raise ConfigurationError(f"{location}: {safe_path or 'document'}: {message}") from None


def load_config(path: str | Path) -> ConfigDocument:
    """Load one bounded YAML file without importing runtime or registry modules."""
    path = Path(path)
    expansion_failure = False
    try:
        path = path.expanduser().absolute()
    except RuntimeError:
        expansion_failure = True
    if expansion_failure:
        raise ConfigurationError("Configuration file: Cannot expand the home directory; provide an absolute path.")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        guidance = (
            "Use the existing flat JSON interface for legacy JSON input."
            if path.suffix.lower() == ".json"
            else "Use a .yaml or .yml file."
        )
        raise ConfigurationError(f"{path}: {guidance}")
    from speech_to_speech._config_yaml import read_document

    return read_document(path)


def _metadata() -> tuple[Any, Any, Any, dict[str, Any]]:
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
    from speech_to_speech.arguments_classes.realtime_server_arguments import RealtimeServerArguments
    from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
    from speech_to_speech.backend_registry import LLM_BACKENDS, STT_BACKENDS, TTS_BACKENDS

    return (
        ModuleArguments,
        RealtimeServerArguments,
        VADHandlerArguments,
        {"stt": STT_BACKENDS, "llm": LLM_BACKENDS, "tts": TTS_BACKENDS},
    )


def _client_type() -> Any:
    from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig

    return RealtimeAudioClientConfig


def _mapping(
    document: ConfigDocument, value: Any, path: tuple[str | int, ...], allowed: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _error(document, path, "Use a mapping with string keys.")
    if allowed is not None and set(value) - allowed:
        _error(document, path, "Unknown field; use only documented fields.")
    return value


def _is_env(value: Any) -> bool:
    return isinstance(value, dict) and "env" in value


def _json_value(document: ConfigDocument, value: Any, path: tuple[str | int, ...]) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(document, item, (*path, index))
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _json_value(document, item, path)
        return
    _error(document, path, "Use finite JSON-compatible values.")


def _value(
    document: ConfigDocument, value: Any, expected: Any, path: tuple[str | int, ...], environ: Mapping[str, str] | None
) -> Any:
    origin, args = get_origin(expected), get_args(expected)
    # Open provider dictionaries are literal JSON, including dictionaries with an 'env' key.
    if expected is Any or origin is dict or expected is dict:
        _json_value(document, value, path)
        if origin is dict or expected is dict:
            if not isinstance(value, dict):
                _error(document, path, "Use a YAML mapping.")
        return deepcopy(value)
    if _is_env(value):
        if set(value) != {"env"} or not isinstance(value["env"], str) or not _ENV.fullmatch(value["env"]):
            _error(document, path, "Use {env: NAME} with a valid environment name.")
        targets = args if origin in (Union, types.UnionType) else (expected,)
        if not any(target in (str, bool, int, float) or get_origin(target) is Literal for target in targets):
            _error(document, path, "Environment references require scalar fields; collections must use YAML.")
        if environ is None:
            return deepcopy(value)
        if value["env"] not in environ:
            _error(document, path, "Required environment reference is missing; supply it explicitly.")
        text = environ[value["env"]]
        if not isinstance(text, str):
            _error(document, path, "Environment values must be strings.")
        targets = args if origin in (Union, types.UnionType) else (expected,)
        if origin is Literal:
            targets = tuple(type(item) for item in args)
        if str in targets:
            value = text
        elif bool in targets and text in ("true", "false"):
            value = text == "true"
        elif int in targets and _INTEGER.fullmatch(text):
            number_failure = False
            try:
                value = int(text)
            except ValueError:
                number_failure = True
            if number_failure:
                _error(document, path, "Integer is too large; use a bounded decimal value.")
        elif float in targets and _FLOAT.fullmatch(text):
            value = float(text)
        else:
            _error(document, path, "Environment value has an invalid scalar type; collections must use YAML.")
    if origin in (Union, types.UnionType):
        # Select by literal type first, so invalid values do not leak through exception chains.
        for target in args:
            if target is type(None) and value is None:
                return None
            base = get_origin(target) or target
            if (
                base is Any
                or (base is float and type(value) in (int, float))
                or (isinstance(base, type) and type(value) is base)
            ):
                return _value(document, value, target, path, environ)
        _error(document, path, "Value does not match the nullable field type.")
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            _error(document, path, "Choose a supported literal value.")
        return value
    if origin in (list, tuple) or expected in (list, tuple):
        if not isinstance(value, list):
            _error(document, path, "Use a YAML sequence.")
        element_type = args[0] if args else Any
        return [_value(document, item, element_type, (*path, index), environ) for index, item in enumerate(value)]
    if expected is float:
        if type(value) not in (int, float):
            _error(document, path, "Use a finite number.")
        overflow = False
        try:
            value = float(value)
        except OverflowError:
            overflow = True
        if overflow or not math.isfinite(value):
            _error(document, path, "Use a finite number.")
        return value
    if type(value) is not expected:
        _error(document, path, "Value does not match the field type; quoted scalars remain strings.")
    return deepcopy(value)


def _settings(
    document: ConfigDocument,
    supplied: Any,
    config_type: Any,
    path: tuple[str | int, ...],
    environ: Mapping[str, str] | None,
    sources: dict[Any, str] | None = None,
    only: set[str] | None = None,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    definitions = {
        item.name: item
        for item in fields(config_type)
        if (only is None or item.name in only) and item.name not in (exclude or set())
    }
    supplied = _mapping(document, supplied, path, set(definitions))
    hints = get_type_hints(config_type)
    result: dict[str, Any] = {}
    for name, definition in definitions.items():
        location = (*path, name)
        explicit = name in supplied
        if not explicit:
            if definition.default is not MISSING:
                value = deepcopy(definition.default)
            elif definition.default_factory is not MISSING:
                value = definition.default_factory()
            else:
                _error(document, location, "Provide the required field.")
        else:
            value = supplied[name]
            if config_type.__name__ == "VADHandlerArguments" and name == "max_speech_ms" and value is None:
                value = float("inf")
            else:
                value = _value(document, value, hints[name], location, environ)
            if not _is_env(value) and "choices" in definition.metadata and value not in definition.metadata["choices"]:
                _error(document, location, "Choose one of the existing supported field choices.")
        if sources is not None:
            reference = (
                explicit
                and _is_env(supplied[name])
                and hints[name] not in (Any, dict)
                and get_origin(hints[name]) is not dict
            )
            sources[location] = "environment" if reference else "file" if explicit else "default"
        if environ is not None and name in _PATH_FIELDS and value:
            expansion_failure = False
            try:
                candidate = Path(value).expanduser()
            except RuntimeError:
                expansion_failure = True
            if expansion_failure:
                _error(document, location, "Cannot expand the home directory; provide an absolute path.")
            value = str(candidate if candidate.is_absolute() else document.path.parent / candidate)
        result[name] = value
    return result


def _client_settings(
    document: ConfigDocument,
    supplied: Any,
    environ: Mapping[str, str] | None,
    sources: dict[tuple[str | int, ...], str] | None = None,
) -> dict[str, Any]:
    path = ("runtime", "client")
    supplied = _mapping(document, supplied, path)
    result = _settings(
        document,
        {key: value for key, value in supplied.items() if key != "tool_module"},
        _client_type(),
        path,
        environ,
        sources,
        exclude={"tool_executor"},
    )
    buffer = result["playback_buffer_ms"]
    if not _is_env(buffer) and buffer < 0:
        _error(document, (*path, "playback_buffer_ms"), "Use a finite non-negative playback buffer.")
    tool_module = supplied.get("tool_module")
    result["tool_module"] = _value(document, tool_module, str | None, (*path, "tool_module"), environ)
    if sources is not None:
        sources[(*path, "tool_module")] = (
            "environment" if _is_env(tool_module) else "file" if "tool_module" in supplied else "default"
        )
    return result


def _check_composition(
    document: ConfigDocument,
    name: str,
    stage_blocks: dict[str, Any],
    options: dict[str, Any],
    registries: dict[str, Any],
) -> None:
    path = ("pipelines", name, "options")
    capabilities = registries["llm"][stage_blocks["llm"]["backend"]].capabilities
    bypass = stage_blocks["stt"]["backend"] == "none"
    if bypass and not capabilities.supports_audio_input:
        _error(document, ("pipelines", name, "stages", "llm"), "STT bypass requires an audio-input LLM backend.")
    model = options.get("diarization_model_name")
    diarization = options.get("diarization")
    active = (not _is_env(model) and bool(model)) or diarization is True
    if active:
        if bypass:
            _error(document, path, "Speaker-aware conversation requires an STT backend.")
        threshold = options["diarization_threshold"]
        if not _is_env(threshold) and not 0 < threshold < 1:
            _error(document, path, "Diarization threshold must be between zero and one.")
    proxy = options.get("enable_llm_proxy")
    if not _is_env(proxy) and proxy and not capabilities.supports_llm_proxy:
        _error(document, path, "The selected LLM backend does not support the proxy.")


def validate_config(document: ConfigDocument) -> None:
    """Validate every definition without reading environment values or starting resources."""
    module, server, vad, registries = _metadata()
    data = _mapping(document, document.data, (), {"schema_version", "runtime", "blocks", "pipelines"})
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        _error(document, ("schema_version",), "Use integer schema_version: 1.")
    runtime = _mapping(document, data.get("runtime", {}), ("runtime",), _MODULE_PROCESS | {"server", "client"})
    _settings(
        document,
        {key: value for key, value in runtime.items() if key in _MODULE_PROCESS},
        module,
        ("runtime",),
        None,
        only=_MODULE_PROCESS,
    )
    _settings(document, runtime.get("server", {}), server, ("runtime", "server"), None)
    if "client" in runtime:
        _client_settings(document, runtime["client"], None)
    blocks = _mapping(document, data.get("blocks"), ("blocks",))
    pipelines = _mapping(document, data.get("pipelines"), ("pipelines",))
    if not blocks or not pipelines:
        _error(document, (), "Define nonempty blocks and pipelines mappings.")
    for name, block in blocks.items():
        path = ("blocks", name)
        if not _ID.fullmatch(name):
            _error(
                document,
                ("blocks",),
                "Use IDs of 1 to 64 letters, digits, underscores or hyphens, starting with a letter.",
            )
        block = _mapping(document, block, path, {"kind", "backend", "settings"})
        kind, backend = block.get("kind"), block.get("backend")
        if not isinstance(kind, str) or kind not in _STAGES or not isinstance(backend, str):
            _error(document, path, "Provide a supported kind and literal backend name.")
        if kind == "vad":
            if backend not in next(item for item in fields(vad) if item.name == "vad").metadata["choices"]:
                _error(document, path, "Choose a supported VAD backend.")
            config_type, excluded = vad, {"vad"}
        else:
            if backend not in registries[kind]:
                _error(document, path, "Choose an existing registered backend.")
            config_type, excluded = registries[kind][backend].config_type, set()
        _settings(document, block.get("settings", {}), config_type, (*path, "settings"), None, exclude=excluded)
    for name, pipeline in pipelines.items():
        path = ("pipelines", name)
        if not _ID.fullmatch(name):
            _error(document, ("pipelines",), "Use valid case-sensitive pipeline identifiers.")
        pipeline = _mapping(document, pipeline, path, {"stages", "num_pipelines", "options"})
        stages = _mapping(document, pipeline.get("stages"), (*path, "stages"), set(_STAGES))
        if set(stages) != set(_STAGES):
            _error(document, (*path, "stages"), "Provide exactly vad, stt, llm and tts references.")
        stage_blocks = {}
        for kind, block_id in stages.items():
            if not isinstance(block_id, str) or block_id not in blocks or blocks[block_id]["kind"] != kind:
                _error(document, (*path, "stages", kind), "Reference an existing block of the matching kind.")
            stage_blocks[kind] = blocks[block_id]
        count = _value(document, pipeline.get("num_pipelines", 1), int, (*path, "num_pipelines"), None)
        if not _is_env(count) and count < 1:
            _error(document, (*path, "num_pipelines"), "Use a positive integer instance count.")
        options = _settings(
            document,
            pipeline.get("options", {}),
            module,
            (*path, "options"),
            None,
            only=_MODULE_PIPELINE,
        )
        _check_composition(document, name, stage_blocks, options, registries)


def resolve_config(
    document: ConfigDocument,
    *,
    names: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    include_client: bool = False,
    client_only: bool = False,
) -> ResolvedConfig:
    """Resolve selected definitions to independent settings and existing backend selections."""
    validate_config(document)
    module, server, vad, registries = _metadata()
    definitions = document.data["pipelines"]
    if client_only:
        if names is not None:
            _error(document, ("pipelines",), "Client-only resolution does not accept pipeline names.")
        selected = []
    elif names is None:
        selected = list(definitions)
    else:
        if isinstance(names, (str, bytes)) or not isinstance(names, Sequence) or not names:
            _error(document, ("pipelines",), "Provide a nonempty sequence of pipeline names.")
        selected = list(names)
        if any(not isinstance(name, str) or name not in definitions for name in selected) or len(set(selected)) != len(
            selected
        ):
            _error(document, ("pipelines",), "Select existing names exactly once.")
    environment = dict(os.environ if environ is None else environ)
    sources: dict[tuple[str | int, ...], str] = {}
    raw_runtime = document.data.get("runtime", {})
    runtime = _settings(
        document,
        {key: value for key, value in raw_runtime.items() if key in _MODULE_PROCESS},
        module,
        ("runtime",),
        environment,
        sources,
        only=_MODULE_PROCESS,
    )
    if not client_only:
        runtime["server"] = _settings(
            document, raw_runtime.get("server", {}), server, ("runtime", "server"), environment, sources
        )
    client = None
    if include_client or client_only:
        client = _client_settings(document, raw_runtime.get("client", {}), environment, sources)
    resolved = {}
    from speech_to_speech.backend_registry import BackendSelection

    for name in selected:
        pipeline = definitions[name]
        count = _value(
            document, pipeline.get("num_pipelines", 1), int, ("pipelines", name, "num_pipelines"), environment
        )
        if count < 1:
            _error(document, ("pipelines", name, "num_pipelines"), "Use a positive integer instance count.")
        sources[("pipelines", name, "num_pipelines")] = (
            "environment"
            if _is_env(pipeline.get("num_pipelines"))
            else "file"
            if "num_pipelines" in pipeline
            else "default"
        )
        options = _settings(
            document,
            pipeline.get("options", {}),
            module,
            ("pipelines", name, "options"),
            environment,
            sources,
            only=_MODULE_PIPELINE,
        )
        stage_blocks = {kind: document.data["blocks"][block_id] for kind, block_id in pipeline["stages"].items()}
        _check_composition(document, name, stage_blocks, options, registries)
        stages = {}
        for kind, block_id in pipeline["stages"].items():
            block = stage_blocks[kind]
            if kind == "vad":
                config_type = vad
            else:
                spec = registries[kind][block["backend"]]
                config_type = spec.config_type
            settings = _settings(
                document,
                block.get("settings", {}),
                config_type,
                ("blocks", block_id, "settings"),
                environment,
                sources,
                exclude={"vad"} if kind == "vad" else None,
            )
            if kind == "vad":
                settings["vad"] = block["backend"]
                selection = None
            else:
                selection = BackendSelection(spec, spec.normalize(config_type(**deepcopy(settings))))
            stages[kind] = ResolvedStage(block_id, kind, block["backend"], settings, selection)
        resolved[name] = ResolvedPipeline(count, stages, options)
    return ResolvedConfig(runtime, resolved, client, sources)
