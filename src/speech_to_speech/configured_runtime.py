"""Explicit configured construction and lifecycle. Construction can load models."""

from __future__ import annotations

import asyncio
import logging
import signal
from copy import deepcopy
from dataclasses import fields
from ipaddress import ip_address
from threading import Event, RLock, Thread, current_thread
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from speech_to_speech.config import _ID, ConfigurationError, ResolvedConfig

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig
    from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
    from speech_to_speech.s2s_pipeline import ParsedArguments

logger = logging.getLogger(__name__)
_JOIN_TIMEOUT_S = 5.0


class ConfiguredRuntimeError(RuntimeError):
    """A runtime error that does not include provider diagnostics or credentials."""


def _explicit(config: ResolvedConfig, *path: str) -> bool:
    return config.sources.get(path, "default") != "default"


def _cleanup(resource: Any) -> None:
    try:
        cleanup = getattr(resource, "cleanup", None) or getattr(resource, "reset", None)
        if cleanup is None and hasattr(resource, "diarizer"):
            cleanup = resource.diarizer.reset
        if cleanup is not None:
            cleanup()
    except Exception:
        logger.warning("Configured resource cleanup failed")


def _validate_counts(config: ResolvedConfig) -> None:
    if not config.pipelines:
        raise ConfigurationError("Select at least one pipeline definition.")
    for pipeline in config.pipelines.values():
        if type(pipeline.num_pipelines) is not int or pipeline.num_pipelines < 1:
            raise ConfigurationError("Use positive integer pipeline instance counts.")


def _adapt(config: ResolvedConfig) -> dict[str, ParsedArguments]:
    from speech_to_speech import s2s_pipeline as pipeline_module
    from speech_to_speech.arguments_classes.local_audio_arguments import LocalAudioArguments
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
    from speech_to_speech.arguments_classes.realtime_server_arguments import RealtimeServerArguments
    from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
    from speech_to_speech.backend_registry import LLM_BACKENDS, STT_BACKENDS, TTS_BACKENDS, BackendSelection

    adapted = {}
    for name, pipeline in config.pipelines.items():
        options = deepcopy(pipeline.options)
        preset = (
            pipeline_module._mac_preset_defaults(pipeline.stages["llm"].backend)
            if options.get("mac_optimal_settings")
            else {}
        )
        for key, value in preset.items():
            if key in options and not _explicit(config, "pipelines", name, "options", key):
                options[key] = value
        options.update(
            stt=pipeline.stages["stt"].backend,
            llm_backend=pipeline.stages["llm"].backend,
            tts=pipeline.stages["tts"].backend,
            num_pipelines=pipeline.num_pipelines,
            log_level=config.runtime.get("log_level", "info"),
            log_transcripts=config.runtime.get("log_transcripts", False),
        )
        if options.get("diarization") and options.get("diarization_model_name") is None:
            options["diarization_model_name"] = "nvidia/Nemotron-3-Diarization"
        selections = {}
        for kind, registry in (("stt", STT_BACKENDS), ("llm", LLM_BACKENDS), ("tts", TTS_BACKENDS)):
            stage = pipeline.stages[kind]
            spec = registry[stage.backend]
            settings = deepcopy(stage.settings)
            for key in {field.name for field in fields(spec.config_type)} & preset.keys():
                if not _explicit(config, "blocks", stage.block_id, "settings", key):
                    settings[key] = deepcopy(preset[key])
            selections[kind] = BackendSelection(spec, spec.normalize(spec.config_type(**settings)))
        args = pipeline_module.ParsedArguments(
            module_kwargs=ModuleArguments(**options),
            realtime_server_kwargs=RealtimeServerArguments(**deepcopy(config.runtime.get("server", {}))),
            local_audio_kwargs=LocalAudioArguments(),
            vad_handler_kwargs=VADHandlerArguments(**deepcopy(pipeline.stages["vad"].settings)),
            stt_backend=selections["stt"],
            llm_backend=selections["llm"],
            tts_backend=selections["tts"],
        )
        try:
            pipeline_module.prepare_all_args(args)
        except ValueError:
            raise ConfigurationError("The selected pipeline settings are incompatible.") from None
        adapted[name] = args
    if pipeline_module.platform == "darwin" and sum(p.num_pipelines for p in config.pipelines.values()) > 1:
        for args in adapted.values():
            args.module_kwargs.enable_live_transcription = False
    return adapted


def _named_client_url(url: str, name: str) -> str:
    from speech_to_speech.api.openai_realtime.audio_client import normalize_realtime_url

    try:
        normalize_realtime_url(url)
        parsed = urlsplit(url.strip())
    except ValueError:
        raise ConfigurationError("Use a complete Realtime client URL.") from None
    path = parsed.path.rstrip("/")
    _, separator, selected = path.rpartition("/pipelines/")
    selected_name = selected.removesuffix("/realtime")
    if separator and selected.endswith("/realtime") and _ID.fullmatch(selected_name):
        if selected != f"{name}/realtime":
            raise ConfigurationError("The client URL selects a different pipeline.")
    else:
        path = path.removesuffix("/realtime") + f"/pipelines/{name}/realtime"
    return urlunsplit(parsed._replace(path=path))


def build_configured_client(
    config: ResolvedConfig,
    *,
    local: bool = False,
    pipeline_name: str | None = None,
) -> RealtimeAudioClientConfig:
    """Build independent packaged client settings and load explicitly selected tools."""
    from speech_to_speech.api.openai_realtime.audio_client import (
        RealtimeAudioClientConfig,
        build_session_update,
        load_realtime_tool_module,
    )

    if pipeline_name is not None and (
        not isinstance(pipeline_name, str)
        or not _ID.fullmatch(pipeline_name)
        or (config.pipelines and pipeline_name not in config.pipelines)
    ):
        raise ConfigurationError("Select an existing valid pipeline name.")
    settings = deepcopy(config.client or {})
    module = settings.pop("tool_module", None)
    process_explicit = _explicit(config, "runtime", "log_transcripts")
    client_explicit = _explicit(config, "runtime", "client", "log_transcripts")
    process_log = config.runtime.get("log_transcripts", False)
    if process_explicit and client_explicit and settings.get("log_transcripts", False) != process_log:
        raise ConfigurationError("Process and client transcript settings must agree.")
    if process_explicit or not client_explicit:
        settings["log_transcripts"] = process_log
    if module and _explicit(config, "runtime", "client", "tools"):
        raise ConfigurationError("Use either a tool module or explicit client tools.")
    if local:
        if len(config.pipelines) != 1:
            raise ConfigurationError("Local startup requires exactly one pipeline definition.")
        server = config.runtime.get("server", {})
        host = server.get("host", "127.0.0.1")
        try:
            loopback = host == "localhost" or ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if _explicit(config, "runtime", "server", "host") and not loopback:
            raise ConfigurationError("Local startup requires a loopback server host.")
        url = f"ws://127.0.0.1:{server.get('port', 8765)}/v1/realtime"
        if pipeline_name is not None:
            url = _named_client_url(url, pipeline_name)
        provided_url = settings.get("url", url)
        if pipeline_name is not None:
            provided_url = _named_client_url(provided_url, pipeline_name)
        if _explicit(config, "runtime", "client", "url") and provided_url != url:
            raise ConfigurationError("Local client URL must match the loopback server port.")
        settings["url"] = url
        if not _explicit(config, "runtime", "client", "playback_buffer_ms"):
            pipeline = next(iter(config.pipelines.values()))
            settings["playback_buffer_ms"] = 196.0 if pipeline.stages["tts"].backend == "openai" else 0.0
    if pipeline_name is not None and not local:
        settings["url"] = _named_client_url(settings.get("url", RealtimeAudioClientConfig().url), pipeline_name)
    if module:
        tools, executor, create_response = load_realtime_tool_module(module)
        settings.update(tools=tools, tool_executor=executor)
        if not _explicit(config, "runtime", "client", "tool_response_create"):
            settings["tool_response_create"] = create_response
    try:
        client = RealtimeAudioClientConfig(**settings)
        build_session_update(client)
    except ValueError:
        raise ConfigurationError("Invalid packaged client settings or tool contract.") from None
    return client


class ConfiguredRuntime:
    """Own named worker pools and optional server/client resources without installing signals.

    Running means managed threads have started. It does not indicate listener readiness.
    """

    def __init__(self, stop_event: Event) -> None:
        self.pools: dict[str, list[PipelineUnit]] = {}
        self.server: Any = None
        self.client: Any = None
        self.state = "constructed"
        self._caller_stop = stop_event
        self._stop = Event()
        self._events: list[Event] = []
        self._resources: list[Any] = []
        self._handlers: list[Any] = []
        self._threads: list[Thread] = []
        self._started: set[int] = set()
        self._monitor: Thread | None = None
        self._failure: str | None = None
        self._lock = RLock()
        self.ready = Event()
        self._readiness_lock = RLock()
        self._initial_ready = False
        self._handler_units: dict[int, PipelineUnit] = {}
        self._worker_started: dict[int, set[int]] = {}

    def _signal_stop(self) -> None:
        with self._readiness_lock:
            self.ready.clear()
            self._stop.set()
            for event in self._events:
                event.set()

    def _unit_failed(self, unit: PipelineUnit) -> None:
        if not self._initial_ready and not self._stop.is_set():
            self._failure = "A configured worker failed during startup."
            self._signal_stop()
        else:
            self._update_ready()

    def _update_ready(self) -> None:
        with self._readiness_lock:
            healthy = all(any(unit.healthy for unit in units) for units in self.pools.values())
            if not self._initial_ready:
                healthy = healthy and all(unit.healthy for units in self.pools.values() for unit in units)
            listener_ready = self.server is None or self.server.ready.is_set()
            if self.state == "running" and not self._stop.is_set() and healthy and listener_ready:
                self._initial_ready = True
                self.ready.set()
            else:
                self.ready.clear()

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self.ready.wait(timeout)

    def _run(self, handler: Any) -> None:
        unit = self._handler_units.get(id(handler))
        if unit is not None:
            with unit.lock:
                started = self._worker_started[id(unit)]
                started.add(id(handler))
                if len(started) == len(unit.handlers) and unit.lifecycle_state == "constructed":
                    unit.lifecycle_state = "ready"
        try:
            if handler is self.client:
                from speech_to_speech.api.openai_realtime.audio_client import listen_and_play_realtime

                asyncio.run(listen_and_play_realtime(handler.config, stop_event=handler.stop_event))
            else:
                handler.run()
        except BaseException:
            if unit is not None:
                unit.mark_failed()
            else:
                self._failure = "A configured listener or client failed."
                self._signal_stop()
        finally:
            if unit is not None:
                if not self._stop.is_set() and not self._caller_stop.is_set():
                    unit.mark_failed()
            else:
                if not self._stop.is_set() and not self._caller_stop.is_set():
                    self._failure = "A configured listener or client stopped unexpectedly."
                self._signal_stop()

    def _watch(self) -> None:
        while not self._stop.wait(0.01):
            if self._caller_stop.is_set():
                self._signal_stop()
                break
            for unit in list(self._handler_units.values()):
                if (
                    unit.stop_event is not None
                    and unit.stop_event.is_set()
                    and unit.lifecycle_state not in {"failed", "stopped"}
                ):
                    unit.mark_failed()
            self._update_ready()

    def start(self) -> None:
        with self._lock:
            if self.state == "running":
                return
            if self.state != "constructed":
                raise ConfiguredRuntimeError("A stopped or failed configured runtime cannot restart.")
            if self._caller_stop.is_set():
                self.stop()
                return
            try:
                for handler in self._handlers:
                    if self._stop.is_set():
                        raise ConfiguredRuntimeError("Configured startup stopped before completion.")
                    thread = Thread(target=self._run, args=(handler,))
                    thread.start()
                    self._threads.append(thread)
                    self._started.add(id(handler))
                monitor = Thread(target=self._watch, daemon=True)
                monitor.start()
                self._monitor = monitor
                self.state = "running"
            except BaseException:
                self._failure = "Configured thread startup failed."
                self.stop()
                raise ConfiguredRuntimeError(self._failure) from None

    def stop(self) -> None:
        with self._lock:
            if self.state in {"stopped", "failed"}:
                return
            self.state = "stopping"
            self._signal_stop()
            deadline = monotonic() + _JOIN_TIMEOUT_S
            for thread in self._threads:
                if thread is not current_thread():
                    thread.join(max(0.0, deadline - monotonic()))
            if self._monitor is not None and self._monitor is not current_thread():
                self._monitor.join(max(0.0, deadline - monotonic()))
            survivors = sum(thread.is_alive() for thread in self._threads)
            if self._monitor is not None and self._monitor.is_alive():
                survivors += 1
            if survivors:
                logger.warning("Configured shutdown has %d surviving threads", survivors)
                raise ConfiguredRuntimeError("Configured shutdown has threads still running.")
            for resource in reversed(self._resources):
                if id(resource) not in self._started:
                    _cleanup(resource)
            for units in self.pools.values():
                for unit in units:
                    with unit.lock:
                        if unit.lifecycle_state != "failed":
                            unit.lifecycle_state = "stopped"
                        unit.on_failure = None
            self.pools.clear()
            self.server = self.client = None
            self._resources.clear()
            self._handlers.clear()
            self._threads.clear()
            self._started.clear()
            self._events.clear()
            self._handler_units.clear()
            self._worker_started.clear()
            self._monitor = None
            self.state = "failed" if self._failure else "stopped"

    def wait(self) -> None:
        if self.state == "constructed":
            return
        while self.state == "running" and any(thread.is_alive() for thread in self._threads):
            if self._caller_stop.wait(0.05) or self._stop.is_set():
                break
        self.stop()
        if self._failure:
            raise ConfiguredRuntimeError(self._failure)


def build_configured_runtime(config: ResolvedConfig, stop_event: Event) -> ConfiguredRuntime:
    """Construct independent named workers. Models can initialize before start()."""
    _validate_counts(config)
    return _build_runtime(_adapt(config), stop_event)


def _build_runtime(arguments: dict[str, ParsedArguments], stop_event: Event) -> ConfiguredRuntime:
    from speech_to_speech.s2s_pipeline import _build_pipeline_unit

    result = ConfiguredRuntime(stop_event)
    index = 0
    try:
        for name, args in arguments.items():
            result.pools[name] = []
            for instance_index in range(args.module_kwargs.num_pipelines):
                event = Event()
                result._events.append(event)
                unit = _build_pipeline_unit(
                    index=index,
                    stop_event=event,
                    module_kwargs=deepcopy(args.module_kwargs),
                    vad_handler_kwargs=deepcopy(args.vad_handler_kwargs),
                    stt_backend=args.stt_backend.copy_for_pipeline(),
                    llm_backend=args.llm_backend.copy_for_pipeline(),
                    tts_backend=args.tts_backend.copy_for_pipeline(),
                    resource_ledger=result._resources,
                )
                unit.pipeline_name = name
                unit.instance_index = instance_index
                unit.stop_event = event
                unit.lifecycle_state = "constructed"
                unit.on_failure = result._unit_failed
                result._worker_started[id(unit)] = set()
                result._handler_units.update({id(handler): unit for handler in unit.handlers})
                result.pools[name].append(unit)
                result._handlers.extend(unit.handlers)
                index += 1
    except BaseException as exc:
        result._failure = f"Configured pipeline {name} construction failed."
        result.stop()
        raise ConfiguredRuntimeError(result._failure) from exc
    return result


def build_configured_server(config: ResolvedConfig, stop_event: Event, *, local: bool = False) -> ConfiguredRuntime:
    """Publish selected definitions through one server, with an optional loopback client."""
    _validate_counts(config)
    if local and len(config.pipelines) != 1:
        raise ConfigurationError("Local startup requires exactly one selected pipeline definition.")
    client_config = build_configured_client(config, local=True) if local else None
    arguments = _adapt(config)
    result = _build_runtime(arguments, stop_event)
    try:
        from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClient
        from speech_to_speech.api.openai_realtime.server import RealtimeServer
        from speech_to_speech.s2s_pipeline import build_llm_proxy_config

        args = next(iter(arguments.values()))
        event = result._stop
        result.server = RealtimeServer(
            stop_event=event,
            pool=[unit for units in result.pools.values() for unit in units],
            pipeline_pools=result.pools,
            llm_proxy_configs={
                name: build_llm_proxy_config(selected.module_kwargs, selected.llm_backend)
                if selected.module_kwargs.enable_llm_proxy
                else None
                for name, selected in arguments.items()
            },
            host="127.0.0.1" if local else args.realtime_server_kwargs.host,
            port=args.realtime_server_kwargs.port,
        )
        result._resources.append(result.server)
        result._handlers.append(result.server)
        if client_config is not None:
            result.client = RealtimeAudioClient(event, client_config)
            result._resources.append(result.client)
            result._handlers.append(result.client)
    except BaseException as exc:
        result._failure = "Configured server construction failed."
        result.stop()
        raise ConfiguredRuntimeError(result._failure) from exc
    return result


def run_configured_command(
    command: Literal["serve", "local", "talk"],
    config: ResolvedConfig,
    *,
    pipeline_name: str | None = None,
) -> None:
    """Own CLI signal handlers and always stop constructed resources."""
    from speech_to_speech.pipeline.transcript_logging import set_log_transcripts, warn_if_log_transcripts_enabled

    stop_event = Event()
    result: ConfiguredRuntime | None = None
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, lambda *_: stop_event.set())
        if command == "talk":
            from speech_to_speech.api.openai_realtime.audio_client import listen_and_play_realtime

            client = build_configured_client(config, pipeline_name=pipeline_name)
            logging.basicConfig(level=config.runtime.get("log_level", "info").upper())
            set_log_transcripts(client.log_transcripts)
            warn_if_log_transcripts_enabled()
            asyncio.run(listen_and_play_realtime(client, stop_event=stop_event))
        else:
            from speech_to_speech.s2s_pipeline import setup_logger

            setup_logger(config.runtime.get("log_level", "info"))
            result = build_configured_server(config, stop_event, local=command == "local")
            set_log_transcripts(
                result.client.config.log_transcripts
                if command == "local"
                else config.runtime.get("log_transcripts", False)
            )
            warn_if_log_transcripts_enabled()
            result.start()
            result.wait()
    finally:
        stop_event.set()
        try:
            if result is not None:
                result.stop()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
