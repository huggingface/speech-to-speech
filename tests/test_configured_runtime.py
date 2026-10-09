import copy
import signal
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest

from speech_to_speech import configured_runtime as runtime
from speech_to_speech.config import ConfigurationError, load_config, resolve_config

BASE = """schema_version: 1
blocks:
  vad: {kind: vad, backend: silero}
  stt: {kind: stt, backend: openai}
  llm: {kind: llm, backend: chat-completions}
  tts: {kind: tts, backend: openai}
pipelines:
  primary:
    num_pipelines: 2
    stages: {vad: vad, stt: stt, llm: llm, tts: tts}
  secondary:
    stages: {vad: vad, stt: stt, llm: llm, tts: tts}
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(BASE)
    return resolve_config(load_config(path), include_client=True, environ={})


class Handler:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.cleaned = 0
        self.queue_in = Queue()

    def run(self):
        self.stop_event.wait()
        self.cleanup()

    def cleanup(self):
        self.cleaned += 1


def _unit(index, stop_event, handlers):
    from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
    from speech_to_speech.api.openai_realtime.service import RealtimeService
    from speech_to_speech.pipeline.cancel_scope import CancelScope

    prompt_queue = Queue()
    listen = Event()
    return PipelineUnit(
        index=index,
        service=RealtimeService(text_prompt_queue=prompt_queue, should_listen=listen),
        cancel_scope=CancelScope(),
        should_listen=listen,
        response_playing=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=prompt_queue,
        handlers=handlers,
    )


@pytest.fixture
def builder(monkeypatch):
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        handler = Handler(kwargs["stop_event"])
        kwargs["resource_ledger"].append(handler)
        return _unit(kwargs["index"], kwargs["stop_event"], [handler])

    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    return calls


def test_named_pools_are_independent_and_explicitly_started(config, builder, monkeypatch):
    original = copy.deepcopy(config)
    monkeypatch.setattr(signal, "signal", lambda *args: pytest.fail("library installs signals"))
    result = runtime.build_configured_runtime(config, Event())
    assert result.state == "constructed"
    assert list(result.pools) == ["primary", "secondary"]
    assert [u.index for pool in result.pools.values() for u in pool] == [0, 1, 2]
    assert builder[0]["stop_event"] is not builder[1]["stop_event"]
    assert builder[0]["stop_event"] is not builder[2]["stop_event"]
    assert builder[0]["llm_backend"].config is not builder[1]["llm_backend"].config
    builder[0]["llm_backend"].config["gen_kwargs"]["changed"] = True
    assert "changed" not in builder[1]["llm_backend"].config["gen_kwargs"]
    assert config == original
    handlers = [unit.handlers[0] for pool in result.pools.values() for unit in pool]
    result.start()
    result.start()
    assert result.state == "running"
    result.stop()
    result.stop()
    result.wait()
    assert result.state == "stopped"
    assert [handler.cleaned for handler in handlers] == [1, 1, 1]
    assert result.pools == {}
    with pytest.raises(runtime.ConfiguredRuntimeError):
        result.start()


def test_stop_before_start_cleans_once(config, builder):
    result = runtime.build_configured_runtime(config, Event())
    handlers = [u.handlers[0] for pool in result.pools.values() for u in pool]
    result.stop()
    result.stop()
    assert [handler.cleaned for handler in handlers] == [1, 1, 1]
    assert result.state == "stopped"


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_modified_count_is_checked_before_allocation(config, builder, count):
    config.pipelines["secondary"].num_pipelines = count
    with pytest.raises(ConfigurationError):
        runtime.build_configured_runtime(config, Event())
    assert builder == []


def test_local_rejects_several_definitions_before_allocation(config, builder):
    with pytest.raises(ConfigurationError):
        runtime.build_configured_server(config, Event(), local=True)
    assert builder == []


def test_construction_failure_cleans_partial_resources_in_reverse(config, monkeypatch):
    resources = []
    cleaned = []

    def build(**kwargs):
        handler = Handler(kwargs["stop_event"])
        order = len(resources)
        handler.cleanup = lambda: cleaned.append(order)
        resources.append(handler)
        kwargs["resource_ledger"].append(handler)
        if order == 2:
            raise ValueError("SECRET_PROVIDER_TEXT")
        return _unit(kwargs["index"], kwargs["stop_event"], [handler])

    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    with pytest.raises(runtime.ConfiguredRuntimeError) as exc:
        runtime.build_configured_runtime(config, Event())
    assert "SECRET_PROVIDER_TEXT" not in str(exc.value)
    assert cleaned == [2, 1, 0]
    assert all(handler.stop_event.is_set() for handler in resources)


def test_caller_cancellation_propagates(config, builder):
    stop = Event()
    result = runtime.build_configured_runtime(config, stop)
    handlers = [u.handlers[0] for pool in result.pools.values() for u in pool]
    result.start()
    stop.set()
    result.wait()
    assert result.state == "stopped"
    assert all(h.stop_event.is_set() and h.cleaned == 1 for h in handlers)


def test_partial_thread_start_stops_previous_workers(config, builder, monkeypatch):
    result = runtime.build_configured_runtime(config, Event())
    handlers = [u.handlers[0] for pool in result.pools.values() for u in pool]
    original = runtime.Thread.start
    count = 0

    def start(thread):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("SECRET_THREAD_ERROR")
        original(thread)

    monkeypatch.setattr(runtime.Thread, "start", start)
    with pytest.raises(runtime.ConfiguredRuntimeError):
        result.start()
    assert result.state == "failed"
    assert all(h.cleaned == 1 for h in handlers)


def test_bounded_stop_retains_resources_until_worker_finishes(config, monkeypatch):
    release = Event()
    handler = Handler(Event())
    handler.run = lambda: release.wait()

    def build(**kwargs):
        kwargs["resource_ledger"].append(handler)
        return _unit(kwargs["index"], kwargs["stop_event"], [handler])

    config.pipelines = {"secondary": config.pipelines["secondary"]}
    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    monkeypatch.setattr(runtime, "_JOIN_TIMEOUT_S", 0.01)
    result = runtime.build_configured_runtime(config, Event())
    result.start()
    with pytest.raises(runtime.ConfiguredRuntimeError, match="still running"):
        result.stop()
    assert result.state == "stopping"
    assert result.pools
    release.set()
    result.wait()
    assert result.state == "stopped"
    assert result.pools == {}


def test_local_uses_one_client_and_preserves_explicit_settings(config, builder, monkeypatch):
    config.pipelines = {"primary": config.pipelines["primary"]}
    config.runtime["server"]["port"] = 12345
    config.client.update(playback_buffer_ms=0, api_key="", model="custom", voice="alloy", input_device=0)
    for key in ("playback_buffer_ms", "api_key", "model", "voice", "input_device"):
        config.sources[("runtime", "client", key)] = "file"
    result = runtime.build_configured_server(config, Event(), local=True)
    assert result.server.host == "127.0.0.1"
    assert len(result.pools["primary"]) == 2
    assert result.client.config.url == "ws://127.0.0.1:12345/v1/realtime"
    assert result.client.config.playback_buffer_ms == 0
    assert result.client.config.api_key == ""
    assert result.client.config.model == "custom"
    assert result.client.config.input_device == 0
    result.stop()


@pytest.mark.parametrize("field,value", [("host", "0.0.0.0"), ("url", "ws://example.test/realtime")])
def test_local_conflicts_fail_before_workers(config, builder, field, value):
    config.pipelines = {"primary": config.pipelines["primary"]}
    section = "server" if field == "host" else "client"
    mapping = config.runtime["server"] if field == "host" else config.client
    mapping[field] = value
    config.sources[("runtime", section, field)] = "file"
    with pytest.raises(ConfigurationError):
        runtime.build_configured_server(config, Event(), local=True)
    assert builder == []


def test_tools_require_executor_before_server_construction(config, builder):
    config.pipelines = {"primary": config.pipelines["primary"]}
    config.client["tools"] = [{"type": "function", "name": "tool"}]
    with pytest.raises(ConfigurationError):
        runtime.build_configured_server(config, Event(), local=True)
    assert builder == []


def test_client_log_conflict_and_single_explicit_choice(config):
    config.runtime["log_transcripts"] = True
    config.sources[("runtime", "log_transcripts")] = "file"
    assert runtime.build_configured_client(config).log_transcripts is True
    config.sources[("runtime", "client", "log_transcripts")] = "file"
    with pytest.raises(ConfigurationError):
        runtime.build_configured_client(config)


def test_local_buffer_default_and_talk_defaults(config):
    config.pipelines = {"primary": config.pipelines["primary"]}
    assert runtime.build_configured_client(config, local=True).playback_buffer_ms == 196
    assert runtime.build_configured_client(config).playback_buffer_ms == 0


def test_cli_restores_signals_after_start_failure(config, monkeypatch):
    calls = []
    owned = SimpleNamespace(
        start=lambda: (_ for _ in ()).throw(RuntimeError("SECRET")), stop=lambda: calls.append("stop")
    )
    monkeypatch.setattr(runtime, "build_configured_server", lambda *args, **kwargs: owned)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: calls.append((sig, handler)))
    monkeypatch.setattr(signal, "getsignal", lambda sig: f"original-{sig}")
    with pytest.raises(RuntimeError):
        runtime.run_configured_command("serve", config)
    assert "stop" in calls
    assert calls[-2:] == [(signal.SIGINT, f"original-{signal.SIGINT}"), (signal.SIGTERM, f"original-{signal.SIGTERM}")]


def test_macos_preset_respects_sources_backend_and_global_device(config, builder, monkeypatch):
    config.pipelines = {"primary": config.pipelines["primary"]}
    pipeline = config.pipelines["primary"]
    pipeline.options.update(mac_optimal_settings=True, device="cpu", diarization_device="auto")
    config.sources[("pipelines", "primary", "options", "diarization_device")] = "file"
    pipeline.stages["llm"].settings.update(model_name="", responses_api_stream=False)
    for key in ("model_name", "responses_api_stream"):
        config.sources[("blocks", "llm", "settings", key)] = "file"
    original = copy.deepcopy(config)
    monkeypatch.setattr("speech_to_speech.s2s_pipeline.platform", "darwin")
    result = runtime.build_configured_runtime(config, Event())
    assert builder[0]["llm_backend"].name == "chat-completions"
    assert builder[0]["llm_backend"].config["model_name"] == ""
    assert builder[0]["llm_backend"].config["stream"] is False
    assert builder[0]["module_kwargs"].enable_live_transcription is False
    assert builder[0]["module_kwargs"].diarization_device == "auto"
    assert config == original
    result.stop()


def test_complete_activation_set_applies_contention_policy(config, builder, monkeypatch):
    for pipeline in config.pipelines.values():
        pipeline.num_pipelines = 1
    monkeypatch.setattr("speech_to_speech.s2s_pipeline.platform", "darwin")
    result = runtime.build_configured_runtime(config, Event())
    assert all(call["module_kwargs"].enable_live_transcription is False for call in builder)
    result.stop()


def test_inactive_definition_has_no_handlers(config, builder):
    config.pipelines = {"secondary": config.pipelines["secondary"]}
    result = runtime.build_configured_runtime(config, Event())
    assert len(builder) == 1
    assert list(result.pools) == ["secondary"]
    result.stop()


def test_cleanup_error_does_not_stop_reverse_cleanup(config, monkeypatch):
    cleaned = []
    original = ValueError("original provider error")

    def build(**kwargs):
        for index in range(3):
            handler = Handler(kwargs["stop_event"])

            def cleanup(index=index):
                cleaned.append(index)
                if index == 1:
                    raise ValueError("cleanup provider error")

            handler.cleanup = cleanup
            kwargs["resource_ledger"].append(handler)
        raise original

    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    with pytest.raises(runtime.ConfiguredRuntimeError) as exc:
        runtime.build_configured_runtime(config, Event())
    assert exc.value.__cause__ is original
    assert cleaned == [2, 1, 0]


def test_listener_failure_stops_workers_and_wait_reports_error(config, builder, monkeypatch):
    config.pipelines = {"primary": config.pipelines["primary"]}
    result = runtime.build_configured_server(config, Event())
    handlers = [u.handlers[0] for u in result.pools["primary"]]
    monkeypatch.setattr(result.server, "run", lambda: (_ for _ in ()).throw(RuntimeError("SECRET_LISTENER")))
    try:
        result.start()
        with pytest.raises(runtime.ConfiguredRuntimeError):
            result.wait()
    except runtime.ConfiguredRuntimeError:
        pass
    assert all(h.cleaned == 1 for h in handlers)
    assert result.state == "failed"


def test_pre_cancelled_runtime_cleans_without_threads(config, builder):
    stop = Event()
    result = runtime.build_configured_runtime(config, stop)
    handlers = [u.handlers[0] for pool in result.pools.values() for u in pool]
    stop.set()
    result.start()
    assert result.state == "stopped"
    assert all(h.cleaned == 1 for h in handlers)


def test_never_started_diarizer_worker_resets_owned_model(config, monkeypatch):
    reset = []
    worker = SimpleNamespace(
        stop_event=Event(), diarizer=SimpleNamespace(reset=lambda: reset.append(True)), run=lambda: None
    )

    def build(**kwargs):
        kwargs["resource_ledger"].append(worker)
        return _unit(kwargs["index"], kwargs["stop_event"], [worker])

    config.pipelines = {"secondary": config.pipelines["secondary"]}
    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    result = runtime.build_configured_runtime(config, Event())
    result.stop()
    result.stop()
    assert reset == [True]


def test_local_loads_tools_once_and_explicit_response_policy_wins(config, builder, monkeypatch):
    config.pipelines = {"primary": config.pipelines["primary"]}
    config.client.update(tool_module="test_tools", tool_response_create=False)
    config.sources[("runtime", "client", "tool_response_create")] = "file"
    loaded = []

    def load(module):
        loaded.append(module)
        return [], lambda *args: None, True

    monkeypatch.setattr("speech_to_speech.api.openai_realtime.audio_client.load_realtime_tool_module", load)
    result = runtime.build_configured_server(config, Event(), local=True)
    assert loaded == ["test_tools"]
    assert result.client.config.tool_response_create is False
    result.stop()


def test_explicit_empty_tools_conflicts_with_module(config):
    config.client["tool_module"] = "test_tools"
    config.sources[("runtime", "client", "tools")] = "file"
    with pytest.raises(ConfigurationError):
        runtime.build_configured_client(config)


def test_started_base_handler_cleans_when_input_gate_raises(config, monkeypatch):
    from speech_to_speech.baseHandler import BaseHandler

    cleaned = []

    class FailingHandler(BaseHandler):
        def setup(self):
            self.client = object()

        def should_process_input(self, item):
            raise RuntimeError("SECRET_PROVIDER")

        def cleanup(self):
            cleaned.append(True)
            self.client = None

    def build(**kwargs):
        handler = FailingHandler(kwargs["stop_event"], Queue(), Queue())
        handler.queue_in.put(object())
        kwargs["resource_ledger"].append(handler)
        return _unit(kwargs["index"], kwargs["stop_event"], [handler])

    config.pipelines = {"secondary": config.pipelines["secondary"]}
    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    result = runtime.build_configured_runtime(config, Event())
    try:
        result.start()
        with pytest.raises(runtime.ConfiguredRuntimeError):
            result.wait()
    except runtime.ConfiguredRuntimeError:
        pass
    assert cleaned == [True]
    assert result.state == "failed"


def test_yaml_cli_and_flat_json_effective_settings_are_equal(tmp_path):
    import json

    from speech_to_speech.s2s_pipeline import parse_arguments, prepare_all_args

    llm = {"model_name": "", "responses_api_api_key": "", "responses_api_stream": False, "chat_size": 0}
    stt = {"openai_stt_api_key": "", "openai_stt_language": None}
    tts = {"openai_tts_api_key": "", "openai_tts_speed": 0.0}
    path = tmp_path / "config.yaml"
    path.write_text(BASE)
    document = load_config(path)
    document.data["blocks"]["llm"]["settings"] = llm
    document.data["blocks"]["stt"]["settings"] = stt
    document.data["blocks"]["tts"]["settings"] = tts
    configured = runtime._adapt(resolve_config(document, names=["secondary"], environ={}))["secondary"]
    flat = tmp_path / "legacy.json"
    flat.write_text(
        json.dumps({"stt": "openai", "llm_backend": "chat-completions", "tts": "openai", **llm, **stt, **tts})
    )
    legacy_json = parse_arguments([str(flat)])
    legacy_cli = parse_arguments(
        [
            "--stt",
            "openai",
            "--llm_backend",
            "chat-completions",
            "--tts",
            "openai",
            "--model_name",
            "",
            "--responses_api_api_key",
            "",
            "--responses_api_stream",
            "false",
            "--chat_size",
            "0",
            "--openai_stt_api_key",
            "",
            "--openai_tts_api_key",
            "",
            "--openai_tts_speed",
            "0",
        ]
    )
    for legacy in (legacy_json, legacy_cli):
        prepare_all_args(legacy)
        assert configured.module_kwargs == legacy.module_kwargs
        assert configured.vad_handler_kwargs == legacy.vad_handler_kwargs
        assert configured.stt_backend.config == legacy.stt_backend.config
        assert configured.llm_backend.config == legacy.llm_backend.config
        assert configured.tts_backend.config == legacy.tts_backend.config


def test_preset_defaults_and_device_override_preserve_settings(tmp_path, builder):
    path = tmp_path / "config.yaml"
    path.write_text(BASE)
    document = load_config(path)
    document.data["blocks"]["llm"] = {"kind": "llm", "backend": "mlx-lm", "settings": {"llm_device": "cuda"}}
    document.data["blocks"]["stt"] = {"kind": "stt", "backend": "whisper", "settings": {"stt_device": "cuda"}}
    document.data["pipelines"]["secondary"]["options"] = {"mac_optimal_settings": True}
    config = resolve_config(document, names=["secondary"], environ={})
    original = copy.deepcopy(config)
    result = runtime.build_configured_runtime(config, Event())
    assert builder[0]["llm_backend"].config["device"] == "cuda"
    assert builder[0]["stt_backend"].config["device"] == "cuda"
    assert builder[0]["llm_backend"].config["model_name"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert config == original
    result.stop()
    builder.clear()
    config.pipelines["secondary"].options["device"] = "cpu"
    result = runtime.build_configured_runtime(config, Event())
    assert builder[0]["llm_backend"].config["device"] == "cpu"
    assert builder[0]["stt_backend"].config["device"] == "cpu"
    assert config.pipelines["secondary"].stages["llm"].settings["llm_device"] == "cuda"
    result.stop()


def test_generation_dictionary_replacement_preserves_explicit_empty(tmp_path, builder):
    path = tmp_path / "config.yaml"
    path.write_text(BASE)
    document = load_config(path)
    document.data["blocks"]["stt"] = {
        "kind": "stt",
        "backend": "mlx-audio-whisper",
        "settings": {"mlx_audio_whisper_gen_kwargs": {}},
    }
    config = resolve_config(document, names=["secondary"], environ={})
    result = runtime.build_configured_runtime(config, Event())
    assert builder[0]["stt_backend"].config["gen_kwargs"] == {}
    result.stop()


def test_local_client_failure_is_recorded_without_provider_logs(config, builder, monkeypatch, caplog):
    config.pipelines = {"primary": config.pipelines["primary"]}
    result = runtime.build_configured_server(config, Event(), local=True)
    monkeypatch.setattr(result.server, "run", lambda: result.server.stop_event.wait())

    async def failed_client(*args, **kwargs):
        raise RuntimeError("SECRET_CLIENT_ENDPOINT")

    monkeypatch.setattr("speech_to_speech.api.openai_realtime.audio_client.listen_and_play_realtime", failed_client)
    result.start()
    with pytest.raises(runtime.ConfiguredRuntimeError):
        result.wait()
    assert result.state == "failed"
    assert "SECRET_CLIENT_ENDPOINT" not in caplog.text


def test_monitor_start_failure_cleans_without_joining_unstarted_thread(config, builder, monkeypatch):
    result = runtime.build_configured_runtime(config, Event())
    handlers = [u.handlers[0] for pool in result.pools.values() for u in pool]
    original = runtime.Thread.start

    def start(thread):
        if thread.daemon:
            raise RuntimeError("SECRET_MONITOR_START")
        original(thread)

    monkeypatch.setattr(runtime.Thread, "start", start)
    with pytest.raises(runtime.ConfiguredRuntimeError):
        result.start()
    assert result.state == "failed"
    assert result.pools == {}
    assert all(handler.cleaned == 1 for handler in handlers)


@pytest.fixture
def viseme_chain(monkeypatch):
    import sys

    from speech_to_speech.baseHandler import BaseHandler
    from speech_to_speech.STV.w2v_stv_handler import Wav2Vec2STVHandler

    class IdleHandler(BaseHandler):
        pass

    class FakeTTS(IdleHandler):
        pass

    visemes = []
    cleaned = []
    original_setup = Wav2Vec2STVHandler.setup
    original_cleanup = Wav2Vec2STVHandler.cleanup

    def setup(handler, **kwargs):
        original_setup(handler, **kwargs, skip=True)
        handler._pending = [b"pending audio"]
        visemes.append(handler)

    def cleanup(handler):
        cleaned.append(handler)
        original_cleanup(handler)

    def create(selection, context):
        handler_type = FakeTTS if selection.kind == "tts" else IdleHandler
        return handler_type(context.stop_event, context.queue_in, context.queue_out)

    monkeypatch.setattr("speech_to_speech.s2s_pipeline.VADHandler", IdleHandler)
    monkeypatch.setattr("speech_to_speech.s2s_pipeline.TranscriptionNotifier", IdleHandler)
    monkeypatch.setattr("speech_to_speech.LLM.lm_output_processor.LMOutputProcessor", IdleHandler)
    monkeypatch.setattr("speech_to_speech.s2s_pipeline.create_backend_handler", create)
    monkeypatch.setattr(Wav2Vec2STVHandler, "setup", setup)
    monkeypatch.setattr(Wav2Vec2STVHandler, "cleanup", cleanup)
    monkeypatch.setattr(
        sys.modules[__name__], "QWEN3_LANGUAGE_ALIASES", {"en": "English", "zh": "Chinese"}, raising=False
    )
    return visemes, cleaned, FakeTTS


def test_configured_viseme_ledger_routing_and_tts_metadata(config, viseme_chain):
    visemes, cleaned, tts_type = viseme_chain
    config.pipelines = {"primary": config.pipelines["primary"]}
    pipeline = config.pipelines["primary"]
    pipeline.options["enable_visemes"] = True
    pipeline.stages["tts"].backend = "qwen3"
    pipeline.stages["tts"].settings = {}
    result = runtime.build_configured_runtime(config, Event())
    units = result.pools["primary"]
    assert len(visemes) == 2
    for unit, viseme in zip(units, visemes):
        assert unit.handlers[-1] is viseme
        assert isinstance(unit.handlers[-2], tts_type)
        assert unit.handlers[-2].queue_out is viseme.queue_in
        assert viseme.queue_out is unit.output_queue
        assert unit.service.tts_supported_languages == {"en", "zh"}
        assert viseme in result._resources
    assert units[0].handlers[-2].queue_out is not units[1].handlers[-2].queue_out
    result.stop()
    result.stop()
    assert cleaned == list(reversed(visemes))
    assert all(viseme._pending == [] for viseme in visemes)
    assert result.pools == {} and result._resources == []


def test_later_replica_failure_cleans_completed_viseme_handlers(config, viseme_chain, monkeypatch):
    from speech_to_speech import s2s_pipeline

    visemes, cleaned, _ = viseme_chain
    config.pipelines = {"primary": config.pipelines["primary"]}
    config.pipelines["primary"].options["enable_visemes"] = True
    original_build = s2s_pipeline._build_pipeline_unit
    original_error = ValueError("later replica failure")

    def build(**kwargs):
        unit = original_build(**kwargs)
        if kwargs["index"] == 1:
            raise original_error
        return unit

    monkeypatch.setattr(s2s_pipeline, "_build_pipeline_unit", build)
    with pytest.raises(runtime.ConfiguredRuntimeError) as exc:
        runtime.build_configured_runtime(config, Event())
    assert exc.value.__cause__ is original_error
    assert len(visemes) == 2
    assert cleaned == list(reversed(visemes))
    assert all(viseme.stop_event.is_set() and viseme._pending == [] for viseme in visemes)


@pytest.mark.parametrize("failure_mode", ["exception", "exit", "stop_event"])
def test_worker_readiness_and_replica_failure_isolation(config, monkeypatch, failure_mode):
    triggers = [Event() for _ in range(3)]

    class FailingHandler(Handler):
        def run(self):
            try:
                while not self.stop_event.wait(0.01):
                    if triggers[self.index].is_set():
                        if failure_mode == "exception":
                            raise RuntimeError("SECRET_WORKER_FAILURE")
                        if failure_mode == "stop_event":
                            self.stop_event.set()
                        return
            finally:
                self.cleanup()

    def build(**kwargs):
        handler = FailingHandler(kwargs["stop_event"])
        handler.index = kwargs["index"]
        kwargs["resource_ledger"].append(handler)
        return _unit(kwargs["index"], kwargs["stop_event"], [handler])

    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    result = runtime.build_configured_runtime(config, Event())
    units = [u for pool in result.pools.values() for u in pool]
    assert not result.ready.is_set()
    assert not result.wait_ready(0)
    assert [u.pipeline_name for u in units] == ["primary", "primary", "secondary"]
    assert [u.instance_index for u in units] == [0, 1, 0]
    assert all(u.lifecycle_state == "constructed" for u in units)
    result.start()
    try:
        assert result.wait_ready(2)
        assert all(u.healthy for u in units)
        triggers[0].set()
        assert units[0].stop_event.wait(2)
        assert units[0].lifecycle_state == "failed"
        assert result.ready.is_set()
        assert units[1].healthy and units[2].healthy
        triggers[1].set()
        assert units[1].stop_event.wait(2)
        for _ in range(100):
            if not result.ready.is_set():
                break
            Event().wait(0.01)
        assert not result.ready.is_set()
        assert units[2].healthy
        assert result.state == "running"
    finally:
        result.stop()
    assert all(u.stop_event.is_set() for u in units)


def test_server_readiness_requires_listener_startup(config, builder, monkeypatch):
    result = runtime.build_configured_server(config, Event())
    server = result.server
    release_listener = Event()

    def serve():
        release_listener.wait()
        if not server.stop_event.is_set():
            server.ready.set()
        server.stop_event.wait()

    monkeypatch.setattr(server, "run", serve)
    result.start()
    try:
        assert not result.wait_ready(0.05)
        release_listener.set()
        assert result.wait_ready(2)
        assert len(server.pipeline_pools["primary"]) == 2
        assert len(server.pipeline_pools["secondary"]) == 1
        assert server.pool == [u for pool in result.pools.values() for u in pool]
    finally:
        release_listener.set()
        result.stop()
    assert not result.ready.is_set()


def test_pipeline_unit_health_failure_is_atomic_and_irreversible():
    unit = _unit(0, Event(), [])
    unit.stop_event = Event()
    called = []
    unit.on_failure = lambda failed: called.append(failed)
    assert unit.healthy and unit.available
    unit.mark_failed()
    unit.mark_failed()
    assert unit.lifecycle_state == "failed"
    assert unit.stop_event.is_set()
    assert not unit.healthy and not unit.available
    assert called == [unit]


@pytest.mark.parametrize(
    "base,expected",
    [
        ("ws://example.test/v1/realtime", "ws://example.test/v1/pipelines/primary/realtime"),
        ("  ws://example.test/v1/realtime  ", "ws://example.test/v1/pipelines/primary/realtime"),
        (
            "wss://example.test/gateway/pipelines/tenant/v1/realtime",
            "wss://example.test/gateway/pipelines/tenant/v1/pipelines/primary/realtime",
        ),
        ("wss://example.test/gateway/v1/realtime", "wss://example.test/gateway/v1/pipelines/primary/realtime"),
        ("https://example.test/api/realtime", "https://example.test/api/pipelines/primary/realtime"),
        ("ws://example.test/v1/pipelines/primary/realtime", "ws://example.test/v1/pipelines/primary/realtime"),
    ],
)
def test_configured_client_named_url_preserves_model_and_base(config, base, expected):
    config.client.update(url=base, model="same-model")
    client = runtime.build_configured_client(config, pipeline_name="primary")
    assert client.url == expected
    assert client.model == "same-model"
    assert config.client["url"] == base


@pytest.mark.parametrize(
    "name,url",
    [
        ("unknown", "ws://example.test/v1/realtime"),
        ("primary", "ws://example.test/v1/pipelines/secondary/realtime"),
        ("", "ws://example.test/v1/realtime"),
    ],
)
def test_configured_client_rejects_unknown_or_conflicting_name(config, name, url):
    config.client["url"] = url
    with pytest.raises(ConfigurationError):
        runtime.build_configured_client(config, pipeline_name=name)


def test_real_listener_readiness_and_bind_failure(config, builder):
    import socket

    config.runtime["server"]["port"] = 0
    result = runtime.build_configured_server(config, Event())
    result.start()
    try:
        assert result.wait_ready(5)
        assert result.server.ready.is_set()
    finally:
        result.stop()
    assert not result.ready.is_set()
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        config.runtime["server"]["port"] = blocker.getsockname()[1]
        failed = runtime.build_configured_server(config, Event())
        try:
            failed.start()
            assert not failed.wait_ready(0.1)
            with pytest.raises(runtime.ConfiguredRuntimeError):
                failed.wait()
        finally:
            failed.stop()
        assert failed.state == "failed"


@pytest.mark.asyncio
async def test_named_websocket_transport_reports_unexpected_send_failure():
    from starlette.websockets import WebSocketState

    from speech_to_speech.api.openai_realtime.transports import WebSocketTransport

    async def send_json(_event):
        raise OSError("transport failed")

    websocket = SimpleNamespace(application_state=WebSocketState.CONNECTED, send_json=send_json)
    transport = WebSocketTransport(websocket, raise_errors=True)
    event = SimpleNamespace(model_dump=lambda: {})
    with pytest.raises(OSError):
        await transport.send_events([event])


@pytest.mark.parametrize("name", ["", 4, "bad/name", "bad{name}", "a" * 65])
def test_server_rejects_invalid_pipeline_identifiers(name):
    from speech_to_speech.api.openai_realtime.server import RealtimeServer

    unit = _unit(0, Event(), [])
    with pytest.raises(ValueError, match="identifiers"):
        RealtimeServer(Event(), [unit], pipeline_pools={name: [unit]})


def test_worker_initial_failure_stops_complete_activation(config, monkeypatch):
    class InitialFailure(Handler):
        def run(self):
            try:
                raise RuntimeError("SECRET_INITIAL_FAILURE")
            finally:
                self.cleanup()

    owned = []

    def build(**kwargs):
        handler = InitialFailure(kwargs["stop_event"])
        owned.append(handler)
        kwargs["resource_ledger"].append(handler)
        return _unit(kwargs["index"], kwargs["stop_event"], [handler])

    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", build)
    result = runtime.build_configured_runtime(config, Event())
    try:
        with pytest.raises(runtime.ConfiguredRuntimeError, match="startup"):
            result.start()
        assert not result.wait_ready(0)
    finally:
        result.stop()
    assert all(handler.cleaned == 1 for handler in owned)
    assert result.pools == {}


@pytest.mark.parametrize("proxy_mode", ["legacy", "partial", "empty"])
def test_server_rejects_incompatible_named_proxy_configuration(proxy_mode):
    from speech_to_speech.api.openai_realtime.server import RealtimeServer

    units = [_unit(index, Event(), []) for index in range(2)]
    kwargs = {"pipeline_pools": {"primary": [units[0]], "secondary": [units[1]]}}
    if proxy_mode == "legacy":
        kwargs["llm_proxy_config"] = SimpleNamespace()
    else:
        kwargs["llm_proxy_configs"] = {"primary": None} if proxy_mode == "partial" else {}
    with pytest.raises(ValueError, match="proxy"):
        RealtimeServer(Event(), units, **kwargs)


def test_stop_cannot_leave_delayed_readiness_published():
    from threading import Thread

    entered, release = Event(), Event()

    class DelayedReady(Event):
        def set(self):
            entered.set()
            assert release.wait(2)
            super().set()

    result = runtime.ConfiguredRuntime(Event())
    result.pools = {"primary": [_unit(0, Event(), [])]}
    result.state = "running"
    result.ready = DelayedReady()
    updater = Thread(target=result._update_ready)
    stopper = Thread(target=result.stop)
    updater.start()
    try:
        assert entered.wait(2)
        stopper.start()
        stopper.join(0.05)
    finally:
        release.set()
        updater.join(2)
        if stopper.ident is not None:
            stopper.join(2)
    assert not updater.is_alive() and not stopper.is_alive()
    assert result.state == "stopped"
    assert not result.ready.is_set()
