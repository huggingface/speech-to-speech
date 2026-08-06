import sys
import threading
from copy import deepcopy
from dataclasses import dataclass, fields
from queue import Queue
from threading import Event

import pytest

from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.backend_registry import (
    LLM_BACKENDS,
    STT_BACKENDS,
    TTS_BACKENDS,
    BackendCapabilities,
    BackendSelection,
    BackendSpec,
    HandlerContext,
    SharedBackendResources,
    build_backend_registry,
    create_backend_handler,
    select_backend,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import (
    build_llm_proxy_config,
    build_local_pipeline,
    build_pipeline,
    parse_arguments,
    prepare_all_args,
    prepare_module_args,
)
from speech_to_speech.utils.thread_manager import ThreadManager


@dataclass
class FakeArguments:
    fake_option: str = "default"


def _context() -> HandlerContext:
    return HandlerContext(
        stop_event=Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        text_output_queue=Queue(),
        should_listen=Event(),
        cancel_scope=CancelScope(),
        speculative_turns=SpeculativeTurnTracker(),
        pipeline_index=0,
        sample_rate=16000,
        enable_live_transcription=False,
        live_transcription_update_interval=0.5,
    )


def _factory(_context, config, shared):
    return config, shared


def test_builtin_registry_lookup_and_cli_choices_share_one_catalog():
    module_fields = {config_field.name: config_field for config_field in fields(ModuleArguments)}

    assert tuple(STT_BACKENDS) == module_fields["stt"].metadata["choices"]
    assert tuple(LLM_BACKENDS) == module_fields["llm_backend"].metadata["choices"]
    assert tuple(TTS_BACKENDS) == module_fields["tts"].metadata["choices"]
    assert STT_BACKENDS["parakeet-tdt"].kind == "stt"
    assert LLM_BACKENDS["responses-api"].kind == "llm"
    assert TTS_BACKENDS["qwen3"].kind == "tts"
    assert LLM_BACKENDS["responses-api"].capabilities.supports_llm_proxy
    assert LLM_BACKENDS["chat-completions"].capabilities.supports_llm_proxy
    assert LLM_BACKENDS["chat-completions"].capabilities.supports_audio_input
    assert not LLM_BACKENDS["transformers"].capabilities.supports_audio_input


def test_registry_rejects_duplicate_names_and_wrong_kinds():
    spec = BackendSpec("fake", "stt", FakeArguments, _factory)

    with pytest.raises(ValueError, match="Duplicate stt backend name"):
        build_backend_registry("stt", [spec, spec])
    with pytest.raises(ValueError, match="expected 'llm'"):
        build_backend_registry("llm", [spec])


def test_audio_input_validation_uses_registry_capability_not_backend_name():
    spec = BackendSpec(
        "future-audio-backend",
        "llm",
        FakeArguments,
        _factory,
        capabilities=BackendCapabilities(supports_audio_input=True),
    )
    selection = BackendSelection(spec, spec.normalize(FakeArguments()))
    module_args = ModuleArguments(stt="none", llm_backend=selection.name)

    prepare_module_args(module_args, selection)


def test_llm_proxy_validation_uses_registry_capability():
    args = parse_arguments(["--llm_backend", "transformers"])

    with pytest.raises(ValueError, match="proxy support.*responses-api.*chat-completions"):
        build_llm_proxy_config(args.module_kwargs, args.llm_backend)


def test_test_backend_only_needs_config_factory_and_registry_entry():
    calls = []

    def factory(context, config, shared):
        calls.append((context, config, shared))
        return "handler"

    registry = build_backend_registry(
        "stt",
        [BackendSpec("fake", "stt", FakeArguments, factory, config_prefix="fake")],
    )
    parsed_config = FakeArguments(fake_option="selected")
    selection = select_backend(registry, "fake", parsed_config)

    assert create_backend_handler(selection, _context(), "shared") == "handler"
    assert selection.config == {"option": "selected", "gen_kwargs": {}}
    assert parsed_config.fake_option == "selected"
    assert calls[0][1] is selection.config


def test_parser_carries_only_selected_normalized_configs():
    args = parse_arguments(
        [
            "--stt",
            "mlx-audio-whisper",
            "--mlx_audio_whisper_model_name",
            "custom/whisper",
            "--language",
            "auto",
            "--llm_backend",
            "transformers",
            "--llm_gen_max_new_tokens",
            "64",
            "--tts",
            "pocket",
            "--pocket_tts_voice",
            "alba",
        ]
    )

    assert args.stt_backend.name == "mlx-audio-whisper"
    assert args.stt_backend.config == {
        "model_name": "custom/whisper",
        "language": "auto",
        "gen_kwargs": {},
    }
    assert args.llm_backend.name == "transformers"
    assert args.llm_backend.config["gen_kwargs"]["max_new_tokens"] == 64
    assert args.tts_backend.name == "pocket"
    assert args.tts_backend.config["voice"] == "alba"
    assert not hasattr(args, "whisper_stt_handler_kwargs")
    assert not hasattr(args, "qwen3_tts_handler_kwargs")


def test_parser_warning_ignores_known_options_for_inactive_backends(caplog):
    args = parse_arguments(
        [
            "--stt",
            "parakeet-tdt",
            "--mlx_audio_whisper_model_name",
            "unused/whisper",
            "--language=auto",
            "--tts",
            "qwen3",
            "--pocket_tts_voice",
            "alba",
        ]
    )

    assert args.stt_backend.name == "parakeet-tdt"
    assert args.tts_backend.name == "qwen3"
    assert args.stt_backend.config["language"] == "auto"
    assert "mlx_audio_whisper_model_name" not in args.stt_backend.config
    assert "pocket_tts_voice" not in args.tts_backend.config
    assert "--language" not in caplog.text
    assert "--mlx_audio_whisper_model_name" in caplog.text
    assert "--pocket_tts_voice" in caplog.text
    assert "unused/whisper" not in caplog.text


def test_parser_still_rejects_unknown_options():
    with pytest.raises(ValueError, match="--unknown_backend_option"):
        parse_arguments(["--unknown_backend_option", "value"])


@pytest.mark.parametrize(
    "backend_name",
    ["whisper", "whisper-mlx", "mlx-audio-whisper", "faster-whisper", "parakeet-tdt"],
)
def test_common_language_flag_reaches_supported_stt_backends(backend_name, caplog):
    args = parse_arguments(["--stt", backend_name, "--language", "de"])

    config = args.stt_backend.config
    language = config["language"] if "language" in config else config["gen_kwargs"]["language"]
    assert language == "de"
    assert "Ignoring options for inactive backends" not in caplog.text


@pytest.mark.parametrize(
    ("kind", "backend_name"),
    [
        *(("stt", name) for name in STT_BACKENDS),
        *(("llm", name) for name in LLM_BACKENDS),
        *(("tts", name) for name in TTS_BACKENDS),
    ],
)
def test_global_device_only_updates_device_aware_builtin_configs(kind, backend_name):
    selector = {"stt": "--stt", "llm": "--llm_backend", "tts": "--tts"}[kind]
    argv = ["--device", "cpu", selector, backend_name]
    if kind == "stt" and backend_name == "none":
        argv.extend(["--llm_backend", "chat-completions"])

    args = parse_arguments(argv)
    field_name = f"{kind}_backend"
    before = deepcopy(getattr(args, field_name).config)

    prepare_all_args(args)

    expected = deepcopy(before)
    if "device" in expected:
        expected["device"] = "cpu"
    assert getattr(args, field_name).config == expected


def test_global_device_does_not_reach_mlx_audio_whisper_setup(monkeypatch):
    from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler

    captured = {}

    def fake_setup(self, model_name, language, gen_kwargs):
        captured.update(model_name=model_name, language=language, gen_kwargs=gen_kwargs)

    monkeypatch.setattr(MLXAudioWhisperSTTHandler, "setup", fake_setup)
    args = parse_arguments(
        [
            "--device",
            "mps",
            "--stt",
            "mlx-audio-whisper",
            "--language",
            "auto",
        ]
    )
    prepare_all_args(args)

    create_backend_handler(args.stt_backend, _context())

    assert captured == {
        "model_name": "mlx-community/whisper-large-v3-turbo",
        "language": "auto",
        "gen_kwargs": {},
    }


def test_factories_keep_backend_modules_lazy():
    module_names = [
        "speech_to_speech.STT.whisper_stt_handler",
        "speech_to_speech.LLM.language_model",
        "speech_to_speech.TTS.chatTTS_handler",
    ]
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    assert STT_BACKENDS["whisper"].create_handler is not None
    assert LLM_BACKENDS["transformers"].create_handler is not None
    assert TTS_BACKENDS["chatTTS"].create_handler is not None
    assert all(module_name not in sys.modules for module_name in module_names)


def test_dependency_error_names_backend_and_required_extra():
    def missing(_context, _config, _shared):
        raise ImportError("missing package")

    spec = BackendSpec(
        "optional",
        "tts",
        FakeArguments,
        missing,
        required_extra="optional-extra",
    )
    selection = BackendSelection(spec, spec.normalize(FakeArguments()))

    with pytest.raises(ImportError, match=r"optional.*tts.*speech-to-speech\[optional-extra\]"):
        create_backend_handler(selection, _context())


def test_shared_resource_is_reused_and_cleanup_is_idempotent():
    seen = []

    class Resource:
        close_calls = 0

        def close(self):
            self.close_calls += 1

    resource = Resource()

    def factory(_context, _config, shared):
        seen.append(shared)
        return object()

    spec = BackendSpec(
        "shared",
        "stt",
        FakeArguments,
        factory,
        create_shared=lambda _config: resource,
    )
    selection = BackendSelection(spec, spec.normalize(FakeArguments()))
    owner = SharedBackendResources.create([selection])

    create_backend_handler(selection, _context(), owner.get(selection))
    create_backend_handler(selection, _context(), owner.get(selection))
    owner.close()
    owner.close()

    assert seen == [resource, resource]
    assert resource.close_calls == 1


def test_partial_shared_construction_cleans_already_created_resources():
    cleaned = []
    stt_spec = BackendSpec(
        "first",
        "stt",
        FakeArguments,
        _factory,
        create_shared=lambda _config: "stt-resource",
        cleanup_shared=cleaned.append,
    )

    def fail(_config):
        raise RuntimeError("construction failed")

    llm_spec = BackendSpec(
        "second",
        "llm",
        FakeArguments,
        _factory,
        create_shared=fail,
    )
    selections = [
        BackendSelection(stt_spec, stt_spec.normalize(FakeArguments())),
        BackendSelection(llm_spec, llm_spec.normalize(FakeArguments())),
    ]

    with pytest.raises(RuntimeError, match="construction failed"):
        SharedBackendResources.create(selections)
    assert cleaned == ["stt-resource"]


def test_partial_shared_construction_preserves_error_when_cleanup_fails(caplog):
    def cleanup(_resource):
        raise RuntimeError("cleanup failed")

    stt_spec = BackendSpec(
        "first",
        "stt",
        FakeArguments,
        _factory,
        create_shared=lambda _config: "stt-resource",
        cleanup_shared=cleanup,
    )

    def fail(_config):
        raise ValueError("construction failed")

    llm_spec = BackendSpec(
        "second",
        "llm",
        FakeArguments,
        _factory,
        create_shared=fail,
    )
    selections = [
        BackendSelection(stt_spec, stt_spec.normalize(FakeArguments())),
        BackendSelection(llm_spec, llm_spec.normalize(FakeArguments())),
    ]

    with pytest.raises(ValueError, match="construction failed"):
        SharedBackendResources.create(selections)
    assert "Failed to clean up shared backend resources" in caplog.text


def test_partial_pipeline_construction_preserves_error_when_cleanup_fails(monkeypatch, caplog):
    cleaned = []

    def cleanup(resource):
        cleaned.append(resource)
        raise ValueError("cleanup failed")

    spec = BackendSpec(
        "shared",
        "stt",
        FakeArguments,
        _factory,
        create_shared=lambda _config: "pipeline-resource",
        cleanup_shared=cleanup,
    )
    args = parse_arguments([])
    args.stt_backend = BackendSelection(spec, spec.normalize(FakeArguments()))
    monkeypatch.setattr(
        "speech_to_speech.s2s_pipeline._build_pipeline_unit",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("unit failed")),
    )

    with pytest.raises(RuntimeError, match="unit failed"):
        build_pipeline(args, Event())
    assert cleaned == ["pipeline-resource"]
    assert "Failed to clean up shared backend resources" in caplog.text


def test_local_client_construction_failure_cleans_server_resources(monkeypatch):
    cleaned = []
    monkeypatch.setattr(
        "speech_to_speech.s2s_pipeline.build_pipeline",
        lambda *_args, **_kwargs: ThreadManager([], cleanup_callbacks=[lambda: cleaned.append("closed")]),
    )

    class FailingClient:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("client failed")

    monkeypatch.setattr(
        "speech_to_speech.api.openai_realtime.audio_client.RealtimeAudioClient",
        FailingClient,
    )

    with pytest.raises(RuntimeError, match="client failed"):
        build_local_pipeline(parse_arguments([], command="local"), Event())
    assert cleaned == ["closed"]


def test_thread_manager_runs_all_cleanup_callbacks_once():
    cleaned = []
    manager = ThreadManager(
        [],
        cleanup_callbacks=[lambda: cleaned.append("first"), lambda: cleaned.append("second")],
    )

    manager.cleanup()
    manager.stop()

    assert cleaned == ["second", "first"]


def test_thread_manager_cleans_up_after_partial_start_failure(monkeypatch):
    class Handler:
        def __init__(self):
            self.stop_event = Event()

        def run(self):
            self.stop_event.wait()

    handlers = [Handler(), Handler()]
    cleaned = []
    real_start = threading.Thread.start
    start_calls = 0

    def fail_second_start(thread):
        nonlocal start_calls
        start_calls += 1
        if start_calls == 2:
            raise RuntimeError("thread startup failed")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_second_start)
    manager = ThreadManager(handlers, cleanup_callbacks=[lambda: cleaned.append("closed")])

    with pytest.raises(RuntimeError, match="thread startup failed"):
        manager.start()

    assert all(handler.stop_event.is_set() for handler in handlers)
    assert len(manager.threads) == 1
    assert all(not thread.is_alive() for thread in manager.threads)
    assert cleaned == ["closed"]


def test_thread_manager_tracks_thread_when_start_raises_after_launch(monkeypatch):
    class Handler:
        def __init__(self):
            self.stop_event = Event()
            self.entered_run = Event()

        def run(self):
            self.entered_run.set()
            self.stop_event.wait()

    handler = Handler()
    cleaned = []
    wrapper_launched = Event()
    allow_wrapper = Event()
    cleanup_finished = Event()
    real_start = threading.Thread.start
    real_join = threading.Thread.join
    start_calls = 0

    def cleanup():
        cleaned.append("closed")
        cleanup_finished.set()

    manager = ThreadManager([handler], cleanup_callbacks=[cleanup])
    run_handler = manager._run_handler

    def delay_handler_wrapper(selected_handler):
        wrapper_launched.set()
        allow_wrapper.wait()
        run_handler(selected_handler)

    monkeypatch.setattr(manager, "_run_handler", delay_handler_wrapper)

    def start_then_raise(thread):
        nonlocal start_calls
        start_calls += 1
        real_start(thread)
        if start_calls == 1:
            assert wrapper_launched.wait(timeout=1.0)
            raise KeyboardInterrupt("interrupted after launch")

    def fast_timed_join(thread, timeout=None):
        if timeout is None:
            real_join(thread)

    monkeypatch.setattr(threading.Thread, "join", fast_timed_join)
    monkeypatch.setattr(threading.Thread, "start", start_then_raise)

    with pytest.raises(KeyboardInterrupt, match="interrupted after launch"):
        manager.start()

    assert handler.stop_event.is_set()
    assert len(manager.threads) == 1

    assert cleaned == []
    allow_wrapper.set()
    assert cleanup_finished.wait(timeout=1.0)
    assert handler.entered_run.is_set()
    assert not manager.threads[0].is_alive()
    assert cleaned == ["closed"]


def test_thread_manager_wait_preserves_join_error_when_cleanup_fails(caplog):
    class FailingThread:
        name = "failing-thread"

        def __init__(self):
            self.alive = True

        def join(self, timeout=None):
            if timeout is None:
                raise RuntimeError("join failed")
            self.alive = False

        def is_alive(self):
            return self.alive

    class Handler:
        def __init__(self):
            self.stop_event = Event()

    def cleanup():
        raise ValueError("cleanup failed")

    handler = Handler()
    manager = ThreadManager([handler], cleanup_callbacks=[cleanup])
    manager.threads = [FailingThread()]  # type: ignore[list-item]

    with pytest.raises(RuntimeError, match="join failed"):
        manager.wait()
    assert handler.stop_event.is_set()
    assert "Failed to clean up resources after thread wait failed" in caplog.text


def test_thread_manager_defers_cleanup_until_slow_thread_finishes():
    release_thread = Event()
    cleanup_finished = Event()
    cleaned = []

    class SlowThread:
        name = "slow-thread"

        def __init__(self):
            self.alive = True

        def join(self, timeout=None):
            if timeout is None:
                release_thread.wait()
                self.alive = False

        def is_alive(self):
            return self.alive

    class Handler:
        def __init__(self):
            self.stop_event = Event()

    def cleanup():
        cleaned.append("closed")
        cleanup_finished.set()

    handler = Handler()
    manager = ThreadManager([handler], cleanup_callbacks=[cleanup])
    manager.threads = [SlowThread()]  # type: ignore[list-item]

    manager.stop()

    assert handler.stop_event.is_set()
    assert cleaned == []

    release_thread.set()
    assert cleanup_finished.wait(timeout=1.0)
    assert cleaned == ["closed"]


def test_thread_manager_falls_back_when_cleanup_reaper_cannot_start(monkeypatch, caplog):
    cleaned = []

    class SlowThread:
        name = "slow-thread"

        def __init__(self):
            self.alive = True

        def join(self, timeout=None):
            if timeout is None:
                self.alive = False

        def is_alive(self):
            return self.alive

    class Handler:
        def __init__(self):
            self.stop_event = Event()

    def fail_reaper_start(_thread):
        raise RuntimeError("cannot start reaper")

    monkeypatch.setattr(threading.Thread, "start", fail_reaper_start)
    manager = ThreadManager([Handler()], cleanup_callbacks=[lambda: cleaned.append("closed")])
    manager.threads = [SlowThread()]  # type: ignore[list-item]

    manager.stop()

    assert cleaned == ["closed"]
    assert "waiting for handler threads synchronously" in caplog.text


def test_thread_manager_stop_logs_cleanup_errors(caplog):
    def cleanup():
        raise RuntimeError("cleanup failed")

    manager = ThreadManager([], cleanup_callbacks=[cleanup])

    manager.stop()

    assert "Failed to clean up resources while stopping threads" in caplog.text
