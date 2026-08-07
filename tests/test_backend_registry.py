import sys
from copy import deepcopy
from dataclasses import dataclass, fields
from queue import Queue
from threading import Event

import pytest

import speech_to_speech.s2s_pipeline as s2s_pipeline
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.backend_registry import (
    LLM_BACKENDS,
    STT_BACKENDS,
    TTS_BACKENDS,
    BackendCapabilities,
    BackendSelection,
    BackendSpec,
    HandlerContext,
    build_backend_registry,
    create_backend_handler,
    select_backend,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import (
    build_llm_proxy_config,
    parse_arguments,
    prepare_all_args,
    prepare_module_args,
)


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


def _factory(_context, config):
    return config


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
    assert STT_BACKENDS["none"].capabilities.bypasses_transcription_notifier
    assert not STT_BACKENDS["whisper"].capabilities.bypasses_transcription_notifier


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


def test_llm_proxy_validation_happens_before_pipeline_construction(monkeypatch):
    constructed = False

    def fake_build_pipeline(*_args, **_kwargs):
        nonlocal constructed
        constructed = True

    monkeypatch.setattr(s2s_pipeline, "setup_logger", lambda _level: None)
    monkeypatch.setattr(s2s_pipeline, "build_pipeline", fake_build_pipeline)

    with pytest.raises(ValueError, match="proxy support.*responses-api.*chat-completions"):
        s2s_pipeline.run_pipeline_command(
            "serve",
            ["--llm_backend", "transformers", "--enable_llm_proxy"],
        )

    assert not constructed


def test_test_backend_only_needs_config_factory_and_registry_entry():
    calls = []

    def factory(context, config):
        calls.append((context, config))
        return "handler"

    registry = build_backend_registry(
        "stt",
        [BackendSpec("fake", "stt", FakeArguments, factory, config_prefix="fake")],
    )
    parsed_config = FakeArguments(fake_option="selected")
    selection = select_backend(registry, "fake", parsed_config)

    assert create_backend_handler(selection, _context()) == "handler"
    assert selection.config == {"option": "selected", "gen_kwargs": {}}
    assert parsed_config.fake_option == "selected"
    assert calls[0][1] is selection.config


def test_new_stt_backend_gets_transcription_notifier_by_default(monkeypatch):
    stt_contexts = []

    class DummyHandler:
        def __init__(self, *_args, **_kwargs):
            pass

    class DummyNotifier(DummyHandler):
        pass

    def stt_factory(context, _config):
        stt_contexts.append(context)
        return object()

    def other_factory(_context, _config):
        return object()

    stt_spec = BackendSpec("future-stt", "stt", FakeArguments, stt_factory)
    llm_spec = BackendSpec("future-llm", "llm", FakeArguments, other_factory)
    tts_spec = BackendSpec("future-tts", "tts", FakeArguments, other_factory)
    stt_output_queue = Queue()
    text_prompt_queue = Queue()

    monkeypatch.setattr(s2s_pipeline, "VADHandler", DummyHandler)
    monkeypatch.setattr(s2s_pipeline, "TranscriptionNotifier", DummyNotifier)
    monkeypatch.setattr("speech_to_speech.LLM.lm_output_processor.LMOutputProcessor", DummyHandler)

    handlers = s2s_pipeline._build_handlers(
        stop_event=Event(),
        should_listen=Event(),
        recv_audio_chunks_queue=Queue(),
        spoken_prompt_queue=Queue(),
        stt_output_queue=stt_output_queue,
        text_prompt_queue=text_prompt_queue,
        lm_response_queue=Queue(),
        lm_processed_queue=Queue(),
        send_audio_chunks_queue=Queue(),
        text_output_queue=Queue(),
        module_kwargs=ModuleArguments(),
        vad_handler_kwargs=VADHandlerArguments(),
        stt_backend=BackendSelection(stt_spec, stt_spec.normalize(FakeArguments())),
        llm_backend=BackendSelection(llm_spec, llm_spec.normalize(FakeArguments())),
        tts_backend=BackendSelection(tts_spec, tts_spec.normalize(FakeArguments())),
        speculative_turns=SpeculativeTurnTracker(),
        cancel_scope=CancelScope(),
        pipeline_index=0,
    )

    assert stt_contexts[0].queue_out is stt_output_queue
    assert any(isinstance(handler, DummyNotifier) for handler in handlers)


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


def test_facebook_mms_options_are_normalized_for_handler_setup(monkeypatch):
    from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler

    captured = {}

    def fake_setup(self, _should_listen, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(FacebookMMSTTSHandler, "setup", fake_setup)
    args = parse_arguments(
        [
            "--tts",
            "facebookMMS",
            "--tts_language",
            "fr",
            "--facebook_mms_model_name",
            "acme/custom-mms",
        ]
    )

    assert args.tts_backend.config["language"] == "fr"
    assert "tts_language" not in args.tts_backend.config
    create_backend_handler(args.tts_backend, _context())
    assert captured["language"] == "fr"
    assert captured["model_name"] == "acme/custom-mms"


def test_facebook_mms_handler_honors_model_override(monkeypatch):
    from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler

    load_calls = []
    monkeypatch.setattr(
        FacebookMMSTTSHandler,
        "load_model",
        lambda self, language, model_name=None: load_calls.append((language, model_name)),
    )
    monkeypatch.setattr(FacebookMMSTTSHandler, "warmup", lambda self: None)
    handler = FacebookMMSTTSHandler.__new__(FacebookMMSTTSHandler)

    handler.setup(Event(), model_name="acme/custom-mms", language="fr", device="cpu")

    assert load_calls == [("fr", "acme/custom-mms")]
    assert handler._initial_model_name == "acme/custom-mms"


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
    assert args.stt_backend.config["language"] is None
    assert "mlx_audio_whisper_model_name" not in args.stt_backend.config
    assert "pocket_tts_voice" not in args.tts_backend.config
    assert "--language" in caplog.text
    assert "--mlx_audio_whisper_model_name" in caplog.text
    assert "--pocket_tts_voice" in caplog.text
    assert "unused/whisper" not in caplog.text


def test_parser_still_rejects_unknown_options():
    with pytest.raises(ValueError, match="--unknown_backend_option"):
        parse_arguments(["--unknown_backend_option", "value"])


@pytest.mark.parametrize("selector", ["--stt", "--llm_backend", "--tts"])
def test_parser_reports_invalid_backend_selectors_with_argparse(selector, capsys):
    with pytest.raises(SystemExit, match="2"):
        parse_arguments([selector, "not-a-backend"])

    stderr = capsys.readouterr().err
    assert "usage: speech-to-speech serve" in stderr
    assert "invalid choice: 'not-a-backend'" in stderr


@pytest.mark.parametrize("backend_name", ["whisper", "whisper-mlx", "mlx-audio-whisper"])
def test_common_language_flag_reaches_compatible_stt_backends(backend_name, caplog):
    args = parse_arguments(["--stt", backend_name, "--language", "de"])

    assert args.stt_backend.config["language"] == "de"
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
    def missing(_context, _config):
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
