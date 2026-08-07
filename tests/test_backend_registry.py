import sys
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
