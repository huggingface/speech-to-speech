import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest
from openai.types.realtime import RealtimeErrorEvent, SessionUpdateEvent

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
    assert STT_BACKENDS["openai"].kind == "stt"
    assert STT_BACKENDS["openai-realtime"].capabilities.streams_audio_chunks
    assert STT_BACKENDS["vllm-realtime"].capabilities.streams_audio_chunks
    assert not STT_BACKENDS["openai"].capabilities.streams_audio_chunks
    assert LLM_BACKENDS["responses-api"].kind == "llm"


@pytest.mark.parametrize(
    ("model_name", "supported_languages", "expected"),
    [
        ("/models/english-checkpoint", ["en"], {"en"}),
        ("/models/multilingual.en", ["en", "es"], {"en", "es"}),
    ],
)
def test_session_language_validation_uses_loaded_faster_whisper_capabilities(model_name, supported_languages, expected):
    selection = BackendSelection(STT_BACKENDS["faster-whisper"], {"model_name": model_name})
    handler = SimpleNamespace(model=SimpleNamespace(supported_languages=supported_languages))

    assert s2s_pipeline._stt_session_languages(selection, handler) == expected
    assert TTS_BACKENDS["qwen3"].kind == "tts"
    assert TTS_BACKENDS["openai"].kind == "tts"
    assert TTS_BACKENDS["supertonic"].required_extra == "supertonic"
    assert LLM_BACKENDS["responses-api"].capabilities.supports_llm_proxy
    assert LLM_BACKENDS["chat-completions"].capabilities.supports_llm_proxy
    assert LLM_BACKENDS["chat-completions"].capabilities.supports_audio_input
    assert not LLM_BACKENDS["transformers"].capabilities.supports_audio_input
    assert STT_BACKENDS["none"].capabilities.bypasses_transcription_notifier
    assert not STT_BACKENDS["whisper"].capabilities.bypasses_transcription_notifier


def test_parakeet_session_language_selection_remains_unsupported():
    selection = BackendSelection(STT_BACKENDS["parakeet-tdt"], {})

    assert s2s_pipeline._stt_session_languages(selection, SimpleNamespace()) == set()


@pytest.mark.parametrize(
    ("model_name", "supported_languages", "accepted"),
    [
        ("/models/english-checkpoint", ["en"], False),
        ("/models/multilingual.en", ["en", "es"], True),
    ],
)
def test_faster_whisper_loaded_languages_control_session_update(monkeypatch, model_name, supported_languages, accepted):
    args = parse_arguments(
        ["--stt", "faster-whisper", "--tts", "openai", "--faster_whisper_stt_model_name", model_name]
    )
    stt_handler = SimpleNamespace(model=SimpleNamespace(supported_languages=supported_languages))
    monkeypatch.setattr(
        s2s_pipeline,
        "_build_handlers",
        lambda **_kwargs: [SimpleNamespace(), stt_handler, SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
    )
    unit = s2s_pipeline._build_pipeline_unit(
        index=0,
        stop_event=Event(),
        module_kwargs=args.module_kwargs,
        vad_handler_kwargs=args.vad_handler_kwargs,
        stt_backend=args.stt_backend,
        llm_backend=args.llm_backend,
        tts_backend=args.tts_backend,
    )
    unit.service.tts_supported_languages = {"en", "es"}
    conn_id = unit.service.register()
    update = SessionUpdateEvent.model_validate(
        {
            "type": "session.update",
            "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": "es"}}}},
        }
    )

    result = unit.service.handle_session_update(conn_id, update)

    if accepted:
        assert result is None
        assert unit.service.build_session_updated(conn_id).session.audio.input.transcription.language == "es"
    else:
        assert isinstance(result, RealtimeErrorEvent)
        assert "STT" in result.error.message
        assert unit.service._state(conn_id).runtime_config.selected_language is None


@pytest.mark.parametrize(
    ("is_multilingual", "language", "accepted"),
    [(False, "en", False), (False, "es", False), (True, "es", True), (None, "es", True)],
)
def test_transformers_whisper_session_language_uses_loaded_generation_config(
    monkeypatch, is_multilingual, language, accepted
):
    args = parse_arguments(["--stt", "whisper", "--tts", "openai"])
    stt_handler = SimpleNamespace(
        model=SimpleNamespace(generation_config=SimpleNamespace(is_multilingual=is_multilingual))
    )
    monkeypatch.setattr(
        s2s_pipeline,
        "_build_handlers",
        lambda **_kwargs: [SimpleNamespace(), stt_handler, SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
    )
    unit = s2s_pipeline._build_pipeline_unit(
        index=0,
        stop_event=Event(),
        module_kwargs=args.module_kwargs,
        vad_handler_kwargs=args.vad_handler_kwargs,
        stt_backend=args.stt_backend,
        llm_backend=args.llm_backend,
        tts_backend=args.tts_backend,
    )
    unit.service.tts_supported_languages = {"en", "es"}
    conn_id = unit.service.register()
    update = SessionUpdateEvent.model_validate(
        {
            "type": "session.update",
            "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": language}}}},
        }
    )

    result = unit.service.handle_session_update(conn_id, update)

    if accepted:
        assert result is None
        assert unit.service.build_session_updated(conn_id).session.audio.input.transcription.language == language
    else:
        assert isinstance(result, RealtimeErrorEvent)
        assert "STT" in result.error.message
        assert unit.service._state(conn_id).runtime_config.selected_language is None


@pytest.mark.parametrize(("setup_language", "rejected"), [("en", True), (None, False)])
def test_openai_realtime_stt_auto_reset_is_rejected_only_when_setup_hint_cannot_be_cleared(
    monkeypatch, setup_language, rejected
):
    cli = ["--stt", "openai-realtime", "--tts", "openai"]
    if setup_language is not None:
        cli.extend(["--openai_realtime_stt_language", setup_language])
    args = parse_arguments(cli)
    monkeypatch.setattr(s2s_pipeline, "_build_handlers", lambda **_kwargs: [SimpleNamespace() for _ in range(5)])
    unit = s2s_pipeline._build_pipeline_unit(
        index=0,
        stop_event=Event(),
        module_kwargs=args.module_kwargs,
        vad_handler_kwargs=args.vad_handler_kwargs,
        stt_backend=args.stt_backend,
        llm_backend=args.llm_backend,
        tts_backend=args.tts_backend,
    )
    conn_id = unit.service.register()
    update = SessionUpdateEvent.model_validate(
        {
            "type": "session.update",
            "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": "Auto"}}}},
        }
    )

    error = unit.service.handle_session_update(conn_id, update)

    if rejected:
        assert isinstance(error, RealtimeErrorEvent)
        assert "Auto" in error.error.message and "STT" in error.error.message
        assert unit.service._state(conn_id).runtime_config.selected_language is None
    else:
        assert error is None
        assert unit.service.build_session_updated(conn_id).session.audio.input.transcription.language == "auto"


def test_omnivoice_tts_backend_is_registered_as_optional():
    spec = TTS_BACKENDS["omnivoice"]

    assert spec.kind == "tts"
    assert spec.required_extra == "omnivoice"


def test_sensevoice_stt_backend_is_registered_with_its_optional_extra():
    spec = STT_BACKENDS["sense-voice"]

    assert spec.kind == "stt"
    assert spec.config_prefix == "sense_voice_stt"
    assert spec.required_extra == "sensevoice"


def test_sensevoice_cli_config_is_normalized_for_the_handler():
    args = parse_arguments(
        [
            "--stt",
            "sense-voice",
            "--sense_voice_stt_device",
            "cpu",
            "--sense_voice_stt_language",
            "yue",
        ]
    )

    assert args.stt_backend.name == "sense-voice"
    assert args.stt_backend.config == {
        "model_name": "FunAudioLLM/SenseVoiceSmall",
        "device": "cpu",
        "language": "yue",
        "gen_kwargs": {},
    }


def test_omnivoice_cli_config_is_normalized_for_the_handler():
    args = parse_arguments(
        [
            "--tts",
            "omnivoice",
            "--omnivoice_model_name",
            "local/omnivoice",
            "--omnivoice_device",
            "xpu",
            "--omnivoice_dtype",
            "bfloat16",
            "--omnivoice_ref_audio",
            "voice.wav",
            "--omnivoice_ref_text",
            "Reference transcript.",
            "--omnivoice_num_steps",
            "16",
            "--omnivoice_speed",
            "1.25",
            "--omnivoice_blocksize",
            "256",
        ]
    )

    assert args.tts_backend.name == "omnivoice"
    assert args.tts_backend.config == {
        "model_name": "local/omnivoice",
        "device": "xpu",
        "dtype": "bfloat16",
        "ref_audio": "voice.wav",
        "ref_text": "Reference transcript.",
        "voice_clone_prompt": None,
        "ref_voices_dir": None,
        "instruct": None,
        "language": None,
        "num_steps": 16,
        "speed": 1.25,
        "blocksize": 256,
        "gen_kwargs": {},
    }


def test_registry_rejects_duplicate_names_and_wrong_kinds():
    spec = BackendSpec("fake", "stt", FakeArguments, _factory)

    with pytest.raises(ValueError, match="Duplicate stt backend name"):
        build_backend_registry("stt", [spec, spec])
    with pytest.raises(ValueError, match="expected 'llm'"):
        build_backend_registry("llm", [spec])


def test_supertonic_cli_config_is_normalized_for_the_handler():
    args = parse_arguments(
        [
            "--tts",
            "supertonic",
            "--supertonic_tts_voice",
            "F3",
            "--supertonic_tts_lang",
            "fr",
            "--supertonic_tts_speed",
            "1.2",
            "--supertonic_tts_blocksize",
            "256",
        ]
    )

    assert args.tts_backend.name == "supertonic"
    assert args.tts_backend.config == {
        "voice": "F3",
        "lang": "fr",
        "speed": 1.2,
        "blocksize": 256,
        "gen_kwargs": {},
    }


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


def test_openai_tts_backend_constructs_through_registry(monkeypatch):
    from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler

    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    args = parse_arguments(["--tts", "openai"])
    tts = create_backend_handler(args.tts_backend, _context())

    assert isinstance(tts, OpenAICompatibleTTSHandler)


def test_openai_tts_factory_receives_assistant_language_opt_in(monkeypatch):
    captured = {}

    class FakeHandler:
        def __init__(self, *_args, setup_kwargs, **_kwargs):
            captured.update(setup_kwargs)

    monkeypatch.setattr("speech_to_speech.backend_registry._load_handler", lambda *_args: FakeHandler)
    context = replace(_context(), detect_llm_output_language=True)
    create_backend_handler(parse_arguments(["--tts", "openai"]).tts_backend, context)

    assert captured["detect_llm_output_language"] is True


def test_qwen3_tts_factory_receives_assistant_language_opt_in(monkeypatch):
    captured = {}

    class FakeHandler:
        def __init__(self, *_args, setup_kwargs, **_kwargs):
            captured.update(setup_kwargs)

    monkeypatch.setattr("speech_to_speech.backend_registry._load_handler", lambda *_args: FakeHandler)
    context = replace(_context(), detect_llm_output_language=True)
    create_backend_handler(parse_arguments(["--tts", "qwen3"]).tts_backend, context)

    assert captured["detect_llm_output_language"] is True


def test_openai_stt_backend_constructs_through_registry(monkeypatch):
    from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler

    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    args = parse_arguments(["--stt", "openai"])
    stt = create_backend_handler(args.stt_backend, _context())

    assert isinstance(stt, OpenAICompatibleSTTHandler)


@pytest.mark.parametrize(
    ("backend_name", "expected_type", "expected_rate"),
    [
        ("openai-realtime", "OpenAIRealtimeSTTHandler", 24000),
        ("vllm-realtime", "VLLMRealtimeSTTHandler", 16000),
    ],
)
def test_streaming_stt_backends_parse_and_construct_through_registry(
    backend_name,
    expected_type,
    expected_rate,
):
    args = parse_arguments(["--stt", backend_name])

    handler = create_backend_handler(args.stt_backend, _context())

    assert type(handler).__name__ == expected_type
    assert handler.audio_sample_rate == expected_rate
    assert args.stt_backend.config["base_url"].endswith("/v1")
    handler.cleanup()


def test_streaming_stt_backend_is_attached_to_vad_without_changing_notifier_path(monkeypatch):
    class DummyHandler:
        def __init__(self, *_args, **_kwargs):
            pass

    class DummyNotifier(DummyHandler):
        pass

    streaming_handler = object()

    def stt_factory(_context, _config):
        return streaming_handler

    def other_factory(_context, _config):
        return object()

    stt_spec = BackendSpec(
        "streaming-stt",
        "stt",
        FakeArguments,
        stt_factory,
        capabilities=BackendCapabilities(streams_audio_chunks=True),
    )
    llm_spec = BackendSpec("future-llm", "llm", FakeArguments, other_factory)
    tts_spec = BackendSpec("future-tts", "tts", FakeArguments, other_factory)

    monkeypatch.setattr(s2s_pipeline, "VADHandler", DummyHandler)
    monkeypatch.setattr(s2s_pipeline, "TranscriptionNotifier", DummyNotifier)
    monkeypatch.setattr("speech_to_speech.LLM.lm_output_processor.LMOutputProcessor", DummyHandler)

    handlers = s2s_pipeline._build_handlers(
        stop_event=Event(),
        should_listen=Event(),
        recv_audio_chunks_queue=Queue(),
        spoken_prompt_queue=Queue(),
        stt_output_queue=Queue(),
        text_prompt_queue=Queue(),
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

    assert handlers[0].streaming_stt_sink is streaming_handler
    assert any(isinstance(handler, DummyNotifier) for handler in handlers)


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
            "--pocket_tts_language",
            "french_24l",
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
    assert args.tts_backend.config["language"] == "french_24l"
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


def test_facebook_mms_restores_custom_model_when_returning_to_initial_language(monkeypatch):
    from speech_to_speech.pipeline.messages import TTSInput
    from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler

    load_calls = []

    def fake_load_model(self, language, model_name=None):
        load_calls.append((language, model_name))
        self.language = language
        self.model_name = model_name or f"default/{language}"

    monkeypatch.setattr(FacebookMMSTTSHandler, "load_model", fake_load_model)
    monkeypatch.setattr(FacebookMMSTTSHandler, "generate_audio", lambda self, _text: None)
    handler = FacebookMMSTTSHandler.__new__(FacebookMMSTTSHandler)
    handler._initial_language = "en"
    handler._initial_model_name = "acme/custom-mms"
    handler.language = "en"
    handler.model_name = "acme/custom-mms"
    handler.cancel_scope = None

    list(handler.process(TTSInput(text="Bonjour", language_code="fr")))
    list(handler.process(TTSInput(text="Hello", language_code="en")))

    assert load_calls == [("fr", None), ("en", "acme/custom-mms")]


def test_facebook_mms_session_reset_restores_custom_model_for_same_language(monkeypatch):
    from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler

    load_calls = []
    monkeypatch.setattr(
        FacebookMMSTTSHandler,
        "load_model",
        lambda self, language, model_name=None: load_calls.append((language, model_name)),
    )
    handler = FacebookMMSTTSHandler.__new__(FacebookMMSTTSHandler)
    handler._initial_language = "en"
    handler._initial_model_name = "acme/custom-mms"
    handler.language = "en"
    handler.model_name = "facebook/mms-tts-eng"

    handler.on_session_end()

    assert load_calls == [("en", "acme/custom-mms")]


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


def test_registry_import_keeps_backend_modules_lazy():
    # A fresh process observes the initial import without clearing its evidence
    # or disturbing the backend modules used by other tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import speech_to_speech.backend_registry

eager_backends = [
    name for name in sys.modules
    if name.startswith(("speech_to_speech.STT.", "speech_to_speech.TTS."))
    or (name.startswith("speech_to_speech.LLM.") and name.endswith("language_model"))
]
assert not eager_backends, f"Registry imported backend modules: {eager_backends}"
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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
