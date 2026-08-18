import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest

from speech_to_speech.arguments_classes.chat_completions_language_model_arguments import (
    ChatCompletionsLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from speech_to_speech.arguments_classes.local_audio_arguments import LocalAudioArguments
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.realtime_server_arguments import RealtimeServerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.backend_registry import BackendSelection
from speech_to_speech.cli import main, parse_command, parse_talk_arguments
from speech_to_speech.s2s_pipeline import ParsedArguments, parse_arguments, prepare_all_args, prepare_module_args


def test_release_defaults_match_responses_api_parakeet_qwen3_profile():
    module_args = ModuleArguments()
    vad_args = VADHandlerArguments()
    responses_api_args = ResponsesApiLanguageModelHandlerArguments()
    qwen3_args = Qwen3TTSHandlerArguments()

    assert module_args.stt == "parakeet-tdt"
    assert module_args.mac_optimal_settings is False
    assert module_args.llm_backend == "responses-api"
    assert module_args.tts == "qwen3"
    assert module_args.log_level == "info"
    assert module_args.enable_live_transcription is True
    assert module_args.live_transcription_update_interval == 0.5

    assert vad_args.thresh == 0.6
    assert vad_args.min_silence_ms == 64
    assert vad_args.min_speech_ms == 384
    assert vad_args.min_speech_continuation_ms == 192
    assert vad_args.realtime_processing_pause == 0.5
    assert vad_args.smart_turn is True
    assert responses_api_args.model_name == "gpt-5.4-mini"
    assert responses_api_args.chat_size == 30
    assert responses_api_args.responses_api_stream is True
    assert responses_api_args.responses_api_audio_content_type == "input_audio"
    assert responses_api_args.responses_api_audio_history_turns == 1
    assert qwen3_args.qwen3_tts_model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert qwen3_args.qwen3_tts_speaker == "Aiden"
    assert qwen3_args.qwen3_tts_language == "auto"
    assert qwen3_args.qwen3_tts_backend == "ggml"
    assert qwen3_args.qwen3_tts_non_streaming_mode is True
    assert qwen3_args.qwen3_tts_ref_audio is None
    assert qwen3_args.qwen3_tts_ref_spk is None
    assert qwen3_args.qwen3_tts_ref_rvq is None
    assert qwen3_args.qwen3_tts_ggml_quantization == "BF16"
    assert qwen3_args.qwen3_tts_gguf_talker_path is None
    assert qwen3_args.qwen3_tts_gguf_codec_path is None
    assert qwen3_args.qwen3_tts_ref_cache_dir is None
    assert qwen3_args.qwen3_tts_mlx_quantization == "6bit"


def test_server_defaults_to_loopback():
    assert RealtimeServerArguments().host == "127.0.0.1"


def test_mac_optimal_settings_flag_does_not_select_a_command():
    args = parse_arguments(["--mac-optimal-settings"])

    assert args.module_kwargs.mac_optimal_settings is True
    assert not hasattr(args.module_kwargs, "mode")
    assert args.module_kwargs.device is None
    assert args.module_kwargs.stt == "parakeet-tdt"
    assert args.module_kwargs.llm_backend == "mlx-lm"
    assert args.module_kwargs.tts == "qwen3"
    assert args.llm_backend.config["device"] == "mps"
    assert args.tts_backend.config["device"] == "mps"
    assert args.llm_backend.config["model_name"] == "mlx-community/Qwen3-4B-Instruct-2507-bf16"


def test_mac_optimal_settings_routes_explicit_model_to_mlx_backend():
    args = parse_arguments(["--mac-optimal-settings", "--model_name", "custom/mlx-model"])

    assert args.module_kwargs.llm_backend == "mlx-lm"
    assert args.llm_backend.spec.config_type is LanguageModelHandlerArguments
    assert args.llm_backend.config["model_name"] == "custom/mlx-model"


def test_mac_optimal_settings_preserves_explicit_component_overrides():
    args = parse_arguments(
        [
            "--mac-optimal-settings",
            "--device",
            "cpu",
            "--stt",
            "whisper",
            "--llm_backend",
            "transformers",
            "--tts",
            "kokoro",
            "--model_name",
            "custom/transformers-model",
        ]
    )

    prepare_all_args(args)

    assert args.module_kwargs.device == "cpu"
    assert args.module_kwargs.stt == "whisper"
    assert args.module_kwargs.llm_backend == "transformers"
    assert args.module_kwargs.tts == "kokoro"
    assert args.llm_backend.config["device"] == "cpu"
    assert args.llm_backend.config["model_name"] == "custom/transformers-model"
    assert args.tts_backend.config["device"] == "cpu"


def test_mac_optimal_settings_preserves_explicit_component_device():
    args = parse_arguments(["--mac-optimal-settings", "--qwen3_tts_device", "cpu"])

    assert args.module_kwargs.device is None
    assert args.tts_backend.config["device"] == "cpu"


@pytest.mark.parametrize("flag", ["--local_mac_optimal_settings", "--mac_optimal_settings"])
def test_noncanonical_mac_optimal_settings_flags_are_rejected(flag):
    with pytest.raises(ValueError, match=flag):
        parse_arguments([flag])


# -- ParsedArguments dataclass tests ------------------------------------------

EXPECTED_FIELD_TYPES = {
    "module_kwargs": ModuleArguments,
    "realtime_server_kwargs": RealtimeServerArguments,
    "local_audio_kwargs": LocalAudioArguments,
    "vad_handler_kwargs": VADHandlerArguments,
    "stt_backend": BackendSelection,
    "llm_backend": BackendSelection,
    "tts_backend": BackendSelection,
}


def test_parsed_arguments_has_all_expected_fields():
    actual_fields = {f.name: f.type for f in fields(ParsedArguments)}
    assert set(actual_fields) == set(EXPECTED_FIELD_TYPES)


def test_parsed_arguments_field_types_match():
    for f in fields(ParsedArguments):
        assert f.type is EXPECTED_FIELD_TYPES[f.name], (
            f"Field {f.name!r}: expected {EXPECTED_FIELD_TYPES[f.name].__name__}, got {f.type}"
        )


def test_parse_arguments_default_backend_returns_openai_api():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert isinstance(args, ParsedArguments)
    assert isinstance(args.module_kwargs, ModuleArguments)
    assert args.llm_backend.name == "responses-api"
    assert args.llm_backend.spec.config_type is ResponsesApiLanguageModelHandlerArguments
    assert args.llm_backend.config["model_name"] == "gpt-5.4-mini"
    assert args.module_kwargs.llm_backend == "responses-api"
    assert args.vad_handler_kwargs.smart_turn is True
    assert args.vad_handler_kwargs.smart_turn_model_path is None
    assert args.vad_handler_kwargs.smart_turn_threshold == 0.5
    assert args.vad_handler_kwargs.smart_turn_max_wait_ms == 2000
    assert args.vad_handler_kwargs.smart_turn_incomplete_delay_ms == 600
    assert args.vad_handler_kwargs.speculative_reopen_ms == 800


def test_parse_arguments_accepts_smart_turn_options():
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "speech-to-speech",
            "--smart_turn",
            "--smart_turn_model_path",
            "/models/smart-turn.onnx",
            "--smart_turn_threshold",
            "0.7",
            "--smart_turn_max_wait_ms",
            "2500",
            "--smart_turn_incomplete_delay_ms",
            "700",
            "--smart_turn_cpu_count",
            "2",
        ]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    vad_args = args.vad_handler_kwargs
    assert vad_args.smart_turn is True
    assert vad_args.smart_turn_model_path == "/models/smart-turn.onnx"
    assert vad_args.smart_turn_threshold == 0.7
    assert vad_args.smart_turn_max_wait_ms == 2500
    assert vad_args.smart_turn_incomplete_delay_ms == 700
    assert vad_args.smart_turn_cpu_count == 2


def test_parse_arguments_can_disable_smart_turn():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--no_smart_turn"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert args.vad_handler_kwargs.smart_turn is False


def test_parse_arguments_rejects_removed_smart_turn_device_option():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--smart_turn_device", "cuda"]
        with pytest.raises(ValueError, match="--smart_turn_device"):
            parse_arguments()
    finally:
        sys.argv = original_argv


def test_parse_arguments_accepts_qwen3_tts_backend_override():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--qwen3_tts_backend", "torch"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert args.tts_backend.config["backend"] == "torch"


def test_parse_arguments_accepts_openai_tts_backend():
    args = parse_arguments(
        [
            "--tts",
            "openai",
            "--openai_tts_base_url",
            "http://localhost:8091/v1",
            "--openai_tts_voice",
            "vivian",
        ]
    )

    assert args.tts_backend.name == "openai"
    assert args.tts_backend.config["base_url"] == "http://localhost:8091/v1"
    assert args.tts_backend.config["voice"] == "vivian"


def test_parse_arguments_accepts_qwen3_tts_ggml_options():
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "speech-to-speech",
            "--qwen3_tts_ggml_quantization",
            "Q4_K_M",
            "--qwen3_tts_gguf_talker_path",
            "/models/talker.gguf",
            "--qwen3_tts_gguf_codec_path",
            "/models/codec.gguf",
            "--qwen3_tts_ref_cache_dir",
            "/voices/cache",
            "--qwen3_tts_ref_spk",
            "/voices/ref.spk",
            "--qwen3_tts_ref_rvq",
            "/voices/ref.rvq",
        ]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    qwen3_config = args.tts_backend.config
    assert qwen3_config["ggml_quantization"] == "Q4_K_M"
    assert qwen3_config["gguf_talker_path"] == "/models/talker.gguf"
    assert qwen3_config["gguf_codec_path"] == "/models/codec.gguf"
    assert qwen3_config["ref_cache_dir"] == "/voices/cache"
    assert qwen3_config["ref_spk"] == "/voices/ref.spk"
    assert qwen3_config["ref_rvq"] == "/voices/ref.rvq"


@pytest.mark.parametrize("command", ["serve", "talk", "local"])
def test_cli_exposes_command_family(command):
    assert parse_command([command]) == (command, [])


def test_local_pipeline_arguments_support_smart_turn():
    command, command_args = parse_command(["local", "--smart_turn"])
    args = parse_arguments(command_args, command=command)

    assert command == "local"
    assert args.vad_handler_kwargs.smart_turn is True


@pytest.mark.parametrize(
    ("mode", "command", "command_flag"),
    [("realtime", "serve", "--host"), ("local", "local", "--local_audio_input_device")],
)
def test_cli_maps_legacy_modes_to_commands_with_warning(mode, command, command_flag, capsys):
    assert parse_command(["--mode", mode, command_flag, "value"]) == (command, [command_flag, "value"])

    assert (
        capsys.readouterr().err == f"Warning: '--mode {mode}' is deprecated and will stop working soon; "
        f"use 'speech-to-speech {command}' instead.\n"
    )


def test_cli_accepts_equals_syntax_for_legacy_mode(capsys):
    assert parse_command(["--mode=realtime", "--port", "9876"]) == ("serve", ["--port", "9876"])
    assert "use 'speech-to-speech serve' instead" in capsys.readouterr().err


@pytest.mark.parametrize(("mode", "command"), [("realtime", "serve"), ("local", "local")])
def test_main_dispatches_legacy_modes_to_pipeline_commands(mode, command, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "--mode", mode, "--port", "9876"])
    monkeypatch.setattr(
        "speech_to_speech.s2s_pipeline.run_pipeline_command",
        lambda selected_command, command_args: calls.append((selected_command, command_args)),
    )

    main()

    assert calls == [(command, ["--port", "9876"])]
    assert f"use 'speech-to-speech {command}' instead" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["socket", "raw-websocket", "websocket"])
def test_cli_rejects_other_legacy_modes_with_migration_guidance(mode, capsys):
    with pytest.raises(SystemExit, match="2"):
        parse_command(["--mode", mode])

    error = capsys.readouterr().err
    assert "only 'realtime' and 'local' remain temporarily" in error
    assert "speech-to-speech serve" in error
    assert "speech-to-speech local" in error


def test_talk_accepts_one_full_url_connection_option():
    config = parse_talk_arguments(["--url", "wss://voice.example/v1/realtime"])

    assert config.url == "wss://voice.example/v1/realtime"


def test_talk_leaves_api_key_unset_for_sdk_environment_authentication():
    assert parse_talk_arguments([]).api_key is None
    assert parse_talk_arguments(["--api-key", "explicit-secret"]).api_key == "explicit-secret"


def test_talk_loads_opt_in_tool_module(monkeypatch):
    async def executor(_name, _arguments):
        return None

    tool = {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
    monkeypatch.setitem(
        sys.modules,
        "test_cli_voice_tools",
        SimpleNamespace(TOOLS=[tool], execute_tool=executor, CREATE_RESPONSE=False),
    )

    config = parse_talk_arguments(["--tool-module", "test_cli_voice_tools"])

    assert config.tools == [tool]
    assert config.tool_executor is executor
    assert config.tool_response_create is False


@pytest.mark.parametrize("flag", ["--host", "--port", "--base-url", "--websocket-base-url", "--stt"])
def test_talk_rejects_server_and_overlapping_connection_flags(flag):
    with pytest.raises(SystemExit):
        parse_talk_arguments([flag, "value"])


@pytest.mark.parametrize("command", ["serve", "local"])
def test_pipeline_commands_reject_talk_url(command):
    _, command_args = parse_command([command, "--url", "ws://127.0.0.1:8765/v1/realtime"])
    with pytest.raises(ValueError, match="--url"):
        parse_arguments(command_args, command=command)


def test_serve_rejects_local_audio_flags():
    with pytest.raises(ValueError, match="--local_audio_input_device"):
        parse_arguments(["--local_audio_input_device", "2"], command="serve")


def test_local_accepts_audio_flags_but_rejects_host():
    args = parse_arguments(["--port", "9876", "--local_audio_input_device", "2"], command="local")

    assert args.realtime_server_kwargs.host == "127.0.0.1"
    assert args.realtime_server_kwargs.port == 9876
    assert args.local_audio_kwargs.local_audio_input_device == 2
    with pytest.raises(ValueError, match="--host"):
        parse_arguments(["--host", "0.0.0.0"], command="local")


@pytest.mark.parametrize(
    "flag",
    ["--tool-module", "--local_audio_tool_module", "--local-audio-tool-module"],
)
def test_local_accepts_opt_in_tool_module(flag):
    args = parse_arguments([flag, "my_voice_tools"], command="local")

    assert args.local_audio_kwargs.local_audio_tool_module == "my_voice_tools"


def test_parse_arguments_transformers_backend():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--llm_backend", "transformers"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert isinstance(args, ParsedArguments)
    assert args.llm_backend.name == "transformers"
    assert args.llm_backend.spec.config_type is LanguageModelHandlerArguments
    assert args.llm_backend.config["model_name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert not hasattr(args, "responses_api_language_model_handler_kwargs")


def test_prepare_module_args_rejects_responses_api_for_stt_none():
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "speech-to-speech",
            "--stt",
            "none",
            "--responses_api_base_url",
            "http://127.0.0.1:8080/v1",
            "--model_name",
            "ggml-org/gemma-4-12B-it-GGUF",
        ]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    with pytest.raises(
        ValueError,
        match="--stt none requires an audio-input LLM backend.*chat-completions",
    ):
        prepare_module_args(args.module_kwargs, args.llm_backend)


def test_parse_arguments_stt_none_supports_chat_completions_audio_path():
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "speech-to-speech",
            "--stt",
            "none",
            "--llm_backend",
            "chat-completions",
            "--model_name",
            "gpt-audio-1.5",
            "--responses_api_audio_content_type",
            "audio_url",
            "--responses_api_audio_history_turns",
            "2",
        ]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    prepare_module_args(args.module_kwargs, args.llm_backend)

    assert args.module_kwargs.stt == "none"
    assert args.module_kwargs.llm_backend == "chat-completions"
    assert args.llm_backend.spec.config_type is ChatCompletionsLanguageModelHandlerArguments
    assert args.llm_backend.config["model_name"] == "gpt-audio-1.5"
    assert args.llm_backend.config["audio_content_type"] == "audio_url"
    assert args.llm_backend.config["audio_history_turns"] == 2


def test_parse_arguments_all_fields_populated():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    for f in fields(ParsedArguments):
        value = getattr(args, f.name)
        assert value is not None, f"Field {f.name!r} is None"
        assert isinstance(value, EXPECTED_FIELD_TYPES[f.name]), (
            f"Field {f.name!r}: expected {EXPECTED_FIELD_TYPES[f.name].__name__}, got {type(value).__name__}"
        )
