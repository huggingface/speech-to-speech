import json
import sys
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
from speech_to_speech.cli import main, parse_command, parse_talk_arguments
from speech_to_speech.pipeline.transcript_logging import set_log_transcripts, transcript_for_log
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
    assert module_args.detect_llm_output_language is False
    assert module_args.log_level == "info"
    assert module_args.enable_live_transcription is True
    assert module_args.live_transcription_update_interval == 0.5

    assert vad_args.vad == "silero"
    assert vad_args.vad_firered_model_dir is None
    assert vad_args.vad_firered_use_gpu is False
    assert vad_args.thresh == 0.6
    assert vad_args.min_silence_ms == 64
    assert vad_args.min_speech_ms == 384
    assert vad_args.min_speech_continuation_ms == 192
    assert vad_args.realtime_processing_pause == 0.5
    assert vad_args.smart_turn is True
    assert responses_api_args.model_name == "gpt-5.6-terra"
    assert responses_api_args.chat_size == 30
    assert responses_api_args.responses_api_stream is True
    assert responses_api_args.responses_api_reasoning_effort == "none"
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


def test_parse_arguments_enables_llm_output_language_detection():
    args = parse_arguments(["--detect_llm_output_language"])

    assert args.module_kwargs.detect_llm_output_language is True


def test_server_defaults_to_loopback():
    assert RealtimeServerArguments().host == "127.0.0.1"


def test_parse_talk_arguments_keeps_retry_timeout_field_name():
    config = parse_talk_arguments(["--connection-retry-timeout", "12.5"])

    assert config.connection_retry_timeout_s == 12.5


def test_vad_firered_flag_is_accepted():
    args = parse_arguments(["--vad", "firered", "--vad_firered_model_dir", "/tmp/firered-stream-vad"])

    assert args.vad_handler_kwargs.vad == "firered"
    assert args.vad_handler_kwargs.vad_firered_model_dir == "/tmp/firered-stream-vad"


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
    assert args.llm_backend.config["model_name"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"


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


def test_mac_diarization_shortcut():
    args = parse_arguments(["--mac-optimal-settings", "--diarization"], command="local")

    assert args.module_kwargs.mac_optimal_settings is True
    assert args.module_kwargs.diarization_model_name == "nvidia/Nemotron-3-Diarization"
    assert args.module_kwargs.diarization_revision is None
    assert args.module_kwargs.diarization_streaming_mode == "low_latency"
    assert args.module_kwargs.diarization_device == "mps"
    assert args.module_kwargs.stt == "parakeet-tdt"
    assert args.module_kwargs.llm_backend == "mlx-lm"


def test_diarization_shortcut_preserves_custom_model():
    args = parse_arguments(["--diarization", "--diarization_model_name", "custom/model"])

    assert args.module_kwargs.diarization_model_name == "custom/model"
    assert args.module_kwargs.diarization_revision is None


def test_diarization_defaults_to_automatic_device_selection():
    args = parse_arguments(["--diarization"])

    assert args.module_kwargs.diarization_device == "auto"


def test_diarization_shortcut_preserves_explicit_settings():
    args = parse_arguments(
        [
            "--mac-optimal-settings",
            "--diarization",
            "--diarization_revision",
            "custom-ref",
            "--diarization_device",
            "cpu",
        ]
    )

    assert args.module_kwargs.diarization_revision == "custom-ref"
    assert args.module_kwargs.diarization_device == "cpu"


def test_diarization_is_opt_in():
    assert parse_arguments([]).module_kwargs.diarization_model_name is None


def test_parse_arguments_default_backend_returns_openai_api():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert isinstance(args, ParsedArguments)
    assert isinstance(args.module_kwargs, ModuleArguments)
    assert args.realtime_server_kwargs.host == "127.0.0.1"
    assert args.local_audio_kwargs.local_audio_playback_buffer_ms is None
    assert args.stt_backend.name == "parakeet-tdt"
    assert args.tts_backend.name == "qwen3"
    assert args.llm_backend.name == "responses-api"
    assert args.llm_backend.spec.config_type is ResponsesApiLanguageModelHandlerArguments
    assert args.llm_backend.config["model_name"] == "gpt-5.6-terra"
    assert args.llm_backend.config["reasoning_effort"] == "none"
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
    assert args.tts_backend.config["stream"] is False


def test_parse_arguments_accepts_vllm_tts_stream_extension():
    args = parse_arguments(["--tts", "openai", "--openai_tts_stream", "true"])

    assert args.tts_backend.config["stream"] is True


def test_parse_arguments_accepts_openai_stt_backend():
    args = parse_arguments(
        [
            "--stt",
            "openai",
            "--openai_stt_base_url",
            "http://localhost:8000/v1",
            "--openai_stt_model",
            "Qwen/Qwen3-ASR-1.7B",
        ]
    )

    assert args.stt_backend.name == "openai"
    assert args.stt_backend.config["base_url"] == "http://localhost:8000/v1"
    assert args.stt_backend.config["model"] == "Qwen/Qwen3-ASR-1.7B"


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


def test_talk_accepts_custom_playback_buffer():
    config = parse_talk_arguments(["--playback-buffer-ms", "240"])

    assert config.playback_buffer_ms == 240


def test_packaged_audio_clients_have_no_general_playback_buffer_default():
    assert parse_talk_arguments([]).playback_buffer_ms == 0
    assert LocalAudioArguments().local_audio_playback_buffer_ms is None


@pytest.mark.parametrize("flag", ["--log-transcripts", "--log_transcripts"])
def test_talk_accepts_transcript_logging_opt_in(flag):
    assert parse_talk_arguments([]).log_transcripts is False
    assert parse_talk_arguments([flag]).log_transcripts is True


def test_main_wires_talk_transcript_logging_before_client_start(monkeypatch):
    events = []
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "talk", "--log-transcripts"])

    def warning():
        assert transcript_for_log("probe") == "probe"
        events.append("warning")

    def run_client(config):
        assert transcript_for_log("probe") == "probe"
        events.append(("client", config.log_transcripts))

    monkeypatch.setattr("speech_to_speech.cli.warn_if_log_transcripts_enabled", warning)
    monkeypatch.setattr("speech_to_speech.cli.run_realtime_audio_client", run_client)

    try:
        main()
    finally:
        set_log_transcripts(False)

    assert events == ["warning", ("client", True)]


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
    args = parse_arguments(
        ["--port", "9876", "--local_audio_input_device", "2", "--playback-buffer-ms", "240"],
        command="local",
    )

    assert args.realtime_server_kwargs.host == "127.0.0.1"
    assert args.realtime_server_kwargs.port == 9876
    assert args.local_audio_kwargs.local_audio_input_device == 2
    assert args.local_audio_kwargs.local_audio_playback_buffer_ms == 240
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


def test_additive_config_api_preserves_flat_json_contract(tmp_path):
    from speech_to_speech.config import load_config

    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "stt": "openai",
                "llm_backend": "chat-completions",
                "tts": "openai",
                "num_pipelines": 2,
                "openai_stt_api_key": "",
                "responses_api_stream": False,
                "inactive_extra_key": "unchanged",
                "qwen3_tts_speaker": "unused",
            }
        )
    )
    args = parse_arguments([str(path)])
    assert isinstance(args, ParsedArguments)
    assert args.module_kwargs.num_pipelines == 2
    assert args.stt_backend.config["api_key"] == ""
    assert args.llm_backend.config["stream"] is False
    with pytest.raises(ValueError, match="existing flat JSON interface"):
        load_config(path)


@pytest.fixture
def configured_cli(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("""schema_version: 1
blocks:
  vad: {kind: vad, backend: silero}
  stt: {kind: stt, backend: openai}
  llm: {kind: llm, backend: chat-completions}
  tts: {kind: tts, backend: openai}
pipelines:
  primary:
    stages: {vad: vad, stt: stt, llm: llm, tts: tts}
  secondary:
    stages: {vad: vad, stt: stt, llm: llm, tts: tts}
""")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "speech_to_speech.configured_runtime",
        SimpleNamespace(run_configured_command=lambda command, config: calls.append((command, config))),
    )
    return path, calls


@pytest.mark.parametrize("flag", ["-f", "--file"])
def test_file_cli_dispatches_selected_pipeline(flag, configured_cli, monkeypatch):
    path, calls = configured_cli
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "serve", flag, str(path), "--name", "secondary"])
    main()
    assert calls[0][0] == "serve"
    assert list(calls[0][1].pipelines) == ["secondary"]


def test_file_talk_resolves_only_client(configured_cli, monkeypatch):
    path, calls = configured_cli
    path.write_text(
        path.read_text().replace(
            "schema_version: 1",
            "schema_version: 1\nruntime: {server: {port: {env: MISSING_PORT}}, client: {model: client-model}}",
        )
    )
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "talk", "--file", str(path)])
    main()
    assert calls[0][0] == "talk"
    assert calls[0][1].pipelines == {}
    assert calls[0][1].client["model"] == "client-model"


@pytest.mark.parametrize("command", ["serve", "local"])
def test_file_cli_rejects_multiple_definitions_before_runtime(command, configured_cli, monkeypatch):
    path, calls = configured_cli
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", command, "-f", str(path)])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert calls == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--name", "primary", "--name", "primary"],
        ["--name", ""],
        ["--name", "unknown"],
        ["--name", "primary,secondary"],
        ["--port", "9876"],
        ["--api-key", "SECRET_MARKER"],
        ["--file", "SECRET_MARKER.yaml"],
    ],
)
def test_file_cli_rejects_invalid_and_mixed_flags(extra, configured_cli, monkeypatch, capsys):
    path, calls = configured_cli
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "serve", "-f", str(path), *extra])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert calls == []
    assert "SECRET_MARKER" not in capsys.readouterr().err


def test_file_talk_rejects_name(configured_cli, monkeypatch):
    path, calls = configured_cli
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "talk", "-f", str(path), "--name", "primary"])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert calls == []


@pytest.mark.parametrize("command", ["serve", "talk", "local"])
def test_cli_help_does_not_read_file_or_run_runtime(command, configured_cli, monkeypatch):
    _, calls = configured_cli
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", command, "--file", "missing.yaml", "--help"])
    with pytest.raises(SystemExit, match="0"):
        main()
    assert calls == []


def test_name_requires_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "serve", "--name", "primary"])
    with pytest.raises(SystemExit, match="2"):
        main()


def test_file_cli_hides_runtime_error(configured_cli, monkeypatch, capsys):
    path, _ = configured_cli

    def fail(*_args):
        raise RuntimeError("provider SECRET_MARKER https://user:password@host")

    monkeypatch.setitem(
        sys.modules, "speech_to_speech.configured_runtime", SimpleNamespace(run_configured_command=fail)
    )
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "serve", "-f", str(path), "--name", "primary"])
    with pytest.raises(SystemExit, match="1"):
        main()
    error = capsys.readouterr().err
    assert "SECRET_MARKER" not in error
    assert "password" not in error


@pytest.mark.parametrize(
    "arguments",
    [
        ["serve", "-f", "missing.yaml", "--help"],
        ["local", "-f", "missing.yaml", "--help"],
        ["talk", "-f", "missing.yaml", "--help"],
        ["serve", "--name", "primary"],
    ],
)
def test_configured_cli_rejects_or_helps_before_heavy_imports(arguments):
    import subprocess

    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.endswith("s2s_pipeline") or fullname.endswith("configured_runtime"):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from speech_to_speech.cli import main
sys.argv = ["speech-to-speech", *sys.argv[1:]]
try:
    main()
except SystemExit as exc:
    assert exc.code == (0 if "--help" in sys.argv else 2)
else:
    raise AssertionError("expected help or rejection")
"""
    result = subprocess.run([sys.executable, "-c", script, *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_file_cli_accepts_attached_short_file_flag(configured_cli, monkeypatch):
    path, calls = configured_cli
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "local", f"-f{path}", "--name=primary"])
    main()
    assert calls[0][0] == "local"


def test_file_cli_malformed_flags_hide_input_values(configured_cli, monkeypatch, capsys):
    path, _ = configured_cli
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", "serve", "-f", str(path), "--name", "--SECRET_MARKER"])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "SECRET_MARKER" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments, flags",
    [
        (["serve", "--help"], ["--host", "--port", "--stt"]),
        (["local", "--help"], ["--port", "--local_audio_input_device"]),
        (["--mode", "local", "--help"], ["--local_audio_input_device"]),
    ],
)
def test_legacy_help_keeps_command_settings(arguments, flags, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["speech-to-speech", *arguments])
    with pytest.raises(SystemExit, match="0"):
        main()
    output = capsys.readouterr().out
    for flag in flags:
        assert flag in output
