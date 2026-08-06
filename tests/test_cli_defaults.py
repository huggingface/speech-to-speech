import sys
from dataclasses import fields

import pytest

from speech_to_speech.arguments_classes.chat_completions_language_model_arguments import (
    ChatCompletionsLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.chat_tts_arguments import ChatTTSHandlerArguments
from speech_to_speech.arguments_classes.facebookmms_tts_arguments import FacebookMMSTTSHandlerArguments
from speech_to_speech.arguments_classes.faster_whisper_stt_arguments import FasterWhisperSTTHandlerArguments
from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
from speech_to_speech.arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from speech_to_speech.arguments_classes.mlx_audio_whisper_arguments import MLXAudioWhisperSTTHandlerArguments
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.paraformer_stt_arguments import ParaformerSTTHandlerArguments
from speech_to_speech.arguments_classes.parakeet_tdt_arguments import ParakeetTDTSTTHandlerArguments
from speech_to_speech.arguments_classes.pocket_tts_arguments import PocketTTSHandlerArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from speech_to_speech.arguments_classes.socket_sender_arguments import SocketSenderArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.arguments_classes.websocket_streamer_arguments import WebSocketStreamerArguments
from speech_to_speech.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from speech_to_speech.s2s_pipeline import ParsedArguments, parse_arguments, prepare_module_args


def test_release_defaults_match_responses_api_parakeet_qwen3_realtime_profile():
    module_args = ModuleArguments()
    vad_args = VADHandlerArguments()
    responses_api_args = ResponsesApiLanguageModelHandlerArguments()
    qwen3_args = Qwen3TTSHandlerArguments()

    assert module_args.mode == "realtime"
    assert module_args.stt == "parakeet-tdt"
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


# -- ParsedArguments dataclass tests ------------------------------------------

EXPECTED_FIELD_TYPES = {
    "module_kwargs": ModuleArguments,
    "socket_receiver_kwargs": SocketReceiverArguments,
    "socket_sender_kwargs": SocketSenderArguments,
    "websocket_streamer_kwargs": WebSocketStreamerArguments,
    "vad_handler_kwargs": VADHandlerArguments,
    "whisper_stt_handler_kwargs": WhisperSTTHandlerArguments,
    "paraformer_stt_handler_kwargs": ParaformerSTTHandlerArguments,
    "faster_whisper_stt_handler_kwargs": FasterWhisperSTTHandlerArguments,
    "mlx_audio_whisper_stt_handler_kwargs": MLXAudioWhisperSTTHandlerArguments,
    "parakeet_tdt_stt_handler_kwargs": ParakeetTDTSTTHandlerArguments,
    "language_model_handler_kwargs": LanguageModelHandlerArguments,
    "responses_api_language_model_handler_kwargs": ResponsesApiLanguageModelHandlerArguments,
    "chat_tts_handler_kwargs": ChatTTSHandlerArguments,
    "facebook_mms_tts_handler_kwargs": FacebookMMSTTSHandlerArguments,
    "pocket_tts_handler_kwargs": PocketTTSHandlerArguments,
    "kokoro_tts_handler_kwargs": KokoroTTSHandlerArguments,
    "qwen3_tts_handler_kwargs": Qwen3TTSHandlerArguments,
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
    assert isinstance(args.responses_api_language_model_handler_kwargs, ResponsesApiLanguageModelHandlerArguments)
    assert isinstance(args.language_model_handler_kwargs, LanguageModelHandlerArguments)
    assert args.responses_api_language_model_handler_kwargs.model_name == "gpt-5.4-mini"
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

    assert args.qwen3_tts_handler_kwargs.qwen3_tts_backend == "torch"


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

    qwen3_args = args.qwen3_tts_handler_kwargs
    assert qwen3_args.qwen3_tts_ggml_quantization == "Q4_K_M"
    assert qwen3_args.qwen3_tts_gguf_talker_path == "/models/talker.gguf"
    assert qwen3_args.qwen3_tts_gguf_codec_path == "/models/codec.gguf"
    assert qwen3_args.qwen3_tts_ref_cache_dir == "/voices/cache"
    assert qwen3_args.qwen3_tts_ref_spk == "/voices/ref.spk"
    assert qwen3_args.qwen3_tts_ref_rvq == "/voices/ref.rvq"


def test_parse_arguments_accepts_raw_websocket_mode():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--mode", "raw-websocket", "--no_smart_turn"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert args.module_kwargs.mode == "raw-websocket"
    assert args.vad_handler_kwargs.smart_turn is False


def test_parse_arguments_rejects_smart_turn_outside_realtime_mode():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--mode", "local", "--smart_turn"]
        with pytest.raises(ValueError, match="--smart_turn is only supported with --mode realtime"):
            parse_arguments()
    finally:
        sys.argv = original_argv


def test_parse_arguments_rejects_removed_websocket_mode():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--mode", "websocket"]
        with pytest.raises(SystemExit):
            parse_arguments()
    finally:
        sys.argv = original_argv


def test_parse_arguments_transformers_backend():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--llm_backend", "transformers"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert isinstance(args, ParsedArguments)
    assert isinstance(args.language_model_handler_kwargs, LanguageModelHandlerArguments)
    assert isinstance(args.responses_api_language_model_handler_kwargs, ResponsesApiLanguageModelHandlerArguments)
    assert args.language_model_handler_kwargs.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    # unused slot gets a default instance
    assert args.responses_api_language_model_handler_kwargs.model_name == "gpt-5.4-mini"


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
        match="--stt none requires --llm_backend chat-completions",
    ):
        prepare_module_args(args.module_kwargs)


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

    prepare_module_args(args.module_kwargs)

    assert args.module_kwargs.stt == "none"
    assert args.module_kwargs.llm_backend == "chat-completions"
    assert isinstance(
        args.responses_api_language_model_handler_kwargs,
        ChatCompletionsLanguageModelHandlerArguments,
    )
    assert args.responses_api_language_model_handler_kwargs.model_name == "gpt-audio-1.5"
    assert args.responses_api_language_model_handler_kwargs.responses_api_audio_content_type == "audio_url"
    assert args.responses_api_language_model_handler_kwargs.responses_api_audio_history_turns == 2


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
