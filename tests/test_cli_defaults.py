import sys
from dataclasses import fields

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
from speech_to_speech.s2s_pipeline import ParsedArguments, parse_arguments


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
    assert responses_api_args.model_name == "gpt-5.4-mini"
    assert responses_api_args.chat_size == 30
    assert responses_api_args.responses_api_stream is True
    assert qwen3_args.qwen3_tts_model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert qwen3_args.qwen3_tts_speaker == "Aiden"
    assert qwen3_args.qwen3_tts_language == "auto"
    assert qwen3_args.qwen3_tts_backend == "ggml"
    assert qwen3_args.qwen3_tts_non_streaming_mode is True
    assert qwen3_args.qwen3_tts_ref_audio is None
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


def test_parse_arguments_accepts_qwen3_tts_backend_override():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--qwen3_tts_backend", "torch"]
        args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert args.qwen3_tts_handler_kwargs.qwen3_tts_backend == "torch"


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
