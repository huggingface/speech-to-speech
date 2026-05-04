from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.open_api_language_model_arguments import OpenApiLanguageModelHandlerArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments


def test_release_defaults_match_openapi_parakeet_qwen3_realtime_profile():
    module_args = ModuleArguments()
    vad_args = VADHandlerArguments()
    open_api_args = OpenApiLanguageModelHandlerArguments()
    qwen3_args = Qwen3TTSHandlerArguments()

    assert module_args.mode == "realtime"
    assert module_args.stt == "parakeet-tdt"
    assert module_args.llm_backend == "openai-api"
    assert module_args.tts == "qwen3"
    assert module_args.log_level == "info"
    assert module_args.enable_live_transcription is True

    assert vad_args.thresh == 0.6
    assert open_api_args.model_name == "gpt-5.4-mini"
    assert open_api_args.chat_size == 30
    assert open_api_args.open_api_stream is True
    assert qwen3_args.qwen3_tts_model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert qwen3_args.qwen3_tts_speaker == "Aiden"
    assert qwen3_args.qwen3_tts_language == "auto"
    assert qwen3_args.qwen3_tts_non_streaming_mode is True
    assert qwen3_args.qwen3_tts_ref_audio is None
    assert qwen3_args.qwen3_tts_mlx_quantization == "6bit"
