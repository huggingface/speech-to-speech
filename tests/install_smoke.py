from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _require_modules(modules: list[str]) -> None:
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(f"Missing expected install-time modules: {', '.join(missing)}")


def _run_installed_cli_help() -> None:
    env = {**os.environ, "OPENAI_API_KEY": ""}
    result = subprocess.run(
        ["speech-to-speech", "--help"],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    expected_flags = ("--mode", "--stt", "--llm", "--tts")
    missing_flags = [flag for flag in expected_flags if flag not in result.stdout]
    if missing_flags:
        raise RuntimeError(f"Installed CLI help is missing expected flags: {', '.join(missing_flags)}")


def _validate_package_defaults() -> None:
    import speech_to_speech
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
    from speech_to_speech.arguments_classes.open_api_language_model_arguments import (
        OpenApiLanguageModelHandlerArguments,
    )
    from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
    from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments

    module_args = ModuleArguments()
    open_api_args = OpenApiLanguageModelHandlerArguments()
    qwen3_args = Qwen3TTSHandlerArguments()
    vad_args = VADHandlerArguments()

    assert module_args.mode == "realtime"
    assert module_args.stt == "parakeet-tdt"
    assert module_args.llm == "open_api"
    assert module_args.tts == "qwen3"
    assert module_args.log_level == "info"
    assert module_args.enable_live_transcription is True
    assert open_api_args.open_api_model_name == "gpt-5.4-mini"
    assert open_api_args.open_api_stream is True
    assert qwen3_args.qwen3_tts_model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert qwen3_args.qwen3_tts_speaker == "Aiden"
    assert qwen3_args.qwen3_tts_language == "auto"
    assert qwen3_args.qwen3_tts_non_streaming_mode is True
    assert qwen3_args.qwen3_tts_ref_audio is None
    assert qwen3_args.qwen3_tts_mlx_quantization == "6bit"
    assert vad_args.thresh == 0.6

    package_root = Path(speech_to_speech.__file__).resolve().parent
    ref_audio = package_root / "TTS" / "ref_audio.wav"
    if not ref_audio.exists():
        raise RuntimeError(f"Packaged Qwen3-TTS reference audio is missing: {ref_audio}")


def _validate_empty_qwen_ref_audio_arg() -> None:
    from speech_to_speech.s2s_pipeline import parse_arguments

    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--qwen3_tts_ref_audio="]
        *_, qwen3_args = parse_arguments()
    finally:
        sys.argv = original_argv

    assert qwen3_args.qwen3_tts_ref_audio == ""
    assert qwen3_args.qwen3_tts_model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert qwen3_args.qwen3_tts_speaker == "Aiden"


def _validate_pipeline_startup_primitives() -> None:
    from speech_to_speech.s2s_pipeline import initialize_queues_and_events

    queues_and_events = initialize_queues_and_events()
    expected_keys = {
        "recv_audio_chunks_queue",
        "send_audio_chunks_queue",
        "spoken_prompt_queue",
        "stt_output_queue",
        "text_prompt_queue",
        "lm_response_queue",
        "lm_processed_queue",
        "text_output_queue",
    }
    missing_keys = expected_keys.difference(queues_and_events)
    if missing_keys:
        raise RuntimeError(f"Pipeline startup primitives are missing: {', '.join(sorted(missing_keys))}")


def main() -> None:
    required_modules = ["fastapi", "openai", "scipy", "sounddevice", "torch", "transformers", "uvicorn"]
    if sys.platform == "darwin":
        required_modules.extend(["mlx_audio", "misaki", "soundfile", "spacy"])
    else:
        required_modules.extend(["faster_qwen3_tts", "nano_parakeet"])

    _require_modules(required_modules)
    _run_installed_cli_help()
    _validate_package_defaults()
    _validate_empty_qwen_ref_audio_arg()
    _validate_pipeline_startup_primitives()
    print("speech-to-speech installed package smoke test passed")


if __name__ == "__main__":
    main()
