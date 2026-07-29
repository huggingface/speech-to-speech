from __future__ import annotations

import importlib
import importlib.metadata as metadata
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
    expected_flags = ("--mode", "--stt", "--llm_backend", "--tts")
    missing_flags = [flag for flag in expected_flags if flag not in result.stdout]
    if missing_flags:
        raise RuntimeError(f"Installed CLI help is missing expected flags: {', '.join(missing_flags)}")


def _validate_package_defaults() -> None:
    import speech_to_speech
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
    from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
    from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
        ResponsesApiLanguageModelHandlerArguments,
    )
    from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments

    module_args = ModuleArguments()
    responses_api_args = ResponsesApiLanguageModelHandlerArguments()
    qwen3_args = Qwen3TTSHandlerArguments()
    vad_args = VADHandlerArguments()

    assert module_args.mode == "realtime"
    assert module_args.stt == "parakeet-tdt"
    assert module_args.llm_backend == "responses-api"
    assert module_args.tts == "qwen3"
    assert module_args.log_level == "info"
    assert module_args.enable_live_transcription is True
    assert module_args.live_transcription_update_interval == 0.5
    assert responses_api_args.model_name == "gpt-5.4-mini"
    assert responses_api_args.responses_api_stream is True
    assert qwen3_args.qwen3_tts_model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert qwen3_args.qwen3_tts_speaker == "Aiden"
    assert qwen3_args.qwen3_tts_language == "auto"
    assert qwen3_args.qwen3_tts_backend == "ggml"
    assert qwen3_args.qwen3_tts_non_streaming_mode is True
    assert qwen3_args.qwen3_tts_ref_audio is None
    assert qwen3_args.qwen3_tts_mlx_quantization == "6bit"
    assert vad_args.thresh == 0.6
    assert vad_args.min_silence_ms == 64
    assert vad_args.min_speech_ms == 384
    assert vad_args.min_speech_continuation_ms == 192
    assert vad_args.realtime_processing_pause == 0.5

    package_root = Path(speech_to_speech.__file__).resolve().parent
    ref_audio = package_root / "TTS" / "ref_audio.wav"
    if not ref_audio.exists():
        raise RuntimeError(f"Packaged Qwen3-TTS reference audio is missing: {ref_audio}")


def _validate_empty_qwen_ref_audio_arg() -> None:
    from speech_to_speech.s2s_pipeline import parse_arguments

    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech", "--qwen3_tts_ref_audio="]
        qwen3_args = parse_arguments().qwen3_tts_handler_kwargs
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


def _validate_default_handler_imports() -> None:
    default_handler_modules = [
        "speech_to_speech.LLM.responses_api_language_model",
        "speech_to_speech.STT.parakeet_tdt_handler",
        "speech_to_speech.TTS.qwen3_tts_handler",
        "speech_to_speech.VAD.vad_handler",
    ]
    for module_name in default_handler_modules:
        importlib.import_module(module_name)


def _validate_realtime_websocket_support() -> None:
    importlib.import_module("uvicorn.protocols.websockets.websockets_impl")


def _validate_darwin_dependency_pins() -> None:
    expected_versions = {
        "miniaudio": "1.61",
        "mlx": "0.31.1",
        "mlx-audio": "0.4.2",
        "mlx-lm": "0.31.1",
        "mlx-metal": "0.31.1",
        "sounddevice": "0.5.3",
        "transformers": "5.6.2",
    }
    mismatches = []
    for package_name, expected_version in expected_versions.items():
        actual_version = metadata.version(package_name)
        if actual_version != expected_version:
            mismatches.append(f"{package_name}=={actual_version} (expected {expected_version})")

    numpy_version = metadata.version("numpy")
    numpy_version_parts = tuple(int(part) for part in numpy_version.split(".")[:3])
    if numpy_version_parts >= (2, 4, 4):
        mismatches.append(f"numpy=={numpy_version} (expected <2.4.4 on macOS)")

    if mismatches:
        raise RuntimeError("Unexpected macOS dependency versions: " + ", ".join(mismatches))


def main() -> None:
    required_modules = [
        "fastapi",
        "lingua",
        "openai",
        "PIL",
        "scipy",
        "sounddevice",
        "torch",
        "torchaudio",
        "transformers",
        "uvicorn",
        "websockets",
    ]
    if sys.platform == "darwin":
        required_modules.extend(["miniaudio", "mlx", "mlx_audio", "mlx_lm", "misaki", "soundfile", "spacy"])
    else:
        required_modules.extend(["faster_qwen3_tts", "nano_parakeet"])

    _require_modules(required_modules)
    if sys.platform == "darwin":
        _validate_darwin_dependency_pins()
    _run_installed_cli_help()
    _validate_package_defaults()
    _validate_empty_qwen_ref_audio_arg()
    _validate_pipeline_startup_primitives()
    _validate_default_handler_imports()
    _validate_realtime_websocket_support()
    print("speech-to-speech installed package smoke test passed")


if __name__ == "__main__":
    main()
