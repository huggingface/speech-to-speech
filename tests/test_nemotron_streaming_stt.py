from speech_to_speech.s2s_pipeline import parse_arguments


def test_cli_nemotron_streaming_defaults() -> None:
    args = parse_arguments(["--stt", "nemotron-streaming"])

    assert args.stt_backend.name == "nemotron-streaming"
    assert args.stt_backend.config["model_name"] == "nvidia/nemotron-speech-streaming-en-0.6b"
    assert args.stt_backend.config["device"] == "auto"
    assert args.stt_backend.config["language"] == "en"
    assert args.stt_backend.spec.required_extra == "nemo"


def test_cli_nemotron_streaming_model_name_override() -> None:
    args = parse_arguments(
        [
            "--stt",
            "nemotron-streaming",
            "--nemotron_streaming_model_name",
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
        ]
    )

    assert args.stt_backend.name == "nemotron-streaming"
    assert args.stt_backend.config["model_name"] == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert args.stt_backend.spec.required_extra == "nemo"
