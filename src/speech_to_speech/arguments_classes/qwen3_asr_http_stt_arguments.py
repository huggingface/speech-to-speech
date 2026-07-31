from dataclasses import dataclass, field


@dataclass
class Qwen3ASRHTTPSTTHandlerArguments:
    """
    Arguments for the Qwen3-ASR HTTP Speech-to-Text handler.

    Talks to a `qwen-asr-serve` process (its own virtualenv, pinned to transformers==4.57.6)
    over HTTP instead of loading the model in-process, working around the transformers==5.6.2
    pin this project uses on macOS. See src/speech_to_speech/STT/README.md for the full
    incompatibility writeup and how to start the server.
    """

    qwen3_asr_http_base_url: str = field(
        default="http://127.0.0.1:8000",
        metadata={
            "help": "Base URL of a running `qwen-asr-serve` server. Default is 'http://127.0.0.1:8000'."
        },
    )
    qwen3_asr_http_timeout_s: float = field(
        default=30.0,
        metadata={"help": "HTTP request timeout in seconds for each transcription call. Default is 30.0."},
    )
