from dataclasses import dataclass, field


@dataclass
class NemotronStreamingSTTHandlerArguments:
    nemotron_streaming_model_name: str = field(
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        metadata={
            "help": "The NeMo Nemotron streaming ASR checkpoint. Default is 'nvidia/nemotron-speech-streaming-en-0.6b'. Use 'nvidia/nemotron-3.5-asr-streaming-0.6b' for multilingual detection per utterance."
        },
    )
    nemotron_streaming_device: str = field(
        default="auto",
        metadata={"help": "The device to run on. 'auto' picks CUDA when available, otherwise CPU. Default is 'auto'."},
    )
    nemotron_streaming_language: str = field(
        default="en",
        metadata={
            "help": "Fallback language code when the checkpoint does not emit a language tag. The English-only checkpoint always reports this value. Nemotron 3.5 reports the detected tag per utterance. Default is 'en'."
        },
    )
