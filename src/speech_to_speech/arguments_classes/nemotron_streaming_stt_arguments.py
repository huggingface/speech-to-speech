from dataclasses import dataclass, field


@dataclass
class NemotronStreamingSTTHandlerArguments:
    nemotron_streaming_model_name: str = field(
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        metadata={
            "help": "The NeMo Nemotron streaming ASR checkpoint. Default is 'nvidia/nemotron-speech-streaming-en-0.6b'. For 'nvidia/nemotron-3.5-asr-streaming-0.6b', also set --nemotron_streaming_language to the expected spoken language."
        },
    )
    nemotron_streaming_device: str = field(
        default="auto",
        metadata={"help": "The device to run on. 'auto' picks CUDA when available, otherwise CPU. Default is 'auto'."},
    )
    nemotron_streaming_language: str = field(
        default="en",
        metadata={
            "help": "The fixed language code reported with every final transcription (for example, 'fr' for French). Does not detect language or condition the ASR model. Default is 'en'."
        },
    )
