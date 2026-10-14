from dataclasses import dataclass, field


@dataclass
class ChatTTSHandlerArguments:
    chat_tts_stream: bool = field(
        default=True,
        metadata={"help": "The tts mode is stream Default is 'stream'."},
    )
    chat_tts_device: str = field(
        default="auto",
        metadata={
            "help": "The device to be used for speech synthesis: 'auto' (ChatTTS picks CUDA/NPU, else CPU), "
            "'cuda', 'npu', 'mps', or 'cpu'. Default is 'auto'."
        },
    )
    chat_tts_chunk_size: int = field(
        default=512,
        metadata={
            "help": "Sets the size of the audio data chunk processed per cycle, balancing playback latency and CPU load.. Default is 512。."
        },
    )
