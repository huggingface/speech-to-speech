from dataclasses import dataclass, field


@dataclass
class Qwen3TTSHandlerArguments:
    qwen3_tts_model_name: str = field(
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        metadata={
            "help": "The Qwen3-TTS model to use (HuggingFace Hub ID or local path)."
        },
    )
    qwen3_tts_device: str = field(
        default="cuda",
        metadata={
            "help": "The device to run Qwen3-TTS on. Options: 'cuda', 'cpu'. Default is 'cuda'."
        },
    )
    qwen3_tts_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type for inference. Options: 'auto', 'float16', 'bfloat16', 'float32'. Default is 'auto'."
        },
    )
    qwen3_tts_attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Attention implementation. Options: 'eager', 'flash_attention_2', 'sdpa'. Use 'eager' on Jetson. Default is 'eager'."
        },
    )
    qwen3_tts_ref_audio: str = field(
        default="TTS/ref_audio.wav",
        metadata={
            "help": "Path to reference audio file for voice cloning."
        },
    )
    qwen3_tts_ref_text: str = field(
        default="I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes.",
        metadata={
            "help": "Transcription of the reference audio for voice cloning."
        },
    )
    qwen3_tts_language: str = field(
        default="English",
        metadata={
            "help": "Target language for synthesis. Default is 'English'."
        },
    )
    qwen3_tts_use_cuda_graphs: bool = field(
        default=True,
        metadata={
            "help": "Use CUDA graphs for real-time inference. Requires NVIDIA GPU. Default is True."
        },
    )
    qwen3_tts_streaming_chunk_size: int = field(
        default=8,
        metadata={
            "help": "Codec steps per streaming chunk (8 = ~667ms of audio). Default is 8."
        },
    )
    qwen3_tts_max_new_tokens: int = field(
        default=200,
        metadata={
            "help": "Maximum codec tokens to generate (~12 tokens per second of audio). Default is 200."
        },
    )
