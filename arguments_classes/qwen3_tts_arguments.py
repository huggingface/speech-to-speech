from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Qwen3TTSHandlerArguments:
    qwen3_tts_model_name: str = field(
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        metadata={
            "help": "The Qwen3-TTS model to use (HuggingFace Hub ID or local path). On Apple Silicon, Qwen/* model IDs are auto-mapped to the corresponding mlx-community/*-bf16 model when possible."
        },
    )
    qwen3_tts_device: str = field(
        default="cuda",
        metadata={
            "help": "Preferred device for Qwen3-TTS. Options: 'cuda', 'cpu', 'mps', 'auto'. Default is 'cuda'. On Apple Silicon the mlx-audio backend is selected automatically."
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
    qwen3_tts_speaker: str = field(
        default=None,
        metadata={
            "help": "Speaker name for CustomVoice models (optional). If not provided, the first supported speaker is used when available."
        },
    )
    qwen3_tts_instruct: str = field(
        default=None,
        metadata={
            "help": "Instruction text for VoiceDesign models (optional, required for voice design)."
        },
    )
    qwen3_tts_xvec_only: bool = field(
        default=False,
        metadata={
            "help": "Use x-vector only voice cloning mode (recommended for cleaner starts and language switching). Default is False."
        },
    )
    qwen3_tts_parity_mode: bool = field(
        default=False,
        metadata={
            "help": "Disable CUDA-graph streaming path and use parity mode for stability. Default is False."
        },
    )
    qwen3_tts_mlx_quantization: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional MLX quantization override on Apple Silicon. Use '6bit' to auto-map MLX Qwen3-TTS models to their 6-bit variant for faster inference, or leave unset for the default bf16 models."
        },
    )
    qwen3_tts_language: str = field(
        default="English",
        metadata={
            "help": "Target language for synthesis. Default is 'English'."
        },
    )
    qwen3_tts_streaming_chunk_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Codec steps per streaming chunk. If unset, the handler uses a backend-specific default: 8 on faster-qwen3-tts and 2 on mlx-audio to reduce audible pauses on macOS."
        },
    )
    qwen3_tts_max_new_tokens: int = field(
        default=360,
        metadata={
            "help": "Maximum codec tokens to generate (~12 tokens per second of audio, ~30s max). Default is 360."
        },
    )
    qwen3_tts_blocksize: int = field(
        default=512,
        metadata={
            "help": "Audio chunk size in samples for streaming output. Must match LocalAudioStreamer blocksize. Default is 512."
        },
    )
