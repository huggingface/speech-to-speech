from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Qwen3TTSHandlerArguments:
    qwen3_tts_model_name: str = field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        metadata={
            "help": "The Qwen3-TTS model to use (HuggingFace Hub ID or local path). Default is 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'. On Apple Silicon, Qwen/* model IDs are auto-mapped to the corresponding mlx-community/* model when possible, defaulting to the 6bit MLX variant unless the model name already pins a specific suffix."
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
    qwen3_tts_backend: Literal["ggml", "torch"] = field(
        default="ggml",
        metadata={
            "help": "faster-qwen3-tts backend on non-macOS platforms. Options: 'ggml' or 'torch'. Default is 'ggml'. On Apple Silicon, mlx-audio is selected automatically and this option is ignored."
        },
    )
    qwen3_tts_ggml_quantization: str = field(
        default="BF16",
        metadata={
            "help": "GGUF quantization for the faster-qwen3-tts GGML backend. Supported values: 'BF16', 'Q8_0', 'Q4_K_M', 'F32'. Default is 'BF16'."
        },
    )
    qwen3_tts_gguf_talker_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional local qwentts.cpp talker GGUF path. Must be provided together with qwen3_tts_gguf_codec_path and requires the GGML backend."
        },
    )
    qwen3_tts_gguf_codec_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional local qwentts.cpp codec GGUF path. Must be provided together with qwen3_tts_gguf_talker_path and requires the GGML backend."
        },
    )
    qwen3_tts_ref_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional directory for automatically cached GGML voice references (.spk/.rvq). The faster-qwen3-tts default cache is used when unset."
        },
    )
    qwen3_tts_ref_audio: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to reference audio file for voice cloning. Leave unset when using a CustomVoice model."
        },
    )
    qwen3_tts_ref_spk: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional precomputed qwentts.cpp .spk speaker embedding for GGML voice cloning. Mutually exclusive with qwen3_tts_ref_audio."
        },
    )
    qwen3_tts_ref_rvq: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional precomputed qwentts.cpp .rvq acoustic codes for GGML ICL voice cloning. Requires qwen3_tts_ref_spk and qwen3_tts_ref_text."
        },
    )
    qwen3_tts_ref_text: str = field(
        default="I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes.",
        metadata={"help": "Transcription of the reference audio for voice cloning."},
    )
    qwen3_tts_speaker: Optional[str] = field(
        default="Aiden",
        metadata={
            "help": "Speaker name for CustomVoice models. Default is 'Aiden'. If not provided, the first supported speaker is used when available."
        },
    )
    qwen3_tts_instruct: Optional[str] = field(
        default=None,
        metadata={"help": "Instruction text for VoiceDesign models (optional, required for voice design)."},
    )
    qwen3_tts_xvec_only: bool = field(
        default=False,
        metadata={
            "help": "Use x-vector only voice cloning mode (recommended for cleaner starts and language switching). Default is False."
        },
    )
    qwen3_tts_parity_mode: bool = field(
        default=False,
        metadata={"help": "Disable CUDA-graph streaming path and use parity mode for stability. Default is False."},
    )
    qwen3_tts_non_streaming_mode: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Optional override for Qwen3-TTS text prefill behavior. Default is true, which pre-fills the full target text before decode on faster-qwen3-tts. Currently ignored on Apple Silicon because mlx-audio does not expose this yet."
        },
    )
    qwen3_tts_mlx_quantization: Optional[str] = field(
        default="6bit",
        metadata={
            "help": "Optional MLX quantization override on Apple Silicon. Supported values: 'bf16', '4bit', '6bit', '8bit'. Default is '6bit'."
        },
    )
    qwen3_tts_language: str = field(
        default="auto",
        metadata={"help": "Target language for synthesis. Default is 'auto'."},
    )
    qwen3_tts_streaming_chunk_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Codec steps per streaming chunk. If unset, the handler uses a backend-specific default: 8 on faster-qwen3-tts and 4 on mlx-audio."
        },
    )
    qwen3_tts_max_new_tokens: int = field(
        default=1536,
        metadata={
            "help": "Upper cap for Qwen3-TTS codec tokens. The handler estimates a per-utterance budget from the text and clamps it to this ceiling (~12 tokens per second of audio). Raise this above 1536 if you want to allow longer utterances."
        },
    )
    qwen3_tts_blocksize: int = field(
        default=512,
        metadata={
            "help": "Audio chunk size in samples for streaming output. Must match LocalAudioStreamer blocksize. Default is 512."
        },
    )
