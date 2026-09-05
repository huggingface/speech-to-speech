from __future__ import annotations

from speech_to_speech.setup.system import GIB, ModelChoice

PARAKEET = ModelChoice("Parakeet TDT 0.6B (small, recommended)", "mlx-community/parakeet-tdt-0.6b-v3", "mlx", 2 * GIB)
KOKORO = ModelChoice("Kokoro 82M (small, recommended)", "mlx-community/Kokoro-82M-bf16", "mlx", GIB)
QWEN_SMALL = ModelChoice(
    "Qwen3 4B MLX 4-bit (small)",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx",
    3 * GIB,
)
GEMMA = ModelChoice(
    "Gemma 4 12B Q4_0",
    "ggml-org/gemma-4-12B-it-GGUF",
    "llama.cpp",
    8 * GIB,
    variant="Q4_0",
    allow_patterns=("*Q4_0*.gguf",),
)


def curated_catalog(memory_bytes: int) -> dict[str, tuple[ModelChoice, ...]]:
    llms = (GEMMA, QWEN_SMALL) if memory_bytes >= 24 * GIB else (QWEN_SMALL, GEMMA)
    return {"stt": (PARAKEET,), "llm": llms, "tts": (KOKORO,)}
