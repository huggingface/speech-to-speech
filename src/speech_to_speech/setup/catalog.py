from __future__ import annotations

from dataclasses import dataclass

from speech_to_speech.setup.system import GIB, ModelChoice


@dataclass(frozen=True)
class CuratedCatalog:
    choices: dict[str, tuple[ModelChoice, ...]]
    defaults: dict[str, ModelChoice]


PARAKEET = ModelChoice(
    "stt", "Parakeet TDT 0.6B (small, recommended)", "mlx-community/parakeet-tdt-0.6b-v3", "mlx", 2 * GIB
)
KOKORO = ModelChoice("tts", "Kokoro 82M (small, recommended)", "hexgrad/Kokoro-82M", "mlx", GIB)
QWEN_SMALL = ModelChoice(
    "llm",
    "Qwen3 4B MLX 4-bit (small)",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx",
    3 * GIB,
)
GEMMA = ModelChoice(
    "llm",
    "Gemma 4 12B Q4_0",
    "ggml-org/gemma-4-12B-it-GGUF",
    "llama.cpp",
    8 * GIB,
    variant="Q4_0",
    allow_patterns=("*Q4_0*.gguf",),
)


def curated_catalog(memory_bytes: int) -> CuratedCatalog:
    llms = (GEMMA, QWEN_SMALL) if memory_bytes >= 24 * GIB else (QWEN_SMALL, GEMMA)
    return CuratedCatalog(
        choices={"stt": (PARAKEET,), "llm": llms, "tts": (KOKORO,)},
        defaults={"stt": PARAKEET, "llm": llms[0], "tts": KOKORO},
    )
