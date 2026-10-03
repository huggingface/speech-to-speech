"""Shared text language detection for transcription metadata and assistant speech."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

try:
    from lingua import Language, LanguageDetectorBuilder

    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False

logger = logging.getLogger(__name__)
MIN_ASSISTANT_CONFIDENCE_GAP = 0.1
_LINGUA_CODE_MAP = {"no": "nb"}
_WARMUP_TEXTS = (
    "This sentence warms the language detector before the first response.",
    "Bonjour, je peux vous aider à trouver la bonne réponse.",
    "Guten Tag, ich helfe Ihnen gerne bei dieser Frage.",
    "Hola, puedo ayudarte a encontrar la mejor respuesta.",
    "Posso aiutarti a trovare le informazioni che cerchi.",
    "Posso ajudar você a encontrar a resposta correta.",
    "Это предложение прогревает определение языка перед первым ответом.",
    "这句话会在第一次回复之前预热语言检测器。",
    "この文章は最初の応答の前に言語検出器を準備します。",
    "이 문장은 첫 번째 응답 전에 언어 감지기를 준비합니다.",
)


def build_language_detector(supported_languages: Sequence[str] | None = None, *, preload: bool = False) -> Any:
    if not LINGUA_AVAILABLE:
        return None
    if supported_languages is None:
        builder = LanguageDetectorBuilder.from_all_languages()
    else:
        lingua_by_code = {
            lang.iso_code_639_1.name.lower(): lang for lang in Language.all() if lang.iso_code_639_1 is not None
        }
        languages = [
            lingua_by_code[_LINGUA_CODE_MAP.get(code, code)]
            for code in supported_languages
            if _LINGUA_CODE_MAP.get(code, code) in lingua_by_code
        ]
        builder = LanguageDetectorBuilder.from_languages(*languages)
    if preload:
        builder = builder.with_preloaded_language_models()
    return builder.build()


@lru_cache(maxsize=2)
def warm_language_detector(supported_languages: tuple[str, ...] | None = None) -> Any:
    """Load models and prime common scripts once during handler setup."""
    detector = build_language_detector(supported_languages, preload=True)
    if detector is not None:
        for sample in _WARMUP_TEXTS:
            detector.detect_language_of(sample)
    return detector


def detect_language_from_text(
    text: str,
    detector: Any = None,
    *,
    minimum_confidence_gap: float = 0.0,
    allow_short_cjk: bool = False,
) -> str | None:
    if detector is None:
        logger.warning("lingua-py not available, cannot detect language from text")
        return None
    stripped = text.strip()
    # Skip very short utterances where language ID is still too noisy.
    minimum = 20
    if allow_short_cjk and any(
        "\u3400" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" or "\uac00" <= char <= "\ud7af"
        for char in stripped
    ):
        minimum = 4
    if len(stripped) < minimum:
        return None
    if minimum_confidence_gap:
        confidences = detector.compute_language_confidence_values(text)
        if not confidences or (
            len(confidences) > 1 and confidences[0].value - confidences[1].value < minimum_confidence_gap
        ):
            return None
        detected = confidences[0].language
    else:
        detected = detector.detect_language_of(text)
    if detected is None or detected.iso_code_639_1 is None:
        return None
    code = detected.iso_code_639_1.name.lower()
    return "no" if code == "nb" else code
