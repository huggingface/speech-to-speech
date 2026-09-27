"""Shared, preloaded text language detector for STT and assistant speech."""

from __future__ import annotations

from threading import Lock
from typing import Any

try:
    from lingua import Language, LanguageDetectorBuilder

    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False

# The Parakeet languages plus the remaining scripts supported by Qwen3-TTS.
# Lingua calls Norwegian Bokmål "nb"; pipeline language codes use "no".
DETECTABLE_LANGUAGE_CODES = (
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "ru",
    "uk",
    "cs",
    "sk",
    "hu",
    "ro",
    "bg",
    "hr",
    "sl",
    "sr",
    "da",
    "nb",
    "sv",
    "fi",
    "et",
    "lv",
    "lt",
    "zh",
    "ja",
    "ko",
)
MIN_DETECTION_CHARS = 20
MIN_CJK_DETECTION_CHARS = 4
_WARMUP_TEXTS = (
    "This sentence warms the language detector before the first response.",
    "Hello, I can help you find the information you need today.",
    "The weather should be pleasant tomorrow afternoon.",
    "I would be happy to explain the next steps in more detail.",
    "Bonjour, je peux vous aider à trouver la bonne réponse.",
    "Je serai ravi de vous expliquer les prochaines étapes.",
    "Guten Tag, ich helfe Ihnen gerne bei dieser Frage.",
    "Die nächste Möglichkeit finden wir morgen früh.",
    "Hola, puedo ayudarte a encontrar la mejor respuesta.",
    "Mañana tendremos más información sobre este tema.",
    "Posso aiutarti a trovare le informazioni che cerchi.",
    "Posso ajudar você a encontrar a resposta correta.",
    "Это предложение прогревает определение языка перед первым ответом.",
    "Я могу помочь вам найти нужную информацию сегодня.",
    "这句话会在第一次回复之前预热语言检测器。",
    "この文章は最初の応答の前に言語検出器を準備します。",
    "이 문장은 첫 번째 응답 전에 언어 감지기를 준비합니다.",
)
_detector: Any = None
_detector_lock = Lock()


def _build_lingua_detector():
    if not LINGUA_AVAILABLE:
        raise RuntimeError("Language detection requires lingua-language-detector")
    languages = [
        language
        for language in Language.all()
        if language.iso_code_639_1 is not None and language.iso_code_639_1.name.lower() in DETECTABLE_LANGUAGE_CODES
    ]
    return LanguageDetectorBuilder.from_languages(*languages).with_preloaded_language_models().build()


def warm_language_detector():
    """Load models and pay first-detection costs during handler setup."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                detector = _build_lingua_detector()
                for sample in _WARMUP_TEXTS:
                    detector.detect_language_of(sample)
                _detector = detector
    return _detector


def detect_language_from_text(text: str, *, detector=None) -> str | None:
    """Return a pipeline language code when there is enough text to classify."""
    stripped = text.strip()
    if not stripped or not LINGUA_AVAILABLE:
        return None
    has_cjk_script = any(
        "\u3400" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" or "\uac00" <= char <= "\ud7af"
        for char in stripped
    )
    minimum = MIN_CJK_DETECTION_CHARS if has_cjk_script else MIN_DETECTION_CHARS
    if len(stripped) < minimum:
        return None
    active_detector = detector if detector is not None else warm_language_detector()
    detected = active_detector.detect_language_of(text)
    if detected is None or detected.iso_code_639_1 is None:
        return None
    code = detected.iso_code_639_1.name.lower()
    return "no" if code == "nb" else code
