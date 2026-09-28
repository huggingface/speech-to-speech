"""Text language detection shared by transcription metadata and assistant speech."""

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

# Lingua uses "nb" (Bokmål) for Norwegian instead of "no".
_LINGUA_CODE_MAP = {"no": "nb"}


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
        # Pay model-loading cost at startup instead of on the first STT request.
        builder = builder.with_preloaded_language_models()
    return builder.build()


@lru_cache(maxsize=1)
def all_language_detector() -> Any:
    return build_language_detector()


def detect_language_from_text(text: str, detector: Any) -> str | None:
    if detector is None:
        logger.warning("lingua-py not available, cannot detect language from text")
        return None
    # Skip very short utterances where language ID is still too noisy.
    if not text or len(text.strip()) < 20:
        return None
    detected = detector.detect_language_of(text)
    if detected is None:
        return None
    code = detected.iso_code_639_1.name.lower()
    return {v: k for k, v in _LINGUA_CODE_MAP.items()}.get(code, code)
