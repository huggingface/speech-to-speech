from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from sys import modules
from typing import Any

from speech_to_speech.LLM.utils import WHISPER_LANGUAGE_TO_LLM_LANGUAGE

_LANGUAGE_NAMES_TO_CODES = {name: code for code, name in WHISPER_LANGUAGE_TO_LLM_LANGUAGE.items()}


def normalize_language(language: str) -> str:
    """Normalize a language code or a known language name without guessing unknown tags."""
    normalized = language.strip().lower()
    return _LANGUAGE_NAMES_TO_CODES.get(normalized, normalized)


@dataclass(frozen=True)
class SpeechLanguageCapabilities:
    """Speech coverage and language controls available through the active handler."""

    languages: frozenset[str] | None = None
    supports_language_selection: bool = True
    supports_auto: bool | None = None

    def __post_init__(self) -> None:
        if self.languages is not None:
            languages = frozenset(normalize_language(language) for language in self.languages)
            if "" in languages or "auto" in languages:
                raise ValueError("Language coverage must contain language codes, not empty values or 'auto'.")
            object.__setattr__(self, "languages", languages)


def backend_language_capabilities(
    kind: str, name: str, config: Mapping[str, Any], handler: Any
) -> SpeechLanguageCapabilities:
    """Read active backend metadata without loading models or optional handler modules."""
    declared = getattr(handler, "language_capabilities", None)
    if isinstance(declared, SpeechLanguageCapabilities):
        return declared

    module = modules.get(type(handler).__module__)
    model = getattr(handler, "model", None)
    languages: Any = None
    selection = True
    auto: bool | None = None

    if kind == "stt":
        selection = name not in {
            "none",
            "parakeet-tdt",
            "parakeet-unified",
            "nemotron-streaming",
            "orukeet",
            "paraformer",
            "sense-voice",
            "openai-realtime",
            "vllm-realtime",
        }
        if name == "faster-whisper":
            # Multilingual decoder vocabulary survives English-only fine-tuning.
            # It constrains hints but cannot establish the checkpoint's coverage.
            supported = getattr(model, "supported_languages", None)
            if supported is not None and set(supported) == {"en"}:
                languages, auto = {"en"}, False
        elif name == "whisper":
            generation = getattr(model, "generation_config", None)
            if getattr(generation, "is_multilingual", None) is False:
                languages, selection, auto = {"en"}, False, False
        elif name == "qwen3-asr":
            languages = getattr(module, "SUPPORTED_LANGUAGES", None)
            auto = True
        elif name == "parakeet-tdt":
            model_name = getattr(handler, "model_name", config.get("model_name"))
            if model_name in {"nvidia/parakeet-tdt-0.6b-v2", "mlx-community/parakeet-tdt-0.6b-v2"}:
                languages, auto = {"en"}, False
            elif model_name in {"nvidia/parakeet-tdt-0.6b-v3", "mlx-community/parakeet-tdt-0.6b-v3"}:
                languages, auto = getattr(module, "SUPPORTED_LANGUAGES", None), True
        elif name in {"parakeet-unified", "nemotron-streaming", "orukeet"}:
            model_name = getattr(handler, "model_name", config.get("model_name"))
            if getattr(handler, "_is_farsi", False):
                languages, auto = {"fa"}, False
            elif model_name in {"nvidia/parakeet-unified-en-0.6b", "nvidia/nemotron-speech-streaming-en-0.6b"}:
                languages, auto = {"en"}, False
            elif getattr(handler, "_detects_utterance_language", False) or getattr(
                handler, "_detect_language_from_text", False
            ):
                auto = True
        elif name == "paraformer" and config.get("model_name") == "paraformer-zh":
            languages, auto = {"zh"}, False
        elif name == "sense-voice":
            auto = str(getattr(handler, "language", config.get("language", ""))).strip().lower() == "auto"
    elif kind == "tts":
        selection = name not in {"chatTTS", "pocket", "kitten", "drama3", "minimax"}
        if name == "kokoro":
            languages, auto = getattr(module, "KOKORO_NATIVE_LANGUAGES", None), False
        elif name == "facebookMMS":
            languages, auto = getattr(module, "WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE", None), False
        elif name == "supertonic":
            codes = getattr(module, "SUPERTONIC_LANGUAGE_CODES", None)
            languages = set(codes) - {"na"} if codes is not None else None
            configured = getattr(handler, "lang", config.get("lang"))
            auto = configured == "na" if configured is not None else None
        elif name == "qwen3":
            for candidate in (
                model,
                getattr(model, "model", None),
                getattr(getattr(model, "model", None), "model", None),
            ):
                get_languages = getattr(candidate, "get_supported_languages", None)
                languages = get_languages() if callable(get_languages) else None
                if languages is None:
                    talker = getattr(getattr(candidate, "config", None), "talker_config", None)
                    languages = (
                        talker.get("codec_language_id")
                        if isinstance(talker, Mapping)
                        else getattr(talker, "codec_language_id", None)
                    )
                if languages is not None:
                    # Qwen's codec table also holds speaker-specific dialect tokens.
                    languages = {
                        language
                        for language in languages
                        if normalize_language(language) != "auto" and "dialect" not in language.lower()
                    }
                    break
            auto = True
        elif name == "pocket":
            # A checkpoint override can select Farsi while the language option still says English.
            if getattr(handler, "is_farsi", False) or config.get("model_name") == "mehdi-hf/pocket-tts-farsi-v2":
                languages = {"fa"}
            else:
                language = getattr(handler, "language", config.get("language"))
                if isinstance(language, str):
                    code = normalize_language(language.partition("_")[0])
                    if code in WHISPER_LANGUAGE_TO_LLM_LANGUAGE:
                        languages = {code}
            auto = False
        elif name == "chatTTS":
            languages, auto = {"en", "zh"}, True
        elif name == "phonon":
            languages, auto = getattr(model, "languages", None), False
        elif name == "kitten":
            if config.get("model_name") == "KittenML/kitten-tts-mini-0.8":
                languages = {"en"}
            auto = False
        elif name == "drama3":
            auto = True
    else:
        raise ValueError(f"Speech language capabilities require an STT or TTS backend, got {kind!r}.")

    return SpeechLanguageCapabilities(
        languages=frozenset(languages) if languages is not None else None,
        supports_language_selection=selection,
        supports_auto=auto,
    )
