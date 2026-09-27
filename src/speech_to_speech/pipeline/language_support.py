"""Which languages the selected STT and TTS routes can actually handle.

A conversation language is only usable if *both* ends support it: the STT backend has to
transcribe it and the TTS backend has to speak it. Those sets differ sharply -- the default
STT reports 25 languages while Kokoro can voice 8 -- so a selection accepted by one end can
still be silently wrong at the other.

Capability is read from each backend's own ``SUPPORTED_LANGUAGES`` via the route metadata in
``BackendCapabilities.languages_source``, lazily, so an uninstalled optional extra reports
*unknown* rather than crashing. Unknown is deliberately distinct from empty: a backend that
does not declare its languages must never be reported as supporting one, and must never be
used to reject a selection either.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

from speech_to_speech.backend_registry import STT_BACKENDS, TTS_BACKENDS, BackendSpec

logger = logging.getLogger(__name__)


def declared_languages(spec: BackendSpec | None) -> frozenset[str] | None:
    """Languages *spec* declares it supports, or ``None`` when it declares nothing.

    ``None`` is also returned when the backend's optional extra is not installed: its
    capability is unknown here, which is not the same as unsupported.
    """
    if spec is None:
        return None
    source = spec.capabilities.languages_source
    if source is None:
        return None
    module_name, attribute = source
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.debug("Language capability for %r is unknown: %s is not importable", spec.name, module_name)
        return None
    languages = getattr(module, attribute, None)
    if languages is None:
        return None
    return frozenset(languages)


def backend_languages(kind: str, name: str | None) -> frozenset[str] | None:
    """Declared languages for a named STT or TTS backend."""
    registry = STT_BACKENDS if kind == "stt" else TTS_BACKENDS
    return declared_languages(registry.get(name)) if name else None


def usable_languages(stt: str | None, tts: str | None) -> frozenset[str] | None:
    """Languages both routes support, or ``None`` if either side is undeclared.

    The intersection is only meaningful when both ends are known; with one unknown the
    honest answer is that the usable set cannot be determined.
    """
    stt_languages = backend_languages("stt", stt)
    tts_languages = backend_languages("tts", tts)
    if stt_languages is None or tts_languages is None:
        return None
    return stt_languages & tts_languages


@dataclass(frozen=True)
class LanguageSupport:
    """Whether a language is usable end to end, and why not when it is not.

    ``supported`` is ``None`` when capability is undeclared -- callers should neither claim
    support nor reject the selection.
    """

    language: str
    supported: bool | None
    reason: str

    @property
    def is_rejected(self) -> bool:
        """True only when a route is known not to support the language."""
        return self.supported is False


def check_language(language: str | None, *, stt: str | None, tts: str | None) -> LanguageSupport:
    """Check *language* against the selected routes.

    ``auto`` and an absent selection are always accepted: they ask the pipeline to decide,
    rather than naming a language a backend has to support.
    """
    if not language or language == "auto":
        return LanguageSupport(language or "auto", True, "automatic language selection")

    stt_languages = backend_languages("stt", stt)
    tts_languages = backend_languages("tts", tts)
    missing = [
        f"{kind} backend {name!r}"
        for kind, name, languages in (("STT", stt, stt_languages), ("TTS", tts, tts_languages))
        if languages is not None and language not in languages
    ]
    if missing:
        return LanguageSupport(language, False, f"{language!r} is not supported by the " + " or the ".join(missing))

    undeclared = [
        f"{kind} backend {name!r}"
        for kind, name, languages in (("STT", stt, stt_languages), ("TTS", tts, tts_languages))
        if languages is None
    ]
    if undeclared:
        return LanguageSupport(
            language,
            None,
            f"{' and the '.join(undeclared)} does not declare supported languages, so {language!r} cannot be verified",
        )
    return LanguageSupport(language, True, f"{language!r} is supported by both routes")


def log_usable_languages(stt: str | None, tts: str | None) -> None:
    """Report the end-to-end usable language set once at startup.

    Surfaces the asymmetry between the two routes before a conversation starts, rather than
    leaving a user to discover that a language transcribes fine but is voiced in the wrong
    one. Logs only -- selection is not rejected here, because an undeclared backend may well
    support more than it advertises.
    """
    usable = usable_languages(stt, tts)
    if usable is None:
        logger.debug("Usable languages for %s + %s are unknown: at least one route is undeclared", stt, tts)
        return
    if not usable:
        logger.warning(
            "STT %r and TTS %r declare no language in common; every turn will fall back on the "
            "TTS default voice regardless of the language spoken",
            stt,
            tts,
        )
        return
    logger.info("Languages usable end to end with STT %r and TTS %r: %s", stt, tts, ", ".join(sorted(usable)))
