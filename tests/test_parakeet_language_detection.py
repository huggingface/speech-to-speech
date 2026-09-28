import pytest

from speech_to_speech.STT import parakeet_tdt_handler
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler
from speech_to_speech.utils import text_language_detection


def test_build_lingua_detector_preloads_language_models(monkeypatch):
    if not text_language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    calls = []
    detector = object()

    class Builder:
        def with_preloaded_language_models(self):
            calls.append("preload")
            return self

        def build(self):
            calls.append("build")
            return detector

    class BuilderFactory:
        @staticmethod
        def from_languages(*languages):
            calls.append(("languages", languages))
            return Builder()

    monkeypatch.setattr(text_language_detection, "LanguageDetectorBuilder", BuilderFactory)

    assert parakeet_tdt_handler._build_lingua_detector() is detector
    assert calls[0][0] == "languages"
    assert len(calls[0][1]) == len(parakeet_tdt_handler.SUPPORTED_LANGUAGES)
    assert calls[1:] == ["preload", "build"]


def test_detect_language_from_short_text_returns_none_without_querying_detector(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)

    class ExplodingDetector:
        def detect_language_of(self, text):
            raise AssertionError("short text should not invoke lingua")

    monkeypatch.setattr(parakeet_tdt_handler, "_lingua_detector", ExplodingDetector(), raising=False)

    assert handler._detect_language_from_text("Okay.") is None


@pytest.mark.parametrize(
    "text",
    [
        "Okay, open the door.",
        "Okay, try that again.",
        "Open the door, please.",
    ],
)
def test_detect_language_from_short_english_text_uses_lingua_successfully(text):
    if not text_language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    handler = object.__new__(ParakeetTDTSTTHandler)

    assert len(text) >= 20
    assert handler._detect_language_from_text(text) == "en"


def test_detect_language_from_long_norwegian_text_maps_nb_to_no():
    if not text_language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    handler = object.__new__(ParakeetTDTSTTHandler)
    text = (
        "Jeg heter Øyvind og bor i Trondheim. Denne teksten er lang nok til å teste "
        "om språkdetektoren virkelig klarer å skille norsk bokmål fra dansk og svensk "
        "i et realistisk avsnitt med vanlige ord og tegn."
    )

    assert handler._detect_language_from_text(text) == "no"
