import pytest

from speech_to_speech.pipeline import language_detection
from speech_to_speech.STT import parakeet_tdt_handler
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler


def test_build_lingua_detector_preloads_language_models(monkeypatch):
    if not language_detection.LINGUA_AVAILABLE:
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

    monkeypatch.setattr(language_detection, "LanguageDetectorBuilder", BuilderFactory)

    assert language_detection._build_lingua_detector() is detector
    assert calls[0][0] == "languages"
    assert {lang.iso_code_639_1.name.lower() for lang in calls[0][1]} == set(
        language_detection.DETECTABLE_LANGUAGE_CODES
    )
    assert calls[1:] == ["preload", "build"]


def test_detect_language_from_short_text_returns_none_without_querying_detector(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)

    class ExplodingDetector:
        def detect_language_of(self, text):
            raise AssertionError("short text should not invoke lingua")

    monkeypatch.setattr(parakeet_tdt_handler, "LINGUA_AVAILABLE", True)
    monkeypatch.setattr(language_detection, "LINGUA_AVAILABLE", True)
    monkeypatch.setattr(language_detection, "warm_language_detector", lambda: ExplodingDetector())

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
    if not parakeet_tdt_handler.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    handler = object.__new__(ParakeetTDTSTTHandler)

    assert len(text) >= 20
    assert handler._detect_language_from_text(text) == "en"


def test_detect_language_from_long_norwegian_text_maps_nb_to_no():
    if not parakeet_tdt_handler.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    handler = object.__new__(ParakeetTDTSTTHandler)
    text = (
        "Jeg heter Øyvind og bor i Trondheim. Denne teksten er lang nok til å teste "
        "om språkdetektoren virkelig klarer å skille norsk bokmål fra dansk og svensk "
        "i et realistisk avsnitt med vanlige ord og tegn."
    )

    assert handler._detect_language_from_text(text) == "no"


@pytest.mark.parametrize(
    ("text", "code"),
    [("你好世界", "zh"), ("こんにちは", "ja"), ("안녕하세요", "ko")],
)
def test_assistant_detector_accepts_short_cjk_text(text, code):
    if not language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    assert language_detection.detect_language_from_text(text) == code
