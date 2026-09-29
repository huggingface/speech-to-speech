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

    assert (
        language_detection.build_language_detector(parakeet_tdt_handler.SUPPORTED_LANGUAGES, preload=True) is detector
    )
    assert calls[0][0] == "languages"
    assert {lang.iso_code_639_1.name.lower() for lang in calls[0][1]} == set(
        ["nb" if code == "no" else code for code in parakeet_tdt_handler.SUPPORTED_LANGUAGES]
    )
    assert calls[1:] == ["preload", "build"]


def test_detect_language_from_short_text_returns_none_without_querying_detector(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)

    class ExplodingDetector:
        def detect_language_of(self, text):
            raise AssertionError("short text should not invoke lingua")

    handler._language_detector = ExplodingDetector()

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
    if not language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    handler = object.__new__(ParakeetTDTSTTHandler)

    assert len(text) >= 20
    handler._language_detector = language_detection.warm_language_detector(
        tuple(parakeet_tdt_handler.SUPPORTED_LANGUAGES)
    )
    assert handler._detect_language_from_text(text) == "en"


def test_detect_language_from_long_norwegian_text_maps_nb_to_no():
    if not language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler._language_detector = language_detection.warm_language_detector(
        tuple(parakeet_tdt_handler.SUPPORTED_LANGUAGES)
    )
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

    assert (
        language_detection.detect_language_from_text(
            text, language_detection.warm_language_detector(), allow_short_cjk=True
        )
        == code
    )


def test_assistant_detector_declines_ambiguous_numeric_chunk():
    if not language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    assert (
        language_detection.detect_language_from_text(
            "100 times 100000 is 10,000,000.",
            language_detection.warm_language_detector(),
            minimum_confidence_gap=language_detection.MIN_ASSISTANT_CONFIDENCE_GAP,
        )
        is None
    )
    assert (
        language_detection.detect_language_from_text(
            "Bonjour, je peux vous aider aujourd'hui.",
            language_detection.warm_language_detector(),
            minimum_confidence_gap=language_detection.MIN_ASSISTANT_CONFIDENCE_GAP,
        )
        == "fr"
    )
