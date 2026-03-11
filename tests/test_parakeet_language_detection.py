from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from STT import parakeet_tdt_handler
from STT.parakeet_tdt_handler import ParakeetTDTSTTHandler


def test_detect_language_from_short_text_returns_none_without_querying_detector(monkeypatch):
    handler = object.__new__(ParakeetTDTSTTHandler)

    class ExplodingDetector:
        def detect_language_of(self, text):
            raise AssertionError("short text should not invoke lingua")

    monkeypatch.setattr(parakeet_tdt_handler, "LINGUA_AVAILABLE", True)
    monkeypatch.setattr(parakeet_tdt_handler, "_lingua_detector", ExplodingDetector())

    # langdetect commonly misclassifies this short English sentence as Tagalog ("tl")
    assert handler._detect_language_from_text("Okay.") is None


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
