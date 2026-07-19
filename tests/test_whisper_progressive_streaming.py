import numpy as np

import speech_to_speech.STT.smart_progressive_streaming as progressive_module
from speech_to_speech.STT.smart_progressive_streaming import SmartProgressiveStreamingHandler


class FakeASRPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return {
            "text": " Hello world. Next bit.",
            "chunks": [
                {"text": " Hello", "timestamp": (0.0, 0.4)},
                {"text": " world.", "timestamp": (0.4, 0.9)},
                {"text": " Next", "timestamp": (0.9, 1.2)},
                {"text": " bit.", "timestamp": (1.2, 1.5)},
            ],
        }


def _handler(monkeypatch, *, language=None, max_window_size=15.0):
    monkeypatch.setattr(progressive_module, "ensure_punkt", lambda: None)
    monkeypatch.setattr(
        progressive_module,
        "split_sentences",
        lambda _text, _language: ["Hello world.", "Next bit."],
    )
    fake_pipeline = FakeASRPipeline()
    handler = SmartProgressiveStreamingHandler(
        fake_pipeline,
        language=language,
        max_window_size=max_window_size,
        sentence_buffer=0.5,
    )
    return handler, fake_pipeline


def test_auto_language_is_not_forced_into_whisper_generation(monkeypatch):
    handler, fake_pipeline = _handler(monkeypatch)

    result = handler.transcribe_incremental(np.zeros(16000 * 2, dtype=np.float32))

    assert result.active_text == "Hello world. Next bit."
    assert fake_pipeline.calls[0][1]["return_timestamps"] == "word"
    assert "language" not in fake_pipeline.calls[0][1]["generate_kwargs"]


def test_configured_language_is_forwarded_to_every_decode(monkeypatch):
    handler, fake_pipeline = _handler(monkeypatch, language="de")

    handler.transcribe_incremental(np.zeros(16000, dtype=np.float32))

    assert fake_pipeline.calls[0][1]["generate_kwargs"]["language"] == "de"


def test_old_completed_sentence_is_frozen_and_removed_from_active_text(monkeypatch):
    handler, _ = _handler(monkeypatch, max_window_size=1.0)

    result = handler.transcribe_incremental(np.zeros(16000 * 2, dtype=np.float32))

    assert result.fixed_text == "Hello world."
    assert result.active_text == "Next bit."
    assert handler.fixed_end_time == 0.9
