from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.backend_registry import STT_BACKENDS, BackendSelection
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.s2s_pipeline import _stt_session_languages, parse_arguments
from speech_to_speech.STT import zerogpu_handler


@pytest.fixture
def handler(monkeypatch):
    runtime = SimpleNamespace(
        get_model_ids=lambda: {"stt": "nvidia/parakeet-tdt-0.6b-v3"},
        transcribe=lambda audio: "Hello.",
    )
    monkeypatch.setattr(zerogpu_handler, "runtime", runtime)
    monkeypatch.setattr(zerogpu_handler, "warm_language_detector", lambda languages: object())
    monkeypatch.setattr(zerogpu_handler, "detect_language_from_text", lambda text, detector: None)
    return zerogpu_handler.ZeroGPUSTTHandler(Event(), Queue(), Queue()), runtime


@pytest.mark.parametrize("mode, expected", [("progressive", PartialTranscription), ("final", Transcription)])
def test_transcription_preserves_native_turn_metadata(handler, mode, expected):
    h, _ = handler
    item = VADAudio(audio=np.zeros(1600), mode=mode, turn_id="turn", turn_revision=2, speech_end_at_s=10)
    (output,) = h.process(item)
    assert isinstance(output, expected)
    assert (output.text, output.turn_id, output.turn_revision) == ("Hello.", "turn", 2)
    if mode == "final":
        assert output.speech_stopped_at_s == 10


def test_failed_final_transcription_reports_failure_without_generating_text(handler):
    h, runtime = handler

    def fail(audio):
        raise RuntimeError("allocation expired")

    runtime.transcribe = fail
    (output,) = h.process(VADAudio(audio=np.zeros(1600), mode="final", turn_id="turn", turn_revision=2))
    assert isinstance(output, TranscriptionFailure)
    assert (output.turn_id, output.turn_revision) == ("turn", 2)
    assert list(h.process(VADAudio(audio=np.zeros(1600), mode="progressive"))) == []


def test_native_registry_configuration_and_language_capability():
    args = parse_arguments(["--stt", "zerogpu", "--llm_backend", "zerogpu", "--tts", "zerogpu"])
    assert [args.stt_backend.name, args.llm_backend.name, args.tts_backend.name] == ["zerogpu"] * 3
    assert args.llm_backend.config["compact_history"] is False
    assert args.llm_backend.config["gen_kwargs"]["max_new_tokens"] == 128
    assert _stt_session_languages(BackendSelection(STT_BACKENDS["zerogpu"], {}), object()) == set()


def test_detected_language_is_automatic_and_session_state_resets(handler, monkeypatch):
    h, runtime = handler
    runtime.transcribe = lambda audio: "Bonjour, comment allez-vous aujourd’hui ?"
    monkeypatch.setattr(zerogpu_handler, "detect_language_from_text", lambda text, detector: "fr")
    (output,) = h.process(VADAudio(audio=np.zeros(1600), mode="final"))
    assert output.language_code == "fr-auto"
    assert h.last_language == "fr"
    assert "fr" in h.language_capabilities.languages
    assert not h.language_capabilities.supports_language_selection
    h.on_session_end()
    assert h.last_language is None


def test_short_transcript_does_not_get_mislabeled_english(handler):
    h, _ = handler
    (output,) = h.process(VADAudio(audio=np.zeros(1600), mode="final"))
    assert output.language_code is None
