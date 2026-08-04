import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from speech_to_speech.VAD.smart_turn import (
    MAX_AUDIO_SECONDS,
    MODEL_SAMPLE_RATE,
    SmartTurnAnalyzer,
    SmartTurnResult,
)
from speech_to_speech.VAD.vad_handler import VADHandler, _SmartTurnDecision


class _FakeIterator:
    def __init__(self) -> None:
        self.triggered = False
        self.temp_end = 123
        self.buffer: list[torch.Tensor] = []
        self.prefix_buffer = [torch.ones(2)]
        self.active_speech_samples = 0
        self.last_utterance_active_speech_samples = 800


class _FakeAnalyzer:
    def __init__(self, *results: SmartTurnResult) -> None:
        self.results = iter(results)
        self.calls: list[tuple[np.ndarray, int]] = []

    def predict(self, audio: np.ndarray, *, sample_rate: int) -> SmartTurnResult:
        self.calls.append((audio, sample_rate))
        return next(self.results)


def _handler_with_smart_turn(analyzer: _FakeAnalyzer) -> VADHandler:
    handler = object.__new__(VADHandler)
    handler.sample_rate = MODEL_SAMPLE_RATE
    handler.iterator = _FakeIterator()
    handler.smart_turn_analyzer = analyzer
    handler.smart_turn_max_wait_samples = 3 * MODEL_SAMPLE_RATE
    handler._smart_turn_pending_since_sample = None
    handler._smart_turn_last_active_samples = 0
    handler._total_samples = MODEL_SAMPLE_RATE
    return handler


def _result(complete: bool, probability: float) -> SmartTurnResult:
    return SmartTurnResult(complete=complete, probability=probability, inference_ms=12.5)


def test_incomplete_turn_is_restored_to_streaming_vad() -> None:
    analyzer = _FakeAnalyzer(_result(False, 0.2))
    handler = _handler_with_smart_turn(analyzer)
    audio = np.arange(32, dtype=np.float32)

    assert handler._smart_turn_should_finalize(audio, active_speech_samples=800) is _SmartTurnDecision.CONTINUE
    assert len(analyzer.calls) == 1
    assert handler._smart_turn_pending_since_sample == MODEL_SAMPLE_RATE
    assert handler.iterator.triggered is True
    assert handler.iterator.temp_end == 0
    assert handler.iterator.active_speech_samples == 800
    assert handler.iterator.last_utterance_active_speech_samples == 0
    assert handler.iterator.prefix_buffer == []
    np.testing.assert_array_equal(handler.iterator.buffer[0].numpy(), audio)


def test_analysis_can_include_prior_turn_audio_without_restoring_it() -> None:
    analyzer = _FakeAnalyzer(_result(False, 0.2))
    handler = _handler_with_smart_turn(analyzer)
    current_segment = np.arange(4, dtype=np.float32)
    full_turn = np.arange(10, dtype=np.float32)

    assert (
        handler._smart_turn_should_finalize(
            current_segment,
            active_speech_samples=800,
            analysis_audio=full_turn,
        )
        is _SmartTurnDecision.CONTINUE
    )

    np.testing.assert_array_equal(analyzer.calls[0][0], full_turn)
    np.testing.assert_array_equal(handler.iterator.buffer[0].numpy(), current_segment)


def test_pending_turn_only_runs_inference_again_after_more_speech() -> None:
    analyzer = _FakeAnalyzer(_result(False, 0.2), _result(True, 0.9))
    handler = _handler_with_smart_turn(analyzer)
    audio = np.ones(32, dtype=np.float32)

    assert handler._smart_turn_should_finalize(audio, active_speech_samples=800) is _SmartTurnDecision.CONTINUE
    handler._total_samples += 512
    assert handler._smart_turn_should_finalize(audio, active_speech_samples=800) is _SmartTurnDecision.CONTINUE
    assert len(analyzer.calls) == 1

    handler._total_samples += 512
    assert handler._smart_turn_should_finalize(audio, active_speech_samples=1312) is _SmartTurnDecision.COMPLETE
    assert len(analyzer.calls) == 2
    assert handler._smart_turn_pending_since_sample is None


def test_pending_turn_finalizes_at_bounded_silence_timeout() -> None:
    analyzer = _FakeAnalyzer(_result(False, 0.2))
    handler = _handler_with_smart_turn(analyzer)
    audio = np.ones(32, dtype=np.float32)

    assert handler._smart_turn_should_finalize(audio, active_speech_samples=800) is _SmartTurnDecision.CONTINUE
    handler._total_samples += handler.smart_turn_max_wait_samples

    assert handler._smart_turn_should_finalize(audio, active_speech_samples=800) is _SmartTurnDecision.MAX_WAIT
    assert len(analyzer.calls) == 1
    assert handler._smart_turn_pending_since_sample is None


def test_prepare_audio_keeps_latest_eight_seconds_and_left_pads() -> None:
    max_samples = MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE
    short_audio = np.arange(100, dtype=np.float32)
    prepared_short = SmartTurnAnalyzer._prepare_audio(short_audio, MODEL_SAMPLE_RATE)
    assert prepared_short.shape == (max_samples,)
    np.testing.assert_array_equal(prepared_short[-100:], short_audio)
    assert np.count_nonzero(prepared_short[:-100]) == 0

    long_audio = np.arange(max_samples + 10, dtype=np.float32)
    prepared_long = SmartTurnAnalyzer._prepare_audio(long_audio, MODEL_SAMPLE_RATE)
    np.testing.assert_array_equal(prepared_long, long_audio[-max_samples:])


def test_prepare_audio_resamples_to_model_rate() -> None:
    prepared = SmartTurnAnalyzer._prepare_audio(np.ones(8000, dtype=np.float32), sample_rate=8000)

    assert prepared.shape == (MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE,)
    assert np.count_nonzero(prepared[-MODEL_SAMPLE_RATE:]) > 0


def test_default_model_download_uses_v32_variant(monkeypatch, tmp_path: Path) -> None:
    downloaded = tmp_path / "model.onnx"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))

    assert SmartTurnAnalyzer._download_model("cpu") == downloaded
    assert calls == [
        {
            "repo_id": "pipecat-ai/smart-turn-v3",
            "filename": "smart-turn-v3.2-cpu.onnx",
        }
    ]


def test_predict_passes_whisper_features_to_onnx_session() -> None:
    class FakeFeatureExtractor:
        def __call__(self, audio, **kwargs):
            assert audio.shape == (MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE,)
            assert kwargs["sampling_rate"] == MODEL_SAMPLE_RATE
            return SimpleNamespace(input_features=np.ones((1, 80, 800), dtype=np.float64))

    class FakeSession:
        def __init__(self) -> None:
            self.feeds = []

        def run(self, _outputs, feeds):
            self.feeds.append(feeds)
            return [np.array([[0.75]], dtype=np.float32)]

    analyzer = object.__new__(SmartTurnAnalyzer)
    analyzer.threshold = 0.5
    analyzer.input_name = "input_features"
    analyzer.feature_extractor = FakeFeatureExtractor()
    analyzer.session = FakeSession()

    result = analyzer.predict(np.ones(100, dtype=np.float32))

    assert result.complete is True
    assert result.probability == 0.75
    features = analyzer.session.feeds[0]["input_features"]
    assert features.shape == (1, 80, 800)
    assert features.dtype == np.float32
