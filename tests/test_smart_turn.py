import sys
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import torch

from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.smart_turn import (
    MAX_AUDIO_SECONDS,
    MODEL_SAMPLE_RATE,
    SmartTurnAnalyzer,
    SmartTurnResult,
)
from speech_to_speech.VAD.vad_handler import VADHandler


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
    handler.smart_turn_analyzer = analyzer
    handler.speculative_reopen_ms = 800
    handler.smart_turn_max_wait_ms = 2000
    handler.smart_turn_incomplete_delay_ms = 600
    return handler


def _result(complete: bool, probability: float) -> SmartTurnResult:
    return SmartTurnResult(complete=complete, probability=probability, inference_ms=12.5)


def test_complete_turn_selects_default_speculative_grace() -> None:
    analyzer = _FakeAnalyzer(_result(True, 0.9))
    handler = _handler_with_smart_turn(analyzer)
    audio = np.arange(32, dtype=np.float32)

    assert handler._smart_turn_timing_ms(audio) == (800, 0)
    assert len(analyzer.calls) == 1
    np.testing.assert_array_equal(analyzer.calls[0][0], audio)


def test_incomplete_turn_selects_longer_speculative_grace() -> None:
    analyzer = _FakeAnalyzer(_result(False, 0.2))
    handler = _handler_with_smart_turn(analyzer)
    full_turn = np.arange(10, dtype=np.float32)

    assert handler._smart_turn_timing_ms(full_turn) == (2000, 600)
    np.testing.assert_array_equal(analyzer.calls[0][0], full_turn)


def test_inference_failure_uses_default_speculative_grace() -> None:
    analyzer = _FakeAnalyzer()
    handler = _handler_with_smart_turn(analyzer)
    audio = np.ones(32, dtype=np.float32)

    assert handler._smart_turn_timing_ms(audio) == (800, 0)
    assert len(analyzer.calls) == 1


def test_unanswered_reopen_cap_covers_smart_turn_wait(monkeypatch) -> None:
    class FakeSileroModel:
        def reset_states(self) -> None:
            pass

    monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: (FakeSileroModel(), None))
    monkeypatch.setattr(
        "speech_to_speech.VAD.smart_turn.SmartTurnAnalyzer",
        lambda **_kwargs: object(),
    )
    tracker = SpeculativeTurnTracker()
    handler = object.__new__(VADHandler)

    handler.setup(
        Event(),
        text_output_queue=Queue(),
        speculative_turns=tracker,
        unanswered_reopen_ms=1000,
        smart_turn_max_wait_ms=2000,
    )

    assert handler.unanswered_reopen_ms == 2000
    tracker.observe("turn_1", 0)
    handler._current_turn_id = "turn_1"
    handler._current_turn_revision = 0
    handler._last_final_audio_ms = 0
    assert handler._should_reopen_current_turn(1500)
    assert not handler._should_reopen_current_turn(2001)


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


def test_default_model_download_uses_v32_cpu_variant(monkeypatch, tmp_path: Path) -> None:
    downloaded = tmp_path / "model.onnx"
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))

    assert SmartTurnAnalyzer._download_model() == downloaded
    assert calls == [
        {
            "repo_id": "pipecat-ai/smart-turn-v3",
            "filename": "smart-turn-v3.2-cpu.onnx",
        }
    ]


def test_analyzer_always_uses_cpu_execution_provider(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    calls = []

    class FakeSessionOptions:
        pass

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input_features")]

    def fake_inference_session(path, *, sess_options, providers):
        calls.append((path, sess_options, providers))
        return FakeSession()

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=fake_inference_session,
    )
    fake_transformers = SimpleNamespace(WhisperFeatureExtractor=lambda **_kwargs: object())
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    SmartTurnAnalyzer(model_path=str(model_path), cpu_count=3, warmup=False)

    assert len(calls) == 1
    path, options, providers = calls[0]
    assert path == str(model_path)
    assert providers == ["CPUExecutionProvider"]
    assert options.execution_mode == "sequential"
    assert options.inter_op_num_threads == 1
    assert options.intra_op_num_threads == 3
    assert options.graph_optimization_level == "all"


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
