from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.TTS import facebookmms_handler
from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler


def _waveform(samples: int = 16000) -> torch.Tensor:
    return torch.linspace(-0.2, 0.2, samples, dtype=torch.float32).unsqueeze(0)


def _handler(*, stream: bool = True, waveform: torch.Tensor | None = None) -> FacebookMMSTTSHandler:
    handler = object.__new__(FacebookMMSTTSHandler)
    handler.stream = stream
    handler.chunk_size = 512
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.language = "en"
    handler._initial_language = "en"
    handler._initial_model_name = None
    handler.model_name = "facebook/mms-tts-eng"
    handler.model = SimpleNamespace(config=SimpleNamespace(sampling_rate=16000))
    handler.generate_audio = lambda _text: waveform  # type: ignore[method-assign]
    return handler


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(facebookmms_handler.console, "print", lambda *_a, **_k: None)


def test_records_ttfa_before_resample_and_e2e_on_first_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])

    def slow_resample(chunk, *, orig_sr, target_sr):
        clock[0] = 100.40
        return np.asarray(chunk, dtype=np.float32)

    # Avoid importing real librosa/numba in unit tests.
    monkeypatch.setattr(facebookmms_handler, "librosa", SimpleNamespace(resample=slow_resample))

    def generate(_text: str) -> torch.Tensor:
        clock[0] = 100.12
        return _waveform()

    handler = _handler(waveform=_waveform())
    handler.generate_audio = generate  # type: ignore[method-assign]
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = list(
        handler.process(
            TTSInput(
                text="hello",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=99.0,
            )
        )
    )

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.12)
    assert tracker.e2e_s == pytest.approx(1.40)


def test_preserves_first_audio_across_later_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [50.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        facebookmms_handler,
        "librosa",
        SimpleNamespace(resample=lambda chunk, *, orig_sr, target_sr: np.asarray(chunk, dtype=np.float32)),
    )

    def first_generate(_text: str) -> torch.Tensor:
        clock[0] = 50.08
        return _waveform()

    handler = _handler()
    handler.generate_audio = first_generate  # type: ignore[method-assign]
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    list(
        handler.process(
            TTSInput(
                text="one",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=49.5,
            )
        )
    )
    assert tracker.tts_ttfa_s == pytest.approx(0.08)
    first_e2e = tracker.e2e_s
    assert first_e2e is not None

    clock[0] = 80.0

    def later_generate(_text: str) -> torch.Tensor:
        clock[0] = 80.50
        return _waveform()

    handler.generate_audio = later_generate  # type: ignore[method-assign]
    list(
        handler.process(
            TTSInput(
                text="two",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=49.5,
            )
        )
    )

    assert tracker.tts_ttfa_s == pytest.approx(0.08)
    assert tracker.e2e_s == pytest.approx(first_e2e)


def test_language_reload_is_excluded_from_ttfa(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [10.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        facebookmms_handler,
        "librosa",
        SimpleNamespace(resample=lambda chunk, *, orig_sr, target_sr: np.asarray(chunk, dtype=np.float32)),
    )

    handler = _handler()
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-2", turn_id="turn_2", turn_revision=0)

    def reload(_language_code: str, _model_name: str | None = None) -> None:
        clock[0] = 10.50  # expensive switch must not enter TTFA
        handler.language = "fr"
        handler.model_name = "facebook/mms-tts-fra"

    def generate(_text: str) -> torch.Tensor:
        clock[0] = 10.57
        return _waveform()

    handler.load_model = reload  # type: ignore[method-assign]
    handler.generate_audio = generate  # type: ignore[method-assign]

    list(
        handler.process(
            TTSInput(
                text="bonjour",
                response_key="resp-2",
                turn_id="turn_2",
                turn_revision=0,
                tts_language_code="fr",
                speech_stopped_at_s=9.0,
            )
        )
    )

    assert tracker.tts_ttfa_s == pytest.approx(0.07)
    assert tracker.e2e_s == pytest.approx(1.57)


def test_empty_audio_leaves_latency_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])

    handler = _handler(waveform=torch.empty(0))
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-3", turn_id="turn_3", turn_revision=0)

    assert list(handler.process(TTSInput(text="x", response_key="resp-3", speech_stopped_at_s=0.5))) == []
    assert tracker.tts_ttfa_s is None
    assert tracker.e2e_s is None


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps", "xpu", "npu"])
@pytest.mark.parametrize("cancel_during_wait", [False, True])
def test_ttfa_waits_for_waveform_readiness_before_cpu_copy(
    monkeypatch: pytest.MonkeyPatch, device_type: str, cancel_during_wait: bool
) -> None:
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    scope = CancelScope()
    synchronized_devices = []
    device = SimpleNamespace(type=device_type)

    def synchronize() -> None:
        synchronized_devices.append(device)
        clock[0] = 100.40
        if cancel_during_wait:
            scope.cancel()

    def current_stream(actual_device):
        assert actual_device is device
        return SimpleNamespace(synchronize=synchronize)

    for backend in ("cuda", "xpu", "npu"):
        monkeypatch.setattr(torch, backend, SimpleNamespace(current_stream=current_stream), raising=False)
    monkeypatch.setattr(torch, "mps", SimpleNamespace(synchronize=synchronize))

    class DeferredWaveform:
        def __init__(self):
            self.device = device

        def numel(self):
            return 16000

        def cpu(self):
            # GPU computation must finish before we begin copying samples.
            assert len(synchronized_devices) == (0 if device_type == "cpu" else 1)
            clock[0] += 0.05
            return _waveform()

    def resample(chunk, **kwargs):
        clock[0] += 0.20
        return np.asarray(chunk, dtype=np.float32)

    monkeypatch.setattr(facebookmms_handler, "librosa", SimpleNamespace(resample=resample))
    handler = _handler()
    handler.cancel_scope = scope
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-ready")

    def generate(_text):
        clock[0] = 100.12
        return DeferredWaveform()

    handler.generate_audio = generate  # type: ignore[method-assign]
    outputs = list(handler.process(TTSInput(text="hello", response_key="resp-ready", speech_stopped_at_s=99.0)))

    if cancel_during_wait and device_type != "cpu":
        assert outputs == []
        assert tracker.tts_ttfa_s is None
        assert tracker.e2e_s is None
    else:
        assert outputs
        assert tracker.tts_ttfa_s == pytest.approx(0.12 if device_type == "cpu" else 0.40)
        assert tracker.e2e_s == pytest.approx(1.37 if device_type == "cpu" else 1.65)


@pytest.mark.parametrize("cancel_at", ["generation", "resample", "after_first", "discard"])
def test_cancelled_or_discarded_audio_cannot_record_another_response(
    monkeypatch: pytest.MonkeyPatch, cancel_at: str
) -> None:
    clock = [10.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    handler = _handler()
    handler.cancel_scope = CancelScope()
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("old")
    other = store.get_or_create_response("new")

    def generate(_text):
        clock[0] = 10.12
        if cancel_at == "generation":
            handler.cancel_scope.cancel()
        elif cancel_at == "discard":
            store.discard_response("old")
        return _waveform()

    def resample(chunk, **kwargs):
        if cancel_at == "resample":
            handler.cancel_scope.cancel()
        return np.asarray(chunk, dtype=np.float32)

    handler.generate_audio = generate  # type: ignore[method-assign]
    monkeypatch.setattr(facebookmms_handler, "librosa", SimpleNamespace(resample=resample))
    output = handler.process(TTSInput(text="hello", response_key="old", speech_stopped_at_s=9.0))
    if cancel_at == "after_first":
        assert next(output).shape == (512,)
        handler.cancel_scope.cancel()
        assert list(output) == []
        assert tracker.tts_ttfa_s == pytest.approx(0.12)
        assert tracker.e2e_s == pytest.approx(1.12)
    elif cancel_at == "discard":
        assert list(output)
        assert store.get_response("old") is None
        assert tracker.tts_ttfa_s is None
        assert tracker.e2e_s is None
    else:
        assert list(output) == []
        assert tracker.e2e_s is None
        assert tracker.tts_ttfa_s == (None if cancel_at == "generation" else pytest.approx(0.12))
    assert other.tts_ttfa_s is None
    assert other.e2e_s is None
