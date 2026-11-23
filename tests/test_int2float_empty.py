"""Empty PCM chunks must not crash the VAD conversion helper."""

from queue import Queue
from threading import Event

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import int2float
from speech_to_speech.VAD.vad_handler import VADHandler
from tests.test_vad_iterator import _FakeVADModel


def test_int2float_accepts_empty_input():
    result = int2float(np.array([], dtype=np.int16))
    assert result.dtype == np.float32
    assert result.size == 0


class _RecordingVADModel(_FakeVADModel):
    def __init__(self) -> None:
        super().__init__([])
        self.calls = 0

    def __call__(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        self.calls += 1
        return super().__call__(x, sampling_rate)


def test_vad_handler_ignores_empty_chunk(monkeypatch: pytest.MonkeyPatch):
    model = _RecordingVADModel()
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: (model, None))
    should_listen = Event()
    should_listen.set()
    handler = VADHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "should_listen": should_listen,
            "speculative_turns": SpeculativeTurnTracker(),
            "thresh": 0.6,
            "min_silence_ms": 64,
            "smart_turn": False,
        },
    )

    assert list(handler.process(b"")) == []
    assert model.calls == 0
    assert handler._total_samples == 0
