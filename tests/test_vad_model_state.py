from queue import Queue
from threading import Event

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.vad_handler import VADHandler
from speech_to_speech.VAD.vad_iterator import VADIterator
from tests.test_vad_iterator import _FakeVADModel


class _CountingModel(_FakeVADModel):
    def __init__(self, probs: list[float]) -> None:
        super().__init__(probs)
        self.resets = 0

    def reset_states(self) -> None:
        self.resets += 1


@pytest.mark.parametrize("sample_rate,chunk_samples", [(8000, 256), (16000, 512)])
@pytest.mark.parametrize("batched", [False, True])
def test_idle_model_resets_preserve_clock_and_pre_speech_padding(sample_rate, chunk_samples, batched):
    model = _CountingModel([0.1] * 32 + [0.9] * 2 + [0.1] * 3)
    iterator = VADIterator(model, sampling_rate=sample_rate, min_silence_duration_ms=64, speech_pad_ms=32)
    chunk = torch.zeros((1, chunk_samples) if batched else chunk_samples)

    for _ in range(15):
        assert iterator(chunk) is None
    assert model.resets == 1
    assert iterator(chunk) is None
    assert model.resets == 2  # First whole chunk at or beyond 500 ms.
    for _ in range(16):
        assert iterator(chunk) is None
    assert model.resets == 3  # Idle state stays bounded in a long session.
    assert iterator.current_sample == 32 * chunk_samples

    speech = torch.ones_like(chunk)
    assert iterator(speech) is None
    assert iterator(speech) is None
    assert iterator(chunk) is None
    assert iterator(chunk) is None
    output = iterator(chunk)

    assert output is not None
    assert torch.equal(output[0], chunk)  # Padding survives the idle reset.
    assert torch.equal(output[1], speech)
    assert torch.equal(output[2], speech)
    assert len(output) == 6  # Padding, speech, and existing trailing silence.
    assert iterator.current_sample == 37 * chunk_samples
    assert iterator.last_speech_end_sample == 34 * chunk_samples
    assert iterator.last_utterance_active_speech_samples == 2 * chunk_samples
    assert model.resets == 4  # Endpoint resets only the model.


def test_idle_reset_waits_for_consecutive_low_confidence_chunks():
    model = _CountingModel([0.1] * 15 + [0.5] + [0.1] * 16)
    iterator = VADIterator(model, threshold=0.6)
    chunk = torch.zeros(512)

    for _ in range(31):
        assert iterator(chunk) is None
    assert model.resets == 1  # The hysteresis-band chunk breaks the idle run.
    assert iterator(chunk) is None
    assert model.resets == 2


def test_active_speech_and_within_segment_pauses_keep_model_state():
    model = _CountingModel([0.9] * 20 + [0.1] * 16 + [0.5] + [0.1] * 23)
    iterator = VADIterator(model, threshold=0.6, min_silence_duration_ms=700)
    chunk = torch.ones(512)

    for _ in range(37):
        assert iterator(chunk) is None
    assert iterator.triggered
    assert model.resets == 1  # Neither long speech nor its 512-ms pause resets.
    for _ in range(22):
        assert iterator(chunk) is None
    output = iterator(chunk)
    assert output is not None
    assert len(output) == 60
    assert iterator.last_utterance_active_speech_samples == 21 * 512
    assert model.resets == 2


def test_session_reset_clears_idle_reset_progress():
    model = _CountingModel([0.1] * 31)
    iterator = VADIterator(model)
    chunk = torch.zeros(512)
    for _ in range(15):
        iterator(chunk)
    iterator.reset_states()
    for _ in range(15):
        iterator(chunk)
    assert model.resets == 2
    assert iterator.current_sample == 15 * 512
    iterator(chunk)
    assert model.resets == 3


class _NoiseSensitiveModel:
    """Deterministic state poisoning; real Silero evidence is a separate replay."""

    def reset_states(self) -> None:
        self.polluted = False

    def __call__(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        level = x.abs().mean().item()
        if 0.04 < level < 0.06:
            self.polluted = True
        probability = 0.9 if level > 0.1 and not self.polluted else 0.1
        return torch.tensor(probability)


def test_handler_emits_short_speech_after_noise_and_preserves_later_turns(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: (_NoiseSensitiveModel(), None))
    listen = Event()
    listen.set()
    events = Queue()
    handler = VADHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "should_listen": listen,
            "speculative_turns": SpeculativeTurnTracker(),
            "smart_turn": False,
            "text_output_queue": events,
        },
    )

    def feed(value, count):
        pcm = np.full(512, int(value * 32767), dtype=np.int16).tobytes()
        return [output for _ in range(count) for output in handler.process(pcm)]

    assert feed(0.05, 64) == []
    assert feed(0, 16) == []
    assert events.empty()
    for turn in range(2):
        start_samples = handler._total_samples
        assert feed(0.2, 12) == []  # 384 ms, unchanged default minimum.
        output = feed(0, 3)
        assert len(output) == 1
        assert output[0].mode == "final"
        assert len(output[0].audio) == 15 * 512 + 480
        started, stopped = events.get_nowait(), events.get_nowait()
        assert isinstance(started, SpeechStartedEvent)
        assert isinstance(stopped, SpeechStoppedEvent)
        assert started.audio_start_ms == int(start_samples / 16) - 30
        assert stopped.audio_end_ms == int((start_samples + 15 * 512) / 16)
        assert started.turn_id == stopped.turn_id == output[0].turn_id
        assert started.turn_id == f"turn_{turn + 1}"
        assert started.turn_revision == stopped.turn_revision == output[0].turn_revision
        assert stopped.duration_s == len(output[0].audio) / 16000
        assert events.empty()
        assert handler.iterator.current_sample == handler._total_samples
        assert handler.iterator.last_speech_end_sample == start_samples + 12 * 512
        assert handler.speculative_turns.wait_for_gate(
            output[0].turn_id, output[0].turn_revision, hold_for_grace=False, commit=True
        )
        assert feed(0, 16) == []

    handler.on_session_end()
    assert handler.iterator.current_sample == handler._total_samples == 0
