from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.TTS import kitten_tts_handler as kitten_module
from speech_to_speech.TTS import supertonic_tts_handler as supertonic_module
from tests.test_kitten_tts_handler import make_handler as make_kitten
from tests.test_supertonic_tts_handler import make_handler as make_supertonic


def _input() -> TTSInput:
    return TTSInput(
        text="First sentence. Second sentence.",
        response_key="resp-1",
        turn_id="turn_1",
        turn_revision=0,
        speech_stopped_at_s=99.0,
    )


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kitten_module.console, "print", lambda *_a, **_k: None)
    monkeypatch.setattr(supertonic_module.console, "print", lambda *_a, **_k: None)


def test_kitten_streams_each_chunk_and_records_first_chunk_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(kitten_module, "perf_counter", lambda: clock[0])
    handler = make_kitten(blocksize=4)
    handler._chunk_text = lambda text: ["First sentence.", "Second sentence."]
    generated = []

    def generate(*, text, voice):
        clock[0] += 0.25
        generated.append(text)
        return np.ones(9, dtype=np.float32) * 0.1

    handler.model.generate_single_chunk = generate
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = handler.process(_input())
    first = next(outputs)

    # The first block leaves before the second chunk is synthesized.
    assert generated == ["First sentence."]
    assert len(first) == 4
    assert tracker.tts_ttfa_s == pytest.approx(0.25)
    assert tracker.e2e_s == pytest.approx(1.25)
    rest = list(outputs)
    assert generated == ["First sentence.", "Second sentence."]
    # 9 native samples become 6 per chunk; 12 carried samples fill 3 blocks with no padding gap.
    assert [len(block) for block in [first, *rest]] == [4, 4, 4]
    assert tracker.tts_ttfa_s == pytest.approx(0.25)


def test_kitten_cancellation_between_chunks_stops_synthesis() -> None:
    scope = CancelScope()
    handler = make_kitten(cancel_scope=scope, blocksize=4)
    handler._chunk_text = lambda text: ["First.", "Second."]
    generated = []

    def generate(*, text, voice):
        generated.append(text)
        return np.ones(12, dtype=np.float32) * 0.1

    handler.model.generate_single_chunk = generate
    outputs = handler.process(TTSInput(text="First. Second."))
    next(outputs)
    scope.cancel()

    assert list(outputs) == []
    assert generated == ["First."]


def test_supertonic_records_ttfa_before_resample_and_e2e_on_first_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(supertonic_module, "perf_counter", lambda: clock[0])

    def synthesize(**_kwargs):
        clock[0] = 100.2
        return np.zeros((1, 441), dtype=np.float32), np.array([0.01])

    def slow_resample(audio, up, down):
        clock[0] = 100.5
        return np.zeros(160, dtype=np.float32)

    monkeypatch.setattr(supertonic_module.scipy.signal, "resample_poly", slow_resample)
    handler = make_supertonic(synthesize)
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = list(handler.process(_input()))

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.2)
    assert tracker.e2e_s == pytest.approx(1.5)


def test_phonon_records_e2e_on_first_delivered_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech_to_speech.TTS import phonon_tts_handler as phonon_module
    from tests.test_phonon_tts_handler import make_handler as make_phonon

    clock = [100.0]
    monkeypatch.setattr(phonon_module, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(phonon_module.console, "print", lambda *_a, **_k: None)

    def samples():
        clock[0] = 100.3
        yield np.full(6, 0.1, dtype=np.float32)
        clock[0] = 100.9
        yield np.full(6, 0.1, dtype=np.float32)

    handler, _stream, _requests = make_phonon(samples(), sample_rate=16000, blocksize=4)
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = list(handler.process(_input()))

    assert [len(block) for block in outputs] == [4, 4, 4]
    assert tracker.tts_ttfa_s == pytest.approx(0.3)
    assert tracker.e2e_s == pytest.approx(1.3)


def test_kitten_trims_leading_silence_but_keeps_a_short_lead_in() -> None:
    rate = kitten_module.KITTEN_SAMPLE_RATE
    audio = np.concatenate([np.zeros(rate // 2, dtype=np.float32), np.full(rate // 10, 0.1, dtype=np.float32)])

    trimmed = kitten_module.trim_leading_silence(audio)

    lead_in = int(kitten_module.LEADING_SILENCE_KEEP_S * rate)
    assert len(trimmed) == rate // 10 + lead_in
    assert np.all(trimmed[:lead_in] == 0) and np.all(trimmed[lead_in:] == pytest.approx(0.1))
    silent = np.zeros(rate // 10, dtype=np.float32)
    assert len(kitten_module.trim_leading_silence(silent)) == len(silent)


def test_kitten_chunks_start_at_speech() -> None:
    handler = make_kitten(blocksize=4)
    handler._chunk_text = lambda text: ["Hi."]
    rate = kitten_module.KITTEN_SAMPLE_RATE
    handler.model.generate_single_chunk = lambda *, text, voice: np.concatenate(
        [np.zeros(rate, dtype=np.float32), np.full(rate // 100, 0.1, dtype=np.float32)]
    )

    (chunk,) = list(handler._generate_chunks("Hi.", "Bruno"))

    assert len(chunk) == rate // 100 + int(kitten_module.LEADING_SILENCE_KEEP_S * rate)


def test_phonon_skips_text_without_words_and_keeps_speaking(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech_to_speech.TTS import phonon_tts_handler as phonon_module
    from tests.test_phonon_tts_handler import make_handler as make_phonon

    monkeypatch.setattr(phonon_module.console, "print", lambda *_a, **_k: None)
    handler, _stream, requests = make_phonon(iter([np.full(8, 0.1, dtype=np.float32)]), sample_rate=16000, blocksize=4)

    assert list(handler.process(TTSInput(text='"'))) == []
    assert requests == []
    assert [len(block) for block in handler.process(TTSInput(text="Nemo."))] == [4, 4]
