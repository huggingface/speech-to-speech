from queue import Queue
from threading import Event

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.vad_handler import VADHandler
from speech_to_speech.VAD.vad_iterator import VADIterator
from tests.test_vad_iterator import _FakeVADModel

CHUNK = np.zeros(512, dtype=np.int16).tobytes()  # 32 ms at 16 kHz; the fake model supplies the probabilities
SILENCE = [0.0] * 4  # more than the 64 ms that ends a segment
CONFIDENT, ORDINARY = 0.95, 0.7


def _handler(monkeypatch: pytest.MonkeyPatch, probs: list[float], **settings) -> VADHandler:
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: (_FakeVADModel(probs), None))
    should_listen = Event()
    should_listen.set()
    return VADHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "should_listen": should_listen,
            "speculative_turns": SpeculativeTurnTracker(),
            "thresh": 0.6,
            "min_silence_ms": 64,
            "speech_pad_ms": 0,
            "smart_turn": False,
            **settings,
        },
    )


def _opened_turns(handler: VADHandler, chunks: int) -> int:
    return sum(isinstance(output, VADAudio) for _ in range(chunks) for output in handler.process(CHUNK))


SCORED = {"min_speech_ms": 800, "min_confident_speech_ms": 128}


@pytest.mark.parametrize(
    ("speech", "opens"),
    [
        ([CONFIDENT] * 4, True),  # 128 ms of confident speech is enough alone
        ([CONFIDENT] * 3, False),  # 96 ms is not
        ([ORDINARY] * 13, False),  # 416 ms of ordinary speech no longer is, with min_speech_ms at 800
        ([ORDINARY] * 25, True),  # 800 ms of ordinary speech still is
        ([CONFIDENT] * 2 + [ORDINARY] * 13, True),  # 64/128 + 416/800 reaches 1
        ([CONFIDENT] * 1 + [ORDINARY] * 6, False),  # 32/128 + 192/800 does not
    ],
)
def test_new_turn_opens_on_confident_and_ordinary_speech_together(monkeypatch, speech, opens) -> None:
    probs = speech + SILENCE
    handler = _handler(monkeypatch, probs, **SCORED)

    assert _opened_turns(handler, len(probs)) == int(opens)


def test_without_the_option_only_active_speech_counts(monkeypatch) -> None:
    probs = [CONFIDENT] * 4 + SILENCE + [ORDINARY] * 13 + SILENCE
    handler = _handler(monkeypatch, probs, min_speech_ms=384)

    assert _opened_turns(handler, 8) == 0  # 128 ms, however confident, is under 384 ms
    assert _opened_turns(handler, 17) == 1


IDLE = {**SCORED, "idle_after_s": 1.0, "idle_min_speech_ms": 1600, "idle_min_confident_speech_ms": 384}


def test_idle_requirements_apply_after_silence(monkeypatch) -> None:
    quiet = [0.0] * 40  # 1.28 s without accepted speech
    probs = quiet + [CONFIDENT] * 4 + SILENCE + quiet + [CONFIDENT] * 12 + SILENCE
    handler = _handler(monkeypatch, probs, **IDLE)

    assert _opened_turns(handler, 48) == 0  # 128 ms confident opens a turn in conversation, not when idle
    assert _opened_turns(handler, 56) == 1  # 384 ms confident does


def test_recent_speech_keeps_the_conversation_requirements(monkeypatch) -> None:
    # 0.64 s apart: past the (shortened) reopen window, so a new turn, but well inside idle_after_s.
    probs = [CONFIDENT] * 4 + SILENCE + [0.0] * 20 + [CONFIDENT] * 4 + SILENCE
    handler = _handler(monkeypatch, probs, **IDLE, speculative_reopen_ms=300, unanswered_reopen_ms=300)

    assert _opened_turns(handler, len(probs)) == 2


def test_continuations_keep_the_plain_rule(monkeypatch) -> None:
    # Inside the reopen window the candidate continues the open turn: 128 ms of speech,
    # however confident, is under min_speech_continuation_ms (192 ms).
    probs = [CONFIDENT] * 4 + SILENCE + [0.0] * 4 + [CONFIDENT] * 4 + SILENCE + [0.0] * 4 + [CONFIDENT] * 6 + SILENCE
    handler = _handler(monkeypatch, probs, **SCORED)

    assert _opened_turns(handler, 8) == 1
    assert _opened_turns(handler, 12) == 0
    assert _opened_turns(handler, 14) == 1


def test_finished_response_restarts_the_idle_clock(monkeypatch) -> None:
    quiet = [0.0] * 40
    probs = quiet + [CONFIDENT] * 4 + SILENCE
    handler = _handler(monkeypatch, probs, **IDLE)
    assert _opened_turns(handler, 40) == 0

    handler.speculative_turns.close(None, None)  # the assistant just finished speaking

    assert _opened_turns(handler, 8) == 1


def test_session_end_restarts_the_idle_clock(monkeypatch) -> None:
    probs = [CONFIDENT] * 4 + SILENCE + [0.0] * 40
    handler = _handler(monkeypatch, probs, **IDLE)
    assert _opened_turns(handler, len(probs)) == 1
    assert handler._last_accept_audio_ms > 0

    handler.on_session_end()

    assert handler._last_accept_audio_ms == 0
    assert handler.speculative_turns.seconds_since_response_closed() is None


@pytest.mark.parametrize(
    "settings",
    [
        {"confident_thresh": 0},
        {"min_confident_speech_ms": -1},
        {"idle_after_s": 60},  # idle settings need the scored rule
        {"min_confident_speech_ms": 128, "vad": "firered", "vad_firered_model_dir": "unused"},
    ],
)
def test_invalid_settings_fail_at_setup(monkeypatch, settings) -> None:
    with pytest.raises(ValueError):
        _handler(monkeypatch, [], **settings)


def test_iterator_reports_confident_speech_per_utterance() -> None:
    iterator = VADIterator(
        _FakeVADModel([CONFIDENT, ORDINARY, CONFIDENT, 0.5] + SILENCE),
        threshold=0.6,
        min_silence_duration_ms=64,
        speech_pad_ms=0,
        confident_threshold=0.9,
    )
    chunk = torch.zeros(512)

    utterance = None
    while utterance is None:
        utterance = iterator(chunk)

    assert iterator.last_utterance_active_speech_samples == 4 * 512  # 0.5 is above thresh - 0.15
    assert iterator.last_utterance_confident_speech_samples == 2 * 512
    assert iterator.confident_speech_samples == 0


@pytest.mark.parametrize("progressive", [False, True])
def test_idle_noise_cannot_open_early_or_emit_progressive_audio(monkeypatch, progressive) -> None:
    probs = [0.0] * 40 + [ORDINARY] * 25 + SILENCE
    handler = _handler(monkeypatch, probs, **IDLE)
    if progressive:
        handler.transcription_window_samples = 512
    outputs = [output for _ in probs for output in handler.process(CHUNK)]
    assert not any(isinstance(output, VADAudio) for output in outputs)
    assert handler._log_speech_starts == 0


@pytest.mark.parametrize("continuation_ms", [0, 800])
def test_equal_continuation_thresholds_keep_plain_admission(monkeypatch, continuation_ms) -> None:
    probs = [CONFIDENT] * 4 + SILENCE + [0.0] * 4 + [CONFIDENT] * 4 + SILENCE
    handler = _handler(monkeypatch, probs, **SCORED, min_speech_continuation_ms=continuation_ms)
    assert _opened_turns(handler, 8) == 1
    assert _opened_turns(handler, 12) == 0


def test_confident_new_turn_opens_before_segment_ends(monkeypatch) -> None:
    handler = _handler(monkeypatch, [CONFIDENT] * 4, **SCORED)
    _opened_turns(handler, 4)
    assert handler._speech_started_emitted


@pytest.mark.parametrize("probability", [ORDINARY, CONFIDENT])
@pytest.mark.parametrize("idle_min_ms, opens", [(0, True), (1600, False)])
def test_zero_minimum_accepts_detected_speech_unless_idle_override_applies(
    monkeypatch, probability, idle_min_ms, opens
) -> None:
    probs = [0.0] * 40 + [probability] + SILENCE
    handler = _handler(
        monkeypatch,
        probs,
        min_speech_ms=0,
        min_confident_speech_ms=128,
        idle_after_s=1.0,
        idle_min_speech_ms=idle_min_ms,
    )
    assert _opened_turns(handler, 40) == 0  # Silence still requires VAD detection.
    assert _opened_turns(handler, 1) == 0
    assert handler._speech_started_emitted == opens
    assert _opened_turns(handler, len(SILENCE)) == int(opens)
