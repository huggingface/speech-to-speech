from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS import kitten_tts_handler as kitten_module
from tests.test_kitten_tts_handler import make_handler


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kitten_module.console, "print", lambda *_a, **_k: None)


def test_first_chunk_leaves_before_the_next_chunk_is_synthesized() -> None:
    handler = make_handler(blocksize=4)
    handler._chunk_text = lambda text: ["First sentence.", "Second sentence."]
    generated = []

    def generate(*, text, voice):
        generated.append(text)
        return np.full(9, 0.1, dtype=np.float32)

    handler.model.generate_single_chunk = generate
    outputs = handler.process(TTSInput(text="First sentence. Second sentence."))
    first = next(outputs)

    assert generated == ["First sentence."]
    assert len(first) == 4
    rest = list(outputs)
    assert generated == ["First sentence.", "Second sentence."]
    # Each native nine-sample chunk resamples to six samples. The remainder
    # carries between chunks, producing three full blocks with no padding gap.
    blocks = [first, *rest]
    assert [len(block) for block in blocks] == [4, 4, 4]
    assert all(block.dtype == np.int16 for block in blocks)
    assert np.all(np.concatenate(blocks) > 0)


@pytest.mark.parametrize("delivered_blocks", [1, 2], ids=["during-chunk", "between-chunks"])
def test_cancellation_stops_before_synthesizing_the_next_chunk(delivered_blocks: int) -> None:
    scope = CancelScope()
    handler = make_handler(cancel_scope=scope, blocksize=4)
    handler._chunk_text = lambda text: ["First.", "Second."]
    generated = []

    def generate(*, text, voice):
        generated.append(text)
        return np.full(12, 0.1, dtype=np.float32)

    handler.model.generate_single_chunk = generate
    outputs = handler.process(TTSInput(text="First. Second."))
    for _ in range(delivered_blocks):
        next(outputs)
    scope.cancel()

    assert list(outputs) == []
    assert generated == ["First."]


def test_chunks_share_partial_blocks_and_only_the_final_tail_is_padded() -> None:
    handler = make_handler(blocksize=4)
    handler._chunk_text = lambda text: ["First.", "Second."]
    handler.model.generate_single_chunk = lambda **_kwargs: np.full(4, 0.1, dtype=np.float32)

    blocks = list(handler.process(TTSInput(text="First. Second.")))

    # Each native four-sample chunk becomes three samples. Six samples fill
    # one block and a two-sample tail, rather than padding each chunk to four.
    assert [len(block) for block in blocks] == [4, 4]
    assert np.all(np.concatenate(blocks)[:6] > 0)
    np.testing.assert_array_equal(blocks[-1][-2:], np.zeros(2, dtype=np.int16))
