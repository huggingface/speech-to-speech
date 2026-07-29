"""Regression tests for Whisper STT language detection.

The handler used to infer the detected language by slicing the *second* generated token
(``pred_ids[0, 1]``), which assumes ``generate()`` echoes the forced decoder prefix
``<|startoftranscript|><|de|><|transcribe|>``. Recent ``transformers`` releases strip that
prefix, so index 1 is an ordinary text token; slicing it produced a word fragment that was
never a valid language, which pushed every call down the "unsupported language" path and
re-generated without a forced language, replacing a correct transcription with a
mis-detected one (German returned as Italian).

These tests drive the handler with a fake processor/model so they cover both the legacy
prefix-echoing behaviour and the current prefix-stripping behaviour without downloading a
model.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.STT.whisper_stt_handler import WhisperSTTHandler

# A small stand-in for the Whisper vocabulary. Ids are arbitrary but stable.
SPECIAL_TOKENS = {
    "<|startoftranscript|>": 50258,
    "<|startofprev|>": 50361,
    "<|transcribe|>": 50360,
    "<|notimestamps|>": 50364,
    "<|endoftext|>": 50257,
    "<|de|>": 50261,
    "<|it|>": 50274,
    "<|en|>": 50259,
    "<|ru|>": 50263,
}

_ID_TO_SPECIAL = {token_id: token for token, token_id in SPECIAL_TOKENS.items()}

# Ordinary text tokens, so that decoding a *text* id yields a word piece rather than a
# language token. This is what makes the position-based parse silently wrong.
TEXT_TOKENS = {2626: " Der", 3021: " heiß", 4488: " Ball"}


class FakeTokenizer:
    all_special_tokens = list(SPECIAL_TOKENS)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return SPECIAL_TOKENS.get(token)

    def decode(self, token_id) -> str:
        token_id = int(token_id)
        if token_id in _ID_TO_SPECIAL:
            return _ID_TO_SPECIAL[token_id]
        return TEXT_TOKENS.get(token_id, "<unk>")


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def batch_decode(self, pred_ids, skip_special_tokens=True, decode_with_timestamps=False):
        # The text carried by a sequence is whatever the fake model attached to it.
        return [getattr(pred_ids, "text", "")]


class FakeSequences:
    """A batch of one sequence, indexable like the tensor ``generate()`` returns.

    Supports both ``pred_ids[0]`` (row) and ``pred_ids[0, 1]`` (scalar) so the fake behaves
    like a real tensor for the position-based parse as well as the token scan.
    """

    def __init__(self, token_ids, text: str) -> None:
        self._ids = np.array([list(token_ids)])
        self.text = text

    def __getitem__(self, key):
        return self._ids[key]


class FakeModel:
    """Records every ``generate()`` call and returns scripted sequences."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, input_features, **gen_kwargs):
        self.calls.append(gen_kwargs)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def make_handler(*, start_language, last_language, gen_kwargs, responses):
    """Build a handler without loading any real model."""
    handler = object.__new__(WhisperSTTHandler)
    handler.processor = FakeProcessor()
    handler.model = FakeModel(responses)
    handler.start_language = start_language
    handler.last_language = last_language
    handler.gen_kwargs = dict(gen_kwargs)
    handler._language_token_id_map = None
    handler.device = "cpu"
    handler.torch_dtype = None
    # prepare_model_inputs is bypassed; the fake model ignores its input entirely.
    handler.prepare_model_inputs = lambda audio: audio  # type: ignore[method-assign]
    return handler


def vad_audio():
    return VADAudio(audio=np.zeros(16000, dtype=np.float32), turn_id="turn_1", turn_revision=0)


def run(handler):
    outputs = list(handler.process(vad_audio()))
    assert len(outputs) == 1
    return outputs[0]


# --- current transformers: no forced decoder prefix in the output -------------------------


def test_forced_language_survives_missing_decoder_prefix():
    """The German -> Italian bug: no prefix + forced language must not re-generate."""
    german = FakeSequences([2626, 3021, 4488], "Der blaue Ball liegt auf dem Tisch.")
    handler = make_handler(
        start_language="de",
        last_language="de",
        gen_kwargs={"language": "de", "task": "transcribe"},
        responses=[german],
    )

    result = run(handler)

    assert result.text == "Der blaue Ball liegt auf dem Tisch."
    assert result.language_code == "de"
    # Exactly one generate() call: the first pass is authoritative when a language is forced.
    assert len(handler.model.calls) == 1


def test_auto_mode_without_prefix_keeps_first_pass_and_defaults_language():
    """Auto mode, no prefix, no prior language: keep the transcription, don't re-generate."""
    text = FakeSequences([2626, 3021, 4488], "Wie heisse ich?")
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[text],
    )

    result = run(handler)

    assert result.text == "Wie heisse ich?"
    assert result.language_code == "en-auto"
    assert len(handler.model.calls) == 1


def test_no_language_forced_and_no_prefix_does_not_regenerate_with_none():
    """The old code re-ran generate() with language=None, doubling cost for the same result."""
    text = FakeSequences([2626, 3021], "Hello there.")
    handler = make_handler(
        start_language=None,
        last_language=None,
        gen_kwargs={},
        responses=[text],
    )

    run(handler)

    assert len(handler.model.calls) == 1
    assert "language" not in handler.model.calls[0]


# --- legacy transformers: forced decoder prefix present ----------------------------------


def test_language_token_read_from_decoder_prefix():
    sequence = FakeSequences(
        [SPECIAL_TOKENS["<|startoftranscript|>"], SPECIAL_TOKENS["<|de|>"], SPECIAL_TOKENS["<|transcribe|>"], 2626],
        "Der blaue Ball liegt auf dem Tisch.",
    )
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[sequence],
    )

    result = run(handler)

    assert result.language_code == "de-auto"
    assert handler.last_language == "de"
    assert len(handler.model.calls) == 1


def test_language_token_found_when_not_at_index_one():
    """The token's position is not fixed, so detection must scan rather than index."""
    sequence = FakeSequences(
        [
            SPECIAL_TOKENS["<|startofprev|>"],
            SPECIAL_TOKENS["<|startoftranscript|>"],
            SPECIAL_TOKENS["<|de|>"],
            SPECIAL_TOKENS["<|transcribe|>"],
        ],
        "Ich heisse Max.",
    )
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[sequence],
    )

    assert run(handler).language_code == "de-auto"


def test_unsupported_detected_language_retries_with_last_language():
    unsupported = FakeSequences(
        [SPECIAL_TOKENS["<|startoftranscript|>"], SPECIAL_TOKENS["<|ru|>"], SPECIAL_TOKENS["<|transcribe|>"]],
        "mis-detected",
    )
    retried = FakeSequences(
        [SPECIAL_TOKENS["<|startoftranscript|>"], SPECIAL_TOKENS["<|de|>"], SPECIAL_TOKENS["<|transcribe|>"]],
        "Ich habe ein bisschen Angst vor morgen.",
    )
    handler = make_handler(
        start_language="auto",
        last_language="de",
        gen_kwargs={"task": "transcribe"},
        responses=[unsupported, retried],
    )

    result = run(handler)

    assert result.text == "Ich habe ein bisschen Angst vor morgen."
    assert result.language_code == "de-auto"
    assert len(handler.model.calls) == 2
    assert handler.model.calls[1]["language"] == "de"
    # The retry must not mutate the handler's shared gen_kwargs.
    assert "language" not in handler.gen_kwargs


def test_unsupported_detected_language_is_kept_when_there_is_no_fallback():
    """A real but unsupported language beats discarding a correct transcription."""
    russian = FakeSequences(
        [SPECIAL_TOKENS["<|startoftranscript|>"], SPECIAL_TOKENS["<|ru|>"], SPECIAL_TOKENS["<|transcribe|>"]],
        "Privet.",
    )
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[russian],
    )

    result = run(handler)

    assert result.text == "Privet."
    assert result.language_code == "ru-auto"
    assert len(handler.model.calls) == 1
    # An unsupported code must not become the sticky fallback for later turns.
    assert handler.last_language is None


# --- helper-level behaviour ---------------------------------------------------------------


def test_detected_language_returns_none_for_plain_text_tokens():
    handler = make_handler(start_language=None, last_language=None, gen_kwargs={}, responses=[])

    assert handler._detected_language(FakeSequences([2626, 3021, 4488], "")) is None


def test_detected_language_handles_numpy_and_tensor_like_rows():
    handler = make_handler(start_language=None, last_language=None, gen_kwargs={}, responses=[])
    pred_ids = np.array([[SPECIAL_TOKENS["<|startoftranscript|>"], SPECIAL_TOKENS["<|it|>"], 2626]])

    assert handler._detected_language(pred_ids) == "it"


def test_detected_language_ignores_tokens_past_the_prefix_window():
    handler = make_handler(start_language=None, last_language=None, gen_kwargs={}, responses=[])
    pred_ids = [[1, 2, 3, 4, SPECIAL_TOKENS["<|de|>"]]]

    assert handler._detected_language(pred_ids) is None


@pytest.mark.parametrize("forced", [None, "", "auto"])
def test_forced_language_treats_absent_and_auto_as_unforced(forced):
    gen_kwargs = {} if forced is None else {"language": forced}
    handler = make_handler(start_language=None, last_language=None, gen_kwargs=gen_kwargs, responses=[])

    assert handler._forced_language() is None


def test_forced_language_reads_gen_kwargs():
    handler = make_handler(start_language="de", last_language="de", gen_kwargs={"language": "de"}, responses=[])

    assert handler._forced_language() == "de"
