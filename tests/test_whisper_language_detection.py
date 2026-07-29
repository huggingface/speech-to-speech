"""Regression tests for Whisper STT language detection.

The handler used to infer the detected language by slicing the *second* generated token
(``pred_ids[0, 1]``), which assumes ``generate()`` echoes the forced decoder prefix
``<|startoftranscript|><|de|><|transcribe|>``. ``generate()`` excludes the decoder input ids
from its output, and the language token is one of them, so index 1 is an ordinary text
token; slicing it produced a word fragment that was never a valid language, which pushed
every call down the "unsupported language" path and re-generated without a forced language,
replacing a correct transcription with a mis-detected one (German returned as Italian).

Two properties are pinned here:

* the reported language is the language actually transcribed, obtained from Whisper's
  ``detect_language()`` rather than guessed from a token position, and
* a correct transcription is never discarded because its language is outside
  ``SUPPORTED_LANGUAGES`` -- that allowlist controls the sticky fallback only.

The model is faked, so no download is needed.
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
# language token. This is what made the position-based parse silently wrong.
TEXT_TOKENS = {2626: " Der", 3021: " heiß", 4488: " Ball"}

SENTINEL_ENCODER_OUTPUTS = object()


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
    """A batch of one sequence, indexable like the tensor ``generate()`` returns."""

    def __init__(self, token_ids, text: str) -> None:
        self._ids = np.array([list(token_ids)])
        self.text = text

    def __getitem__(self, key):
        return self._ids[key]


class FakeModel:
    """Records ``generate()`` calls and optionally supports ``detect_language()``.

    ``detect_language_result`` is ``None`` to emulate a transformers version that does not
    expose the API at all, or an exception instance to emulate detection failing.
    """

    def __init__(self, responses, detect_language_result=None):
        self._responses = list(responses)
        self._detect_language_result = detect_language_result
        self.calls: list[dict] = []
        self.detect_language_calls = 0
        self.encoder_calls = 0

        if detect_language_result is None:
            # Emulate a model without the detect_language API at all.
            self.detect_language = None  # type: ignore[assignment]

    def get_encoder(self):
        def encoder(input_features):
            self.encoder_calls += 1
            return SENTINEL_ENCODER_OUTPUTS

        return encoder

    def detect_language(self, encoder_outputs=None):  # type: ignore[no-redef]
        self.detect_language_calls += 1
        assert encoder_outputs is SENTINEL_ENCODER_OUTPUTS, "encoder output should be reused"
        if isinstance(self._detect_language_result, Exception):
            raise self._detect_language_result
        return np.array([SPECIAL_TOKENS[self._detect_language_result]])

    def generate(self, input_features, **gen_kwargs):
        self.calls.append(gen_kwargs)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def make_handler(*, start_language, last_language, gen_kwargs, responses, detect_language_result=None):
    """Build a handler without loading any real model."""
    handler = object.__new__(WhisperSTTHandler)
    handler.processor = FakeProcessor()
    handler.model = FakeModel(responses, detect_language_result)
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


# --- forced language ----------------------------------------------------------------------


def test_forced_language_survives_missing_decoder_prefix():
    """The German -> Italian bug: no prefix + forced language must not re-generate."""
    german = FakeSequences([2626, 3021, 4488], "Der blaue Ball liegt auf dem Tisch.")
    handler = make_handler(
        start_language="de",
        last_language="de",
        gen_kwargs={"language": "de", "task": "transcribe"},
        responses=[german],
        detect_language_result="<|it|>",
    )

    result = run(handler)

    assert result.text == "Der blaue Ball liegt auf dem Tisch."
    assert result.language_code == "de"
    # One generate() call, and no detection at all: the request is authoritative.
    assert len(handler.model.calls) == 1
    assert handler.model.detect_language_calls == 0


# --- auto mode: detection via detect_language() -------------------------------------------


def test_auto_mode_uses_detect_language_and_forces_the_result():
    """`generate()` never exposes the language token, so detection must be explicit."""
    german = FakeSequences([2626, 3021, 4488], "Wie heisse ich?")
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[german],
        detect_language_result="<|de|>",
    )

    result = run(handler)

    assert result.text == "Wie heisse ich?"
    assert result.language_code == "de-auto"
    assert handler.last_language == "de"
    assert handler.model.detect_language_calls == 1
    assert len(handler.model.calls) == 1
    # The detected language is forced, so generate() does not detect a second time.
    assert handler.model.calls[0]["language"] == "de"


def test_auto_mode_reuses_the_encoder_output_instead_of_re_encoding():
    """Detection and generation share one encoder pass, so detection is not extra cost."""
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[FakeSequences([2626], "Hallo.")],
        detect_language_result="<|de|>",
    )

    run(handler)

    assert handler.model.encoder_calls == 1
    assert handler.model.calls[0]["encoder_outputs"] is SENTINEL_ENCODER_OUTPUTS


def test_detection_does_not_mutate_shared_gen_kwargs():
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[FakeSequences([2626], "Hallo.")],
        detect_language_result="<|de|>",
    )

    run(handler)

    assert handler.gen_kwargs == {"task": "transcribe"}


# --- auto mode: a real but unsupported language must be preserved -------------------------


def test_unsupported_detected_language_is_reported_not_retranscribed():
    """Russian audio must not be re-transcribed as German just because German was last."""
    russian = FakeSequences([2626], "Privet, kak dela?")
    handler = make_handler(
        start_language="auto",
        last_language="de",
        gen_kwargs={"task": "transcribe"},
        responses=[russian],
        detect_language_result="<|ru|>",
    )

    result = run(handler)

    assert result.text == "Privet, kak dela?"
    assert result.language_code == "ru-auto"
    # Exactly one generate(), forced to the detected language -- no destructive retry.
    assert len(handler.model.calls) == 1
    assert handler.model.calls[0]["language"] == "ru"
    # An unsupported code must not become the sticky fallback for later turns.
    assert handler.last_language == "de"


def test_unsupported_detected_language_without_any_fallback():
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[FakeSequences([2626], "Privet.")],
        detect_language_result="<|ru|>",
    )

    result = run(handler)

    assert result.language_code == "ru-auto"
    assert handler.last_language is None


# --- auto mode when detect_language() is unavailable or fails -----------------------------


def test_falls_back_to_prefix_token_when_detect_language_is_unavailable():
    sequence = FakeSequences(
        [SPECIAL_TOKENS["<|startoftranscript|>"], SPECIAL_TOKENS["<|de|>"], SPECIAL_TOKENS["<|transcribe|>"], 2626],
        "Der blaue Ball liegt auf dem Tisch.",
    )
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[sequence],
        detect_language_result=None,
    )

    result = run(handler)

    assert result.language_code == "de-auto"
    assert handler.last_language == "de"
    assert len(handler.model.calls) == 1


def test_prefix_token_is_found_when_not_at_index_one():
    """The token's position is not fixed, so the fallback must scan rather than index."""
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
        detect_language_result=None,
    )

    assert run(handler).language_code == "de-auto"


def test_no_detection_and_no_prefix_falls_back_to_last_language():
    handler = make_handler(
        start_language="auto",
        last_language="de",
        gen_kwargs={"task": "transcribe"},
        responses=[FakeSequences([2626, 3021], "Ich heisse Max.")],
        detect_language_result=None,
    )

    result = run(handler)

    assert result.text == "Ich heisse Max."
    assert result.language_code == "de-auto"
    assert len(handler.model.calls) == 1


def test_no_detection_and_no_fallback_defaults_to_english_without_regenerating():
    """The old code re-ran generate() with language=None, doubling cost for the same result."""
    handler = make_handler(
        start_language="auto",
        last_language=None,
        gen_kwargs={"task": "transcribe"},
        responses=[FakeSequences([2626, 3021], "Hello there.")],
        detect_language_result=None,
    )

    result = run(handler)

    assert result.text == "Hello there."
    assert result.language_code == "en-auto"
    assert len(handler.model.calls) == 1


def test_detect_language_failure_is_survivable():
    handler = make_handler(
        start_language="auto",
        last_language="de",
        gen_kwargs={"task": "transcribe"},
        responses=[FakeSequences([2626], "Ich heisse Max.")],
        detect_language_result=RuntimeError("no kernel"),
    )

    result = run(handler)

    assert result.text == "Ich heisse Max."
    assert result.language_code == "de-auto"
    assert len(handler.model.calls) == 1
    # Detection failed before producing a usable encoder output, so none is passed on.
    assert "encoder_outputs" not in handler.model.calls[0]


def test_language_code_has_no_auto_suffix_when_start_language_is_not_auto():
    handler = make_handler(
        start_language=None,
        last_language=None,
        gen_kwargs={},
        responses=[FakeSequences([2626], "Hello.")],
        detect_language_result="<|en|>",
    )

    assert run(handler).language_code == "en"


# --- helper-level behaviour ---------------------------------------------------------------


def test_prefix_scan_returns_none_for_plain_text_tokens():
    handler = make_handler(start_language=None, last_language=None, gen_kwargs={}, responses=[])

    assert handler._language_from_prefix(FakeSequences([2626, 3021, 4488], "")) is None


def test_prefix_scan_handles_numpy_rows():
    handler = make_handler(start_language=None, last_language=None, gen_kwargs={}, responses=[])
    pred_ids = np.array([[SPECIAL_TOKENS["<|startoftranscript|>"], SPECIAL_TOKENS["<|it|>"], 2626]])

    assert handler._language_from_prefix(pred_ids) == "it"


def test_prefix_scan_ignores_tokens_past_the_prefix_window():
    handler = make_handler(start_language=None, last_language=None, gen_kwargs={}, responses=[])
    pred_ids = [[1, 2, 3, 4, SPECIAL_TOKENS["<|de|>"]]]

    assert handler._language_from_prefix(pred_ids) is None


@pytest.mark.parametrize("forced", [None, "", "auto"])
def test_forced_language_treats_absent_and_auto_as_unforced(forced):
    gen_kwargs = {} if forced is None else {"language": forced}
    handler = make_handler(start_language=None, last_language=None, gen_kwargs=gen_kwargs, responses=[])

    assert handler._forced_language() is None


def test_forced_language_reads_gen_kwargs():
    handler = make_handler(start_language="de", last_language="de", gen_kwargs={"language": "de"}, responses=[])

    assert handler._forced_language() == "de"
