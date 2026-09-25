"""`adapt_gen_kwargs` must translate its argument without consuming it.

It maps the pipeline's `return_timestamps` onto faster-whisper's inverse
`without_timestamps`. Popping the source key out of the caller's dict makes the
translation single-use: the second handler built from the same config reads the
default instead of the supplied value, and the setting silently inverts.
"""

from typing import Any

import pytest

faster_whisper = pytest.importorskip("faster_whisper")

from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler  # noqa: E402


@pytest.fixture
def handler() -> FasterWhisperSTTHandler:
    """A handler instance without running `setup`, which loads a model."""
    return object.__new__(FasterWhisperSTTHandler)


@pytest.mark.parametrize(
    "return_timestamps,expected",
    [(True, False), (False, True)],
)
def test_return_timestamps_is_inverted(
    handler: FasterWhisperSTTHandler, return_timestamps: bool, expected: bool
) -> None:
    adapted = handler.adapt_gen_kwargs({"return_timestamps": return_timestamps})

    assert adapted == {"without_timestamps": expected}


def test_default_is_timestamps_on(handler: FasterWhisperSTTHandler) -> None:
    assert handler.adapt_gen_kwargs({}) == {"without_timestamps": False}


def test_caller_dict_is_not_consumed(handler: FasterWhisperSTTHandler) -> None:
    config: dict[str, Any] = {"return_timestamps": False, "beam_size": 5}

    handler.adapt_gen_kwargs(config)

    assert config == {"return_timestamps": False, "beam_size": 5}


def test_one_config_reused_gives_both_handlers_the_same_setting(
    handler: FasterWhisperSTTHandler,
) -> None:
    """The sharpest consequence: the second read used to flip the setting."""
    config: dict[str, Any] = {"return_timestamps": False}

    first = handler.adapt_gen_kwargs(config)
    second = handler.adapt_gen_kwargs(config)

    assert first == second == {"without_timestamps": True}


def test_shared_setup_default_is_not_polluted(handler: FasterWhisperSTTHandler) -> None:
    default_gen_kwargs = FasterWhisperSTTHandler.setup.__defaults__[-1]
    assert default_gen_kwargs == {}, "precondition: the default starts empty"

    handler.adapt_gen_kwargs(default_gen_kwargs)

    assert default_gen_kwargs == {}


def test_other_kwargs_pass_through(handler: FasterWhisperSTTHandler) -> None:
    adapted = handler.adapt_gen_kwargs({"beam_size": 5, "language": "fr"})

    assert adapted == {"beam_size": 5, "language": "fr", "without_timestamps": False}
