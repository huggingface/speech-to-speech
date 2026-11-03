from threading import Event

import numpy as np

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.model_speech_handler import ModelSpeechHandler


def _handler() -> ModelSpeechHandler:
    handler = ModelSpeechHandler.__new__(ModelSpeechHandler)
    handler.setup(Event(), blocksize=4, cancel_scope=CancelScope())
    return handler


def test_model_speech_plays_without_gaps_and_pads_only_the_response_tail():
    handler = _handler()
    speech = np.arange(1, 11, dtype="<i2")

    played = [
        *handler.process(TTSInput(text="Hello there", response_key="r1", cancel_generation=0)),
        *handler.process(TTSInput(text="", audio=speech[:6], response_key="r1", cancel_generation=0)),
        *handler.process(TTSInput(text="", audio=speech[6:], response_key="r1", cancel_generation=0)),
    ]
    tail = list(handler.process(EndOfResponse(response_key="r1", cancel_generation=0)))

    assert np.concatenate(played).tolist() == list(range(1, 9))
    assert tail[0].tolist() == [9, 10, 0, 0]
    assert tail[1] == AUDIO_RESPONSE_DONE


def test_model_speech_from_an_interrupted_response_is_dropped():
    handler = _handler()
    list(handler.process(TTSInput(text="", audio=np.ones(2, dtype="<i2"), response_key="r1", cancel_generation=0)))
    handler.cancel_scope.cancel()

    stale = list(
        handler.process(TTSInput(text="", audio=np.ones(8, dtype="<i2"), response_key="r1", cancel_generation=0))
    )
    done = list(handler.process(EndOfResponse(response_key="r1", cancel_generation=0)))

    assert stale == []
    assert done == [AUDIO_RESPONSE_DONE]
