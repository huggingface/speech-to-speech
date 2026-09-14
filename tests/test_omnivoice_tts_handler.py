from __future__ import annotations

import sys
from collections.abc import Callable
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS import omnivoice_handler as omnivoice_module
from speech_to_speech.TTS.omnivoice_handler import OmniVoiceTTSHandler


class FakeModel:
    sampling_rate = 24000

    def __init__(
        self,
        generate: Callable[..., list[np.ndarray]] | None = None,
    ) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.create_prompt_calls: list[dict[str, Any]] = []
        self._generate = generate or (lambda **_kwargs: [np.zeros(768, dtype=np.float32)])

    def create_voice_clone_prompt(self, **kwargs: Any) -> object:
        self.create_prompt_calls.append(kwargs)
        return object()

    def generate(self, **kwargs: Any) -> list[np.ndarray]:
        self.generate_calls.append(kwargs)
        return self._generate(**kwargs)


def make_handler(
    model: FakeModel,
    *,
    voice_clone_prompt: object | None = None,
    instruct: str | None = None,
    language: str | None = None,
    blocksize: int = 512,
    cancel_scope: CancelScope | None = None,
) -> OmniVoiceTTSHandler:
    handler = OmniVoiceTTSHandler.__new__(OmniVoiceTTSHandler)
    handler.model = model
    handler.voice_clone_prompt = voice_clone_prompt
    handler.instruct = instruct
    handler.language = language
    handler.num_steps = 32
    handler.speed = 1.0
    handler.blocksize = blocksize
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    return handler


def test_setup_precomputes_and_reuses_raw_voice_clone_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    load_calls = []

    class FakeOmniVoice:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> FakeModel:
            load_calls.append((args, kwargs))
            return model

    monkeypatch.setitem(
        sys.modules,
        "omnivoice",
        SimpleNamespace(OmniVoice=FakeOmniVoice, VoiceClonePrompt=SimpleNamespace(load=lambda _path: object())),
    )
    handler = OmniVoiceTTSHandler.__new__(OmniVoiceTTSHandler)

    handler.setup(
        Event(),
        model_name="local/omnivoice",
        device="mps",
        dtype="bfloat16",
        ref_audio="voice.wav",
        ref_text="Reference transcript.",
    )
    list(handler.process(TTSInput(text="First chunk.", language_code="en")))
    list(handler.process(TTSInput(text="Second chunk.", language_code="en")))

    assert load_calls == [
        (("local/omnivoice",), {"device_map": "mps", "dtype": torch.bfloat16}),
    ]
    assert model.create_prompt_calls == [{"ref_audio": "voice.wav", "ref_text": "Reference transcript."}]
    assert [call["voice_clone_prompt"] for call in model.generate_calls] == [
        handler.voice_clone_prompt,
        handler.voice_clone_prompt,
    ]


def test_setup_loads_saved_voice_clone_prompt_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    saved_prompt = object()
    prompt_loads = []

    class FakeOmniVoice:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> FakeModel:
            return model

    class FakeVoiceClonePrompt:
        @classmethod
        def load(cls, path: str) -> object:
            prompt_loads.append(path)
            return saved_prompt

    monkeypatch.setitem(
        sys.modules,
        "omnivoice",
        SimpleNamespace(OmniVoice=FakeOmniVoice, VoiceClonePrompt=FakeVoiceClonePrompt),
    )
    handler = OmniVoiceTTSHandler.__new__(OmniVoiceTTSHandler)

    handler.setup(Event(), voice_clone_prompt="saved-voice.pt")
    list(handler.process(TTSInput(text="Hello.")))

    assert prompt_loads == ["saved-voice.pt"]
    assert model.create_prompt_calls == []
    assert model.generate_calls[0]["voice_clone_prompt"] is saved_prompt


@pytest.mark.parametrize(
    ("voice_clone_prompt", "instruct", "expected_key"),
    [
        (object(), None, "voice_clone_prompt"),
        (None, "female, low pitch", "instruct"),
        (None, None, None),
    ],
)
def test_process_selects_clone_design_or_auto_mode(
    voice_clone_prompt: object | None,
    instruct: str | None,
    expected_key: str | None,
) -> None:
    model = FakeModel()
    handler = make_handler(model, voice_clone_prompt=voice_clone_prompt, instruct=instruct)

    list(handler.process(TTSInput(text="Hello.", language_code="fr")))

    call = model.generate_calls[0]
    assert call["text"] == "Hello."
    assert call["language"] == "fr"
    assert call["num_step"] == 32
    assert call["speed"] == 1.0
    assert ("voice_clone_prompt" in call, "instruct" in call) == {
        "voice_clone_prompt": (True, False),
        "instruct": (False, True),
        None: (False, False),
    }[expected_key]


def test_process_resamples_clips_and_pads_16khz_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(lambda **_kwargs: [np.zeros(8, dtype=np.float32)])
    resample_calls = []

    def fake_resample(audio: np.ndarray, up: int, down: int) -> np.ndarray:
        resample_calls.append((audio, up, down))
        return np.array([2.0, -2.0, 0.5], dtype=np.float32)

    monkeypatch.setattr(omnivoice_module.scipy.signal, "resample_poly", fake_resample)
    handler = make_handler(model, blocksize=4)

    chunks = list(handler.process(TTSInput(text="Hello.")))

    assert (resample_calls[0][1], resample_calls[0][2]) == (2, 3)
    assert len(chunks) == 1
    assert chunks[0].dtype == np.int16
    np.testing.assert_array_equal(chunks[0], np.array([32767, -32768, 16384, 0], dtype=np.int16))


def test_process_drops_audio_when_cancelled_during_blocking_generation() -> None:
    cancel_scope = CancelScope()

    def generate(**_kwargs: Any) -> list[np.ndarray]:
        cancel_scope.cancel()
        return [np.zeros(768, dtype=np.float32)]

    handler = make_handler(FakeModel(generate), cancel_scope=cancel_scope)

    assert list(handler.process(TTSInput(text="Hello."))) == []


def test_process_stops_emitting_blocks_after_cancellation() -> None:
    cancel_scope = CancelScope()
    model = FakeModel(lambda **_kwargs: [np.zeros(5, dtype=np.float32)])
    model.sampling_rate = 16000
    handler = make_handler(model, blocksize=4, cancel_scope=cancel_scope)
    chunks = handler.process(TTSInput(text="Hello."))

    assert len(next(chunks)) == 4
    cancel_scope.cancel()
    with pytest.raises(StopIteration):
        next(chunks)


def test_stale_keyed_terminal_becomes_cleanup_after_lm_tts_handoff() -> None:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    handler = OmniVoiceTTSHandler.__new__(OmniVoiceTTSHandler)
    handler.speculative_turns = tracker
    terminal = EndOfResponse(
        response_key="response_1",
        turn_id="turn_1",
        turn_revision=0,
        cancel_generation=7,
    )
    tracker.observe("turn_1", 1)

    outputs = list(handler.process(terminal))
    queued = handler.output_for_queue(outputs[0], terminal)

    assert outputs == [AUDIO_RESPONSE_DONE]
    assert terminal.cleanup_only is True
    assert isinstance(queued, AudioOutput)
    assert queued.response_key == "response_1"
    assert queued.cancel_generation == 7
    assert queued.cleanup_only is True
