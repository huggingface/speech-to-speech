from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.TTS import irodori_tts_handler as irodori_module
from speech_to_speech.TTS.irodori_tts_handler import DEFAULT_CAPTION, IrodoriTTSHandler


class FakeRuntime:
    """Stands in for irodori_tts.inference_runtime.InferenceRuntime; requests are plain dicts."""

    def __init__(self, synthesize: Callable[[dict[str, Any]], torch.Tensor] | None = None, sample_rate: int = 48000):
        self.requests: list[dict[str, Any]] = []
        self.sample_rate = sample_rate
        self._synthesize = synthesize or (lambda _request: torch.zeros(1, 768))

    def synthesize(self, request: dict[str, Any]) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(audio=self._synthesize(request), sample_rate=self.sample_rate)


def install_fake_irodori(monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime) -> list[dict[str, Any]]:
    """Register a fake irodori_tts package and return the runtime keys it is asked to load."""
    keys: list[dict[str, Any]] = []

    def from_key(key: dict[str, Any]) -> FakeRuntime:
        keys.append(key)
        return runtime

    module = SimpleNamespace(
        InferenceRuntime=SimpleNamespace(from_key=from_key),
        RuntimeKey=lambda **kwargs: kwargs,
        SamplingRequest=lambda **kwargs: kwargs,
        download_hf_checkpoint=lambda source: f"/hub/{source}/model.safetensors",
    )
    monkeypatch.setitem(sys.modules, "irodori_tts", SimpleNamespace(inference_runtime=module))
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", module)
    return keys


def make_handler(
    runtime: FakeRuntime, *, blocksize: int = 512, cancel_scope: CancelScope | None = None
) -> IrodoriTTSHandler:
    handler = IrodoriTTSHandler.__new__(IrodoriTTSHandler)
    handler.runtime = runtime
    handler._request = partial(dict, caption=DEFAULT_CAPTION, ref_wav=None, no_ref=True, num_steps=None)
    handler.blocksize = blocksize
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = None
    return handler


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(irodori_module.console, "print", lambda *_a, **_k: None)


@pytest.mark.parametrize(
    ("setup_kwargs", "message"),
    [
        ({"blocksize": 0}, "blocksize must be positive"),
        ({"num_steps": 0}, "num_steps must be positive"),
        ({"ref_audio": "/missing/voice.wav"}, "reference audio not found"),
    ],
)
def test_setup_rejects_invalid_configuration_before_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, setup_kwargs: dict[str, Any], message: str
) -> None:
    keys = install_fake_irodori(monkeypatch, FakeRuntime())
    handler = IrodoriTTSHandler.__new__(IrodoriTTSHandler)

    with pytest.raises(ValueError, match=message):
        handler.setup(Event(), **setup_kwargs)

    assert keys == []


def test_setup_without_the_package_points_at_the_install_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "irodori_tts", None)
    handler = IrodoriTTSHandler.__new__(IrodoriTTSHandler)

    with pytest.raises(ImportError, match="README"):
        handler.setup(Event())


def test_default_voice_is_one_caption_for_the_warmup_and_every_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime()
    keys = install_fake_irodori(monkeypatch, runtime)
    handler = IrodoriTTSHandler.__new__(IrodoriTTSHandler)

    # The parsed CLI config is exactly what the pipeline passes to setup().
    handler.setup(Event(), **parse_arguments(["--tts", "irodori", "--irodori_device", "cpu"]).tts_backend.config)
    list(handler.process(TTSInput(text="おはよう。", language_code="ja")))
    list(handler.process(TTSInput(text="元気？", language_code="ja")))

    assert keys == [
        {
            "checkpoint": "/hub/Aratako/Irodori-TTS-v4.1-Small-MF/model.safetensors",
            "model_device": "cpu",
            "codec_device": "cpu",
        }
    ]
    assert [request["text"] for request in runtime.requests] == ["こんにちは。", "おはよう。", "元気？"]
    for request in runtime.requests:
        assert request["caption"] == DEFAULT_CAPTION
        assert request["ref_wav"] is None
        assert request["no_ref"] is True
        assert request["num_steps"] is None
    # One seed for every sentence: with a caption alone, the sampling noise still shapes the voice.
    assert len({request["seed"] for request in runtime.requests}) == 1
    assert handler.language_capabilities.languages == frozenset({"ja"})
    assert handler.language_capabilities.supports_language_selection is False


def test_reference_audio_clones_without_the_default_caption(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = FakeRuntime()
    keys = install_fake_irodori(monkeypatch, runtime)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    reference = tmp_path / "voice.wav"
    reference.touch()
    handler = IrodoriTTSHandler.__new__(IrodoriTTSHandler)

    handler.setup(Event(), model_name=str(checkpoint), device="cpu", ref_audio=str(reference), num_steps=8)
    list(handler.process(TTSInput(text="おはよう。")))

    assert keys[0]["checkpoint"] == str(checkpoint)
    assert runtime.requests[-1] == {
        "text": "おはよう。",
        "caption": None,
        "ref_wav": str(reference),
        "no_ref": False,
        "num_steps": 8,
        "seed": runtime.requests[0]["seed"],
    }


def test_caption_can_style_a_cloned_voice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = FakeRuntime()
    install_fake_irodori(monkeypatch, runtime)
    reference = tmp_path / "voice.wav"
    reference.touch()
    handler = IrodoriTTSHandler.__new__(IrodoriTTSHandler)

    handler.setup(Event(), device="cpu", caption="ささやき声", ref_audio=str(reference))

    assert runtime.requests[0]["caption"] == "ささやき声"
    assert runtime.requests[0]["ref_wav"] == str(reference)


def test_process_resamples_clips_and_pads_16khz_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    resample_calls = []

    def fake_resample(audio: np.ndarray, up: int, down: int) -> np.ndarray:
        resample_calls.append((audio, up, down))
        return np.array([2.0, -2.0, 0.5], dtype=np.float32)

    monkeypatch.setattr(irodori_module.scipy.signal, "resample_poly", fake_resample)
    handler = make_handler(FakeRuntime(lambda _request: torch.zeros(1, 9)), blocksize=4)

    chunks = list(handler.process(TTSInput(text="おはよう。")))

    assert resample_calls[0][0].shape == (9,)
    assert (resample_calls[0][1], resample_calls[0][2]) == (1, 3)
    assert len(chunks) == 1
    assert chunks[0].dtype == np.int16
    np.testing.assert_array_equal(chunks[0], np.array([32767, -32768, 16384, 0], dtype=np.int16))


@pytest.mark.parametrize("text", ["", "  \n"])
def test_process_skips_blank_text(text: str) -> None:
    runtime = FakeRuntime()
    handler = make_handler(runtime)

    assert list(handler.process(TTSInput(text=text))) == []
    assert runtime.requests == []


def test_process_drops_audio_when_cancelled_during_blocking_synthesis() -> None:
    cancel_scope = CancelScope()

    def synthesize(_request: dict[str, Any]) -> torch.Tensor:
        cancel_scope.cancel()
        return torch.zeros(1, 768)

    handler = make_handler(FakeRuntime(synthesize), cancel_scope=cancel_scope)

    assert list(handler.process(TTSInput(text="おはよう。"))) == []


def test_process_stops_emitting_blocks_after_cancellation() -> None:
    cancel_scope = CancelScope()
    runtime = FakeRuntime(lambda _request: torch.zeros(1, 5), sample_rate=16000)
    handler = make_handler(runtime, blocksize=4, cancel_scope=cancel_scope)
    chunks = handler.process(TTSInput(text="おはよう。"))

    assert len(next(chunks)) == 4
    cancel_scope.cancel()
    with pytest.raises(StopIteration):
        next(chunks)


def test_records_ttfa_before_resample_and_e2e_on_first_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])

    def synthesize(_request: dict[str, Any]) -> torch.Tensor:
        clock[0] = 100.12
        return torch.linspace(-0.2, 0.2, 768).reshape(1, -1)

    def slow_resample(audio: np.ndarray, up: int, down: int) -> np.ndarray:
        clock[0] = 100.40
        return np.asarray(audio, dtype=np.float32)

    monkeypatch.setattr(irodori_module.scipy.signal, "resample_poly", slow_resample)
    handler = make_handler(FakeRuntime(synthesize))
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = list(
        handler.process(
            TTSInput(
                text="おはよう。",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=99.0,
            )
        )
    )

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.12)
    assert tracker.e2e_s == pytest.approx(1.40)
