"""COMMITMENT: independent PCM streams yield ordered partials and explicit finals.

WHO BREAKS: applications owning their own VAD, transport and turn boundary.
SEAM: public stream API -> real lifecycle/admission logic -> a fake Qwen model.
SPEC: packages/qwen-vllm-stt/README.md.
DOES NOT PROVE: recognition accuracy, CUDA compatibility or serving capacity.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Event
from typing import Any

import numpy as np
import pytest

from hf_s2s_qwen_vllm import (
    QwenVllmAdmissionError,
    QwenVllmEngine,
    QwenVllmSettings,
    QwenVllmUnhealthyError,
)


@dataclass
class _State:
    text: str = ""
    language: str = "English"


class _FakeQwen:
    def __init__(self) -> None:
        self.states: list[_State] = []
        self.audio: list[np.ndarray] = []
        self.active = 0
        self.peak_active = 0
        self.block_next = False
        self.entered = Event()
        self.overlap = Event()
        self.release = Event()

    def init_streaming_state(self, **kwargs: Any) -> _State:
        state = _State(language=kwargs["language"])
        self.states.append(state)
        return state

    def streaming_transcribe(self, samples: np.ndarray, state: _State) -> None:
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        if self.active > 1:
            self.overlap.set()
        try:
            if self.block_next:
                self.block_next = False
                self.entered.set()
                if not self.release.wait(5):
                    raise RuntimeError("test did not release blocked inference")
            self.audio.append(samples.copy())
            state.text += f"{len(samples)} "
        finally:
            self.active -= 1

    def finish_streaming_transcribe(self, state: _State) -> None:
        pass


def _settings(**changes: Any) -> QwenVllmSettings:
    values: dict[str, Any] = dict(
        model="Qwen/Qwen3-ASR-0.6B",
        language="English",
        gpu_memory_utilization=0.8,
        max_new_tokens=32,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=0.5,
        stream_step_ms=250,
        max_segment_sec=30,
        inference_timeout_s=8,
        inference_rt_margin=3,
        max_concurrency=1,
        max_streams=64,
    )
    values.update(changes)
    return QwenVllmSettings(**values)


def _engine(model: _FakeQwen, **changes: Any) -> QwenVllmEngine:
    return QwenVllmEngine(_settings(**changes), model_factory=lambda **_: model)


def _pcm(samples: int, value: int = 0) -> bytes:
    return np.full(samples, value, dtype="<i2").tobytes()


async def _entered(model: _FakeQwen) -> None:
    assert await asyncio.to_thread(model.entered.wait, 5), "inference did not start"


async def test_streams_and_turns_keep_separate_state_and_commit_the_tail() -> None:
    model = _FakeQwen()
    engine = _engine(model)
    first, second = await engine.new_stream(), await engine.new_stream()
    try:
        assert [(u.text, u.final) for u in await first.push(_pcm(4000, 16384))] == [("4000 ", False)]
        assert await first.push(_pcm(100, -16384)) == []
        assert [(u.text, u.final) for u in await second.push(_pcm(5000))] == [("5000 ", False)]
        updates = await first.commit()
        assert [(u.text, u.final, u.forced) for u in updates] == [
            ("4000 100 ", False, False),
            ("4000 100 ", True, False),
        ]
        assert updates[-1].language == "English"
        assert np.allclose(model.audio[-1], -0.5)
        assert (await second.commit())[-1].text == "5000 "
        assert (await first.push(_pcm(4000)))[0].text == "4000 "
    finally:
        await first.close()
        await second.close()
    with pytest.raises(RuntimeError, match="closed"):
        await first.commit()


@pytest.mark.parametrize("step_ms", [50, 250])
async def test_segment_cap_splits_frames_without_losing_samples(step_ms: int) -> None:
    model = _FakeQwen()
    stream = await _engine(model, max_segment_sec=0.1, stream_step_ms=step_ms).new_stream()
    warmup_calls = len(model.audio)
    try:
        await stream.push(_pcm(800))
        updates = await stream.push(_pcm(1280))
        forced = [u for u in updates if u.final]
        assert len(forced) == 1 and forced[0].forced
        assert sum(int(part) for part in forced[0].text.split()) == 1600
        final = (await stream.commit())[-1]
        assert final.final and not final.forced
        assert final.text == "480 "
        assert sum(len(audio) for audio in model.audio[warmup_calls:]) == 2080
        # An explicit commit still receives a final after an exact cap boundary.
        await stream.push(_pcm(1600))
        assert (await stream.commit())[-1].final
    finally:
        await stream.close()


async def test_capacity_rejects_only_new_streams_and_close_releases_once() -> None:
    model = _FakeQwen()
    engine = _engine(model, max_streams=2)
    first, second = await engine.new_stream(), await engine.new_stream()
    model.block_next = True
    running = asyncio.create_task(first.push(_pcm(4000)))
    await _entered(model)
    waiting = asyncio.create_task(second.push(_pcm(4000)))
    try:
        with pytest.raises(QwenVllmAdmissionError, match="capacity"):
            await engine.new_stream()
        await asyncio.sleep(0)
        assert not await asyncio.to_thread(model.overlap.wait, 0.25)
        assert not waiting.done()
    finally:
        model.release.set()
        await asyncio.gather(running, waiting)
        await first.close()
        await second.close()
    assert model.peak_active == 1
    await first.close()
    third = await engine.new_stream()
    assert (await third.push(_pcm(4000)))[0].text == "4000 "
    await third.close()


async def test_cancelled_stream_cannot_reuse_state_and_inference_slot_drains() -> None:
    model = _FakeQwen()
    engine = _engine(model)
    first, second = await engine.new_stream(), await engine.new_stream()
    model.block_next = True
    running = asyncio.create_task(first.push(_pcm(4000)))
    await _entered(model)
    running.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await running
        with pytest.raises(RuntimeError, match="closed"):
            await first.push(b"")
        waiting = asyncio.create_task(second.push(_pcm(4000)))
        await asyncio.sleep(0)
        assert not await asyncio.to_thread(model.overlap.wait, 0.25)
        assert not waiting.done()
    finally:
        model.release.set()
        await first.close()
    assert (await waiting)[0].text == "4000 "
    assert model.peak_active == 1
    await second.close()


@pytest.mark.parametrize("reset", [False, True])
async def test_close_during_inference_discards_late_output(reset: bool) -> None:
    model = _FakeQwen()
    engine = _engine(model, max_streams=1)
    stream = await engine.new_stream()
    if reset:
        await stream.commit()
        original = model.init_streaming_state

        def blocked_state(**kwargs: Any) -> _State:
            model.entered.set()
            if not model.release.wait(5):
                raise RuntimeError("test did not release state initialization")
            return original(**kwargs)

        model.init_streaming_state = blocked_state
    else:
        model.block_next = True
    previous_audio = len(model.audio)
    running = asyncio.create_task(stream.push(_pcm(4000)))
    await _entered(model)
    closing = asyncio.create_task(stream.close())
    try:
        await asyncio.sleep(0)
        with pytest.raises(QwenVllmAdmissionError, match="capacity"):
            await asyncio.wait_for(engine.new_stream(), 1)
    finally:
        model.release.set()
        await closing
    with pytest.raises(RuntimeError, match="closed"):
        await running
    assert len(model.audio) == previous_audio + (0 if reset else 1)
    recovered = await engine.new_stream()
    await recovered.close()


async def test_cancelled_startup_waiter_does_not_load_a_second_model() -> None:
    model = _FakeQwen()
    entered, release = Event(), Event()
    loads = []

    def factory(**kwargs: Any) -> _FakeQwen:
        loads.append(kwargs)
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test did not release model loading")
        return model

    engine = QwenVllmEngine(_settings(), model_factory=factory)
    first = asyncio.create_task(engine.start())
    assert await asyncio.to_thread(entered.wait, 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(engine.new_stream())
    release.set()
    stream = await second
    assert len(loads) == 1
    assert len(model.audio) == 1  # one real warm-up, shared by both callers
    await stream.close()


@pytest.mark.parametrize("existing_stream", [False, True])
async def test_queued_deadline_does_not_poison_the_running_model(existing_stream: bool) -> None:
    model = _FakeQwen()
    engine = _engine(model, inference_timeout_s=0.5)
    stream = await engine.new_stream()
    queued_stream = await engine.new_stream() if existing_stream else None
    model.block_next = True
    running = asyncio.create_task(stream.push(_pcm(16000)))
    await _entered(model)
    try:
        with pytest.raises(QwenVllmAdmissionError, match="waiting for inference"):
            queued = queued_stream.push(_pcm(4000)) if queued_stream else engine.new_stream()
            await asyncio.wait_for(queued, 2)
        assert engine.health() == (True, "ok")
        if queued_stream:
            with pytest.raises(RuntimeError, match="closed"):
                await queued_stream.push(b"")
    finally:
        model.release.set()
        await running
        await stream.close()
        if queued_stream:
            await queued_stream.close()
    recovered = await engine.new_stream()
    await recovered.close()


@pytest.mark.parametrize("cancel_call", [False, True])
async def test_running_deadline_marks_engine_unhealthy(cancel_call: bool) -> None:
    model = _FakeQwen()
    engine = _engine(model, inference_timeout_s=0.5, inference_rt_margin=0)
    stream = await engine.new_stream()
    model.block_next = True
    running = asyncio.create_task(stream.push(_pcm(4000)))
    await _entered(model)
    try:
        if cancel_call:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            async with asyncio.timeout(2):
                while engine.health()[0]:
                    await asyncio.sleep(0.01)
        else:
            with pytest.raises(QwenVllmUnhealthyError, match="streaming_transcribe_timeout"):
                await running
        assert engine.health() == (False, "streaming_transcribe_timeout")
        with pytest.raises(QwenVllmUnhealthyError):
            await engine.new_stream()
    finally:
        model.release.set()
        await stream.close()


async def test_failed_stream_initialization_releases_capacity() -> None:
    model = _FakeQwen()
    engine = _engine(model, max_streams=1)
    await engine.start()
    original = model.init_streaming_state

    def fail_once(**kwargs: Any) -> _State:
        model.init_streaming_state = original
        raise ValueError("bad stream state")

    model.init_streaming_state = fail_once
    with pytest.raises(ValueError, match="bad stream state"):
        await engine.new_stream()
    stream = await engine.new_stream()
    await stream.close()


async def test_failed_startup_is_reported_as_unhealthy_without_retrying() -> None:
    def failed_model(**_: Any) -> None:
        raise RuntimeError("model loading failed")

    engine = QwenVllmEngine(_settings(), model_factory=failed_model)
    with pytest.raises(RuntimeError, match="model loading failed"):
        await engine.start()
    assert engine.health() == (False, "startup_failed")
    with pytest.raises(QwenVllmUnhealthyError, match="startup_failed"):
        await engine.new_stream()


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_rate": 8000},
        {"max_concurrency": 0},
        {"max_streams": 0.5},
        {"inference_timeout_s": float("nan")},
        {"max_segment_sec": float("inf")},
    ],
)
def test_invalid_settings_fail_before_model_loading(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _settings(**changes)


async def test_invalid_frames_are_rejected_without_consuming_the_stream() -> None:
    stream = await _engine(_FakeQwen()).new_stream()
    try:
        with pytest.raises(ValueError, match="complete samples"):
            await stream.push(b"\x01")
        with pytest.raises(ValueError, match="segment cap"):
            await stream.push(_pcm(16000 * 31))
        assert (await stream.push(_pcm(4000)))[0].text == "4000 "
    finally:
        await stream.close()
