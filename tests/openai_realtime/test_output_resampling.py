"""Completed WebSocket audio must retain its duration and item identity."""

import base64

import numpy as np
import pytest
from openai.types.realtime import SessionUpdateEvent
from scipy.signal import resample_poly

from speech_to_speech.pipeline.events import AssistantOutputEvent, AssistantResponseDoneEvent, ResponseFailedEvent


@pytest.fixture(autouse=True)
def output_24khz(service, conn_id):
    service.handle_session_update(
        conn_id,
        SessionUpdateEvent(
            type="session.update",
            session={"type": "realtime", "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}},
        ),
    )


def _audio(events):
    return b"".join(base64.b64decode(event.delta) for event in events if event.type == "response.output_audio.delta")


def _reference(samples):
    return np.clip(np.round(resample_poly(samples.astype(np.float64), 3, 2)), -32768, 32767).astype("<i2").tobytes()


@pytest.mark.parametrize("sample_count", [1, 10, 11, 511, 4096])
@pytest.mark.parametrize("terminal", ["response", "ordered_audio"])
def test_completed_audio_matches_whole_signal_reference(service, conn_id, sample_count, terminal):
    samples = np.round(np.cos(np.arange(sample_count) * 2 * np.pi * 997 / 16000) * 12000).astype("<i2")
    events = []
    cuts = sorted({cut for cut in (1, 7, 18, 129, sample_count - 1) if 0 < cut < sample_count})
    for chunk in np.split(samples, cuts):
        events.extend(service.encode_audio_chunk(conn_id, chunk.tobytes(), "response_a"))
    if terminal == "ordered_audio":
        events.extend(service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key="response_a")))
    events.extend(service.finish_response(conn_id, response_key="response_a"))

    assert _audio(events) == _reference(samples)
    assert len(_audio(events)) // 2 == (sample_count * 3 + 1) // 2
    done = [event for event in events if event.type == "response.output_audio.done"]
    assert len(done) == 1
    deltas = [event for event in events if event.type == "response.output_audio.delta"]
    assert deltas
    assert all(
        (event.response_id, event.item_id, event.output_index, event.content_index)
        == (done[0].response_id, done[0].item_id, done[0].output_index, 0)
        for event in deltas
    )
    assert events.index(deltas[-1]) < events.index(done[0])
    assert service.finish_audio_output(conn_id, "response_a") == []
    assert service.finish_response(conn_id, response_key="response_a") == []
    assert service._state(conn_id).output_audio_resampler is None
    assert service._state(conn_id).output_audio_resampler_key is None


def test_tool_boundary_flushes_old_item_and_isolates_next_item(service, conn_id):
    first = np.full(512, 12000, dtype="<i2")
    second = np.zeros(512, dtype="<i2")
    service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="Before tool."))
    first_events = service.encode_audio_chunk(conn_id, first.tobytes())
    tool_events = service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(tools=[{"type": "function_call", "call_id": "c1", "name": "tool", "arguments": "{}"}]),
    )
    assert _audio(first_events + tool_events) == _reference(first)
    tail, done = tool_events[:2]
    assert tail.type == "response.output_audio.delta"
    assert done.type == "response.output_audio.done"
    assert (tail.item_id, tail.output_index, tail.content_index) == (done.item_id, 0, 0)
    assert service._state(conn_id).output_audio_resampler is None

    service.dispatch_pipeline_event(conn_id, AssistantOutputEvent(text="After tool."))
    second_events = service.encode_audio_chunk(conn_id, second.tobytes()) + service.finish_response(conn_id)
    assert _audio(second_events) == _reference(second)
    second_deltas = [event for event in second_events if event.type == "response.output_audio.delta"]
    assert all(event.item_id != done.item_id and event.output_index == 2 for event in second_deltas)
    assert all(event.response_id == done.response_id for event in second_deltas)


@pytest.mark.parametrize("termination", ["cancelled", "failed", "incomplete", "pipeline_failure"])
def test_abandoned_audio_discards_tail_and_next_response_is_clean(service, conn_id, termination):
    service.encode_audio_chunk(conn_id, np.full(512, 12000, dtype="<i2").tobytes())
    if termination == "pipeline_failure":
        events = service.dispatch_pipeline_event(conn_id, ResponseFailedEvent(message="TTS failed"))
        events.extend(service.finish_response(conn_id))
    elif termination == "cancelled":
        events = service.handle_response_cancel(conn_id)
    else:
        events = service.finish_response(conn_id, status=termination)
    assert _audio(events) == b""
    assert service._state(conn_id).output_audio_resampler is None
    assert service._state(conn_id).output_audio_resampler_key is None

    samples = np.zeros(512, dtype="<i2")
    next_events = service.encode_audio_chunk(conn_id, samples.tobytes()) + service.finish_response(conn_id)
    assert _audio(next_events) == _reference(samples)


def test_unrelated_terminal_does_not_flush_active_audio(service, conn_id):
    samples = np.full(512, 12000, dtype="<i2")
    events = service.encode_audio_chunk(conn_id, samples.tobytes(), "active")
    assert service.finish_audio_output(conn_id, "late") == []
    assert service.finish_response(conn_id, response_key="late") == []
    events.extend(service.finish_response(conn_id, response_key="active"))
    assert _audio(events) == _reference(samples)
