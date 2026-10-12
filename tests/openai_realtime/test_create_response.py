"""Automatic response control through the effective Realtime session config."""

import base64
import io
import wave
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from openai.types.realtime import ResponseCreateEvent

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import (
    AudioInputCompletedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from tests.llm_history import drive_llm
from tests.test_responses_api_language_model import _make_handler


def _update(service, conn_id, **turn_detection):
    event = service.parse_client_event(
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            **turn_detection,
                        }
                    }
                },
            },
        }
    )
    assert event is not None
    assert service.handle_session_update(conn_id, event) is None


@pytest.mark.parametrize("kind", ["server_vad", "semantic_vad", "dict"])
@pytest.mark.parametrize("value", [None, False, True])
def test_flag_reads_supported_configuration(runtime_config, kind, value):
    detection = {"type": "server_vad" if kind == "dict" else kind, "create_response": value}
    if kind == "dict":
        runtime_config.session.audio.input.turn_detection = detection
    else:
        from openai.types.realtime.realtime_audio_input_turn_detection import SemanticVad, ServerVad

        runtime_config.session.audio.input.turn_detection = (
            ServerVad(**detection) if kind == "server_vad" else SemanticVad(**detection)
        )
    assert runtime_config.create_response_enabled is (value is not False)


@pytest.mark.parametrize("interrupt_response", [False, True])
@pytest.mark.parametrize("tracked", [False, True])
def test_disabled_response_retains_transcript_and_accepts_explicit_request(
    service, conn_id, runtime_config, text_prompt_queue, interrupt_response, tracked
):
    _update(service, conn_id, create_response=False, interrupt_response=interrupt_response)
    assert service.build_session_updated(conn_id).session.audio.input.turn_detection.create_response is False
    tracker = SpeculativeTurnTracker() if tracked else None
    service.speculative_turns = tracker
    turn = {}
    if tracker is not None:
        turn_id, revision = tracker.start_turn()
        turn = {"turn_id": turn_id, "turn_revision": revision}
    started = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(**turn))
    if tracker is not None:
        tracker.segment_finalized(1250)
    stopped = service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=1.25, **turn))
    completed = service.dispatch_pipeline_event(
        conn_id, TranscriptionCompletedEvent(transcript="What is the project code word?", **turn)
    )
    events = started + stopped + completed
    assert text_prompt_queue.empty()
    state = service._state(conn_id)
    assert not state.response_pending and not state.in_response
    assert state.pending_response_keys == set()
    assert state.pending_input_terminals == {}
    assert [event.type for event in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert events[-1].item_id == started[0].item_id
    assert events[-1].content_index == 0
    assert events[-1].usage.seconds == 1.25
    user_items = [item for item in runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
    assert [item.content[0].text for item in user_items] == ["What is the project code word?"]
    created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    assert created.type == "response.created"
    request = text_prompt_queue.get_nowait()
    assert request.runtime_config is runtime_config
    assert request.turn_id == turn.get("turn_id")
    assert request.turn_revision == turn.get("turn_revision")
    assert text_prompt_queue.empty()
    service.response.mark_response_created_sent(conn_id, request.response_key)
    done = service.finish_response(conn_id, response_key=request.response_key)
    assert done[-1].type == "response.done"
    assert done[-1].response.status == "completed"
    assert state.response_usage.audio_duration_s == 0
    assert service.total_usage.audio_duration_s == 1.25


@pytest.mark.parametrize("setting", [None, True, False])
@pytest.mark.parametrize("direct_audio", [False, True])
def test_effective_flag_controls_automatic_generation(service, conn_id, text_prompt_queue, setting, direct_audio):
    if setting is not None:
        _update(service, conn_id, create_response=setting)
        # A partial nested update must keep the earlier create_response value.
        _update(service, conn_id, interrupt_response=False)
    event = (
        AudioInputCompletedEvent(audio=np.zeros(1600, dtype=np.float32), audio_duration_s=0.1)
        if direct_audio
        else TranscriptionCompletedEvent(transcript="hello")
    )
    service.dispatch_pipeline_event(conn_id, event)
    assert text_prompt_queue.qsize() == (0 if setting is False else 1)
    if setting is False:
        _update(service, conn_id, create_response=True)
        # Updating the flag does not retroactively schedule a response.
        assert text_prompt_queue.empty()
        service.dispatch_pipeline_event(conn_id, event)
        assert text_prompt_queue.qsize() == 1


@pytest.mark.parametrize("failure", [False, True])
def test_disabled_direct_audio_reaches_real_backend_once_and_survives_response_failure(failure):
    prompts = Queue()
    service = RealtimeService(text_prompt_queue=prompts)
    conn_id = service.register()
    _update(service, conn_id, create_response=False)
    cfg = service._state(conn_id).runtime_config
    handler = _make_handler(stream=False)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        if failure:
            raise RuntimeError("provider failed")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="I heard both.", refusal=None, tool_calls=[]))],
            usage=None,
        )

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=generate)))
    for number in (1, 2):
        turn = {"turn_id": f"turn_{number}", "turn_revision": 0}
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(**turn))
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=0.1, **turn))
        service.dispatch_pipeline_event(
            conn_id,
            AudioInputCompletedEvent(
                audio=np.full(1600, number / 10, dtype=np.float32),
                audio_duration_s=0.1,
                **turn,
            ),
        )
        assert prompts.empty()
    user_items = [item for item in cfg.chat.buffer if getattr(item, "role", None) == "user"]
    assert len(user_items) == 2
    assert all(item.content[0].type == "input_audio" for item in user_items)
    created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    assert created.type == "response.created"
    request = prompts.get_nowait()
    assert request.turn_id == "turn_2"
    outputs = list(drive_llm(handler, request, service=service, conn_id=conn_id))
    assert len(calls) == 1
    audio_inputs = [
        part["input_audio"]
        for item in calls[0]["messages"]
        if item["role"] == "user"
        for part in item["content"]
        if part["type"] == "input_audio"
    ]
    assert len(audio_inputs) == 2
    for number, audio in enumerate(audio_inputs, 1):
        with wave.open(io.BytesIO(base64.b64decode(audio["data"])), "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnframes() == 1600
            assert np.frombuffer(wav.readframes(1600), dtype="<i2")[0] == int(number / 10 * 32767)
    assert outputs[-1].tag == "end_of_response"
    assert bool(outputs[-1].error) is failure
    # Cancellation/failure rolls back generated output, never accepted user audio.
    assert len([item for item in cfg.chat.buffer if getattr(item, "role", None) == "user"]) == 2
    service.unregister(conn_id)


def test_disabled_audio_revision_replaces_input_without_generation(service, conn_id, runtime_config, text_prompt_queue):
    _update(service, conn_id, create_response=False)
    tracker = SpeculativeTurnTracker()
    service.speculative_turns = tracker
    turn_id, revision = tracker.start_turn()
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id=turn_id, turn_revision=revision))
    tracker.segment_finalized(100, output_hold_ms=10000)
    service.dispatch_pipeline_event(
        conn_id, SpeechStoppedEvent(duration_s=0.1, turn_id=turn_id, turn_revision=revision)
    )
    assert (
        service.dispatch_pipeline_event(
            conn_id,
            AudioInputCompletedEvent(
                audio=np.zeros(1600, dtype=np.float32),
                audio_duration_s=0.1,
                turn_id=turn_id,
                turn_revision=revision,
            ),
        )
        == []
    )
    original_id = runtime_config.chat.buffer[-1].id
    candidate = tracker.begin_reopen_candidate(turn_id, revision)
    assert tracker.confirm_reopen_candidate(turn_id, revision, candidate)
    service.dispatch_pipeline_event(
        conn_id, SpeechStartedEvent(turn_id=turn_id, turn_revision=candidate, reopened=True)
    )
    tracker.segment_finalized(200)
    service.dispatch_pipeline_event(
        conn_id, SpeechStoppedEvent(duration_s=0.2, turn_id=turn_id, turn_revision=candidate)
    )
    events = service.dispatch_pipeline_event(
        conn_id,
        AudioInputCompletedEvent(
            audio=np.zeros(3200, dtype=np.float32),
            audio_duration_s=0.2,
            turn_id=turn_id,
            turn_revision=candidate,
        ),
    )
    assert [event.type for event in events] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
    ]
    assert len(runtime_config.chat.buffer) == 1
    assert runtime_config.chat.buffer[-1].id == original_id
    assert service._state(conn_id).response_usage.audio_duration_s == 0.2
    assert text_prompt_queue.empty()
    # A late completion for the superseded revision cannot replace the input.
    assert (
        service.dispatch_pipeline_event(
            conn_id,
            AudioInputCompletedEvent(
                audio=np.zeros(1600, dtype=np.float32),
                audio_duration_s=0.1,
                turn_id=turn_id,
                turn_revision=revision,
            ),
        )
        == []
    )
    request_event = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    assert request_event.type == "response.created"
    request = text_prompt_queue.get_nowait()
    assert (request.turn_id, request.turn_revision) == (turn_id, candidate)


def test_disabling_automatic_responses_does_not_cancel_existing_work(service, conn_id, text_prompt_queue):
    service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="first"))
    first = text_prompt_queue.get_nowait()
    _update(service, conn_id, create_response=False)
    assert first.response_key in service._state(conn_id).pending_response_keys
    service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="second"))
    assert text_prompt_queue.empty()
    assert first.response_key in service._state(conn_id).pending_response_keys


@pytest.mark.parametrize("revised", [False, True])
def test_audio_arriving_during_manual_generation_remains_available_for_next_request(revised):
    prompts = Queue()
    service = RealtimeService(text_prompt_queue=prompts)
    conn_id = service.register()
    _update(service, conn_id, create_response=False, interrupt_response=False)
    handler = _make_handler(stream=False)
    handler.audio_history_turns = 0 if revised else 1
    calls = []

    def supply_audio(number):
        turn = {
            "turn_id": "turn_1" if revised else f"turn_{number}",
            "turn_revision": number - 1 if revised else 0,
        }
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(reopened=revised and number > 1, **turn))
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=0.1, **turn))
        service.dispatch_pipeline_event(
            conn_id,
            AudioInputCompletedEvent(
                audio=np.full(1600, number / 10, dtype=np.float32),
                audio_duration_s=0.1,
                **turn,
            ),
        )

    def generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # These inputs arrive after the provider snapshot, while its reply
            # is active. Neither input may be compacted before being consumed.
            supply_audio(2)
            if not revised:
                supply_audio(3)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer.", refusal=None, tool_calls=[]))],
            usage=None,
        )

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=generate)))
    supply_audio(1)
    for _ in range(2):
        assert (
            service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create")).type
            == "response.created"
        )
        request = prompts.get_nowait()
        list(drive_llm(handler, request, service=service, conn_id=conn_id))
        service.response.mark_response_created_sent(conn_id, request.response_key)
        service.finish_response(conn_id, response_key=request.response_key)
    assert len(calls) == 2
    heard = []
    for item in calls[1]["messages"]:
        if item["role"] != "user" or isinstance(item["content"], str):
            continue
        for part in item["content"]:
            if part["type"] != "input_audio":
                continue
            with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"])), "rb") as wav:
                heard.append(int(np.frombuffer(wav.readframes(1600), dtype="<i2")[0]))
    assert int(0.2 * 32767) in heard
    if not revised:
        assert int(0.3 * 32767) in heard
    assert prompts.empty()
    service.unregister(conn_id)
