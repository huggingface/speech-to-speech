"""Actual committed-input gates and the opt-in, content-free websocket extension."""

import asyncio
import json
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from openai.types.realtime import ConversationItemCreateEvent, ConversationItemDeleteEvent, ResponseCreateEvent
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime import user_input_observation as observation
from speech_to_speech.api.openai_realtime.pipeline_unit import SessionState
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.transports import SessionTransport, WebSocketTransport
from speech_to_speech.api.openai_realtime.websocket_router import (
    _dispatch_client_event,
    _send_user_input_observations,
    create_app,
)
from speech_to_speech.LLM.chat import ChatItemError
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage
from speech_to_speech.pipeline.events import (
    AudioInputCompletedEvent,
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from tests.openai_realtime.conftest import make_pipeline_unit
from tests.turns import reopen


@pytest.fixture
def observed():
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    conn_id = service.register(observe_user_inputs=True)
    started = service.drain_user_input_observations(conn_id)
    assert len(started) == 1 and isinstance(started[0], observation.UserInputObservationStarted)
    yield service, conn_id
    service.unregister(conn_id)


def _user(parts=None):
    return {
        "type": "message",
        "role": "user",
        "content": parts or [{"type": "input_text", "text": "Synthetic input"}],
    }


def _add(service, conn_id, item=None):
    return service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent.model_validate({"type": "conversation.item.create", "item": item or _user()}),
    )


def _accepted(service, conn_id):
    return [e for e in service.drain_user_input_observations(conn_id) if isinstance(e, observation.UserInputAccepted)]


def _manual(service, conn_id):
    service._state(conn_id).runtime_config.session.audio.input.turn_detection = {
        "type": "server_vad",
        "create_response": False,
    }


def _speech(service, conn_id, text="Synthetic input", *, turn_id="turn_1", revision=0, failed=False):
    events = service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id=turn_id, turn_revision=revision))
    events += service.dispatch_pipeline_event(
        conn_id, SpeechStoppedEvent(turn_id=turn_id, turn_revision=revision, duration_s=0.1)
    )
    events += service.dispatch_pipeline_event(
        conn_id,
        TranscriptionFailedEvent(turn_id=turn_id, turn_revision=revision, message="Synthetic failure")
        if failed
        else TranscriptionCompletedEvent(turn_id=turn_id, turn_revision=revision, transcript=text),
    )
    return events


def test_default_registration_has_no_extension_or_public_configuration():
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        _add(service, conn_id)
        assert service.drain_user_input_observations(conn_id) == []
        assert "user_input" not in service.build_session_created(conn_id).model_dump_json()
        assert "_user_input_observations" not in service._state(conn_id).model_dump()
    finally:
        service.unregister(conn_id)


@pytest.mark.parametrize(
    ("parts", "kind"),
    [
        ([{"type": "input_text", "text": "Synthetic"}], "text"),
        ([{"type": "input_image", "image_url": "data:image/png;base64,c3ludGhldGlj"}], "image"),
        ([{"type": "input_audio", "audio": "c3ludGhldGlj"}], "audio"),
        (
            [
                {"type": "input_text", "text": "Synthetic"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
            ],
            "mixed",
        ),
    ],
)
def test_actual_applied_user_modalities_are_payload_free(observed, parts, kind):
    service, cid = observed
    item = {**_user(parts), "id": "client-content-sentinel"}
    events = _add(service, cid, item)
    assert events[0].type == "conversation.item.created"
    accepted = _accepted(service, cid)
    assert len(accepted) == 1
    event = accepted[0]
    assert event.input_kind == kind and event.sequence == 1 and event.session_id == cid
    assert set(event.model_dump()) == {
        "type",
        "version",
        "event_id",
        "session_id",
        "timestamp_s",
        "sequence",
        "input_kind",
        "dropped_events",
    }
    payload = event.model_dump_json()
    assert "Synthetic" not in payload and "sentinel" not in payload and "base64" not in payload
    assert len(payload) < 512
    assert service.text_prompt_queue.empty()


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_user_text_is_not_a_meaningful_input(observed, text):
    service, cid = observed
    _add(service, cid, _user([{"type": "input_text", "text": text}]))
    assert _accepted(service, cid) == []


@pytest.mark.parametrize(
    "item",
    [
        {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "Synthetic system"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Synthetic assistant"}]},
        {"type": "function_call", "call_id": "synthetic-call", "name": "synthetic", "arguments": "{}"},
    ],
)
def test_non_user_context_is_not_an_input(observed, item):
    service, cid = observed
    assert _add(service, cid, item)[0].type == "conversation.item.created"
    if item["type"] == "function_call":
        assert (
            _add(service, cid, {"type": "function_call_output", "call_id": "synthetic-call", "output": "Synthetic"})[
                0
            ].type
            == "conversation.item.created"
        )
    assert _accepted(service, cid) == []


def test_deferred_input_observes_actual_application_and_not_acknowledgement_replay(observed):
    service, cid = observed
    state = service._state(cid)
    state.in_response = True
    assert _add(service, cid) == []
    assert _accepted(service, cid) == []
    state.in_response = False
    assert service.conversation.flush_deferred_items(cid, defer_acknowledgements=True) == []
    assert len(_accepted(service, cid)) == 1
    assert len(service.conversation.flush_pending_item_acks(cid)) == 1
    assert _accepted(service, cid) == []


@pytest.mark.parametrize("out_of_band", [False, True])
def test_inline_input_uses_actual_default_conversation_application(observed, out_of_band):
    service, cid = observed
    response = {"input": [_user(), _user([{"type": "input_image", "image_url": "data:image/png;base64,AA=="}])]}
    if out_of_band:
        response["conversation"] = "none"
    created = service.handle_response_create(
        cid, ResponseCreateEvent.model_validate({"type": "response.create", "response": response})
    )
    assert created.type == "response.created"
    events = _accepted(service, cid)
    assert [e.input_kind for e in events] == ([] if out_of_band else ["text", "image"])
    assert [e.sequence for e in events] == ([] if out_of_band else [1, 2])


def test_invalid_inline_batch_does_not_observe_preflight_copy(observed):
    service, cid = observed
    result = service.handle_response_create(
        cid,
        ResponseCreateEvent.model_validate(
            {"type": "response.create", "response": {"input": [_user(), _user([{"type": "input_text", "text": ""}])]}}
        ),
    )
    assert result.type == "error"
    assert not service._state(cid).runtime_config.chat.buffer
    assert _accepted(service, cid) == []


def test_partial_inline_actual_failure_keeps_only_real_prior_acceptance(observed, monkeypatch):
    service, cid = observed
    original = service.conversation._append_item
    calls = []

    def apply(conn_id, item):
        calls.append(item)
        if len(calls) == 2:
            raise ChatItemError("Synthetic actual application failure")
        original(conn_id, item)

    monkeypatch.setattr(service.conversation, "_append_item", apply)
    with pytest.raises(ChatItemError):
        service.handle_response_create(
            cid,
            ResponseCreateEvent.model_validate({"type": "response.create", "response": {"input": [_user(), _user()]}}),
        )
    assert len(service._state(cid).runtime_config.chat.buffer) == 1
    assert len(_accepted(service, cid)) == 1


def test_response_without_input_has_no_accepted_input(observed):
    service, cid = observed
    assert (
        service.handle_response_create(cid, ResponseCreateEvent.model_validate({"type": "response.create"})).type
        == "response.created"
    )
    assert _accepted(service, cid) == []


@pytest.mark.parametrize("failed,text", [(False, ""), (False, "  "), (True, "Synthetic")])
def test_empty_and_failed_speech_do_not_count_committed_audio_ack(observed, failed, text):
    service, cid = observed
    _manual(service, cid)
    events = _speech(service, cid, text, failed=failed)
    assert any(e.type == "conversation.item.created" for e in events)
    assert _accepted(service, cid) == []


def test_recognized_manual_speech_counts_without_a_response(observed):
    service, cid = observed
    _manual(service, cid)
    _speech(service, cid)
    assert service.text_prompt_queue.empty()
    events = _accepted(service, cid)
    assert len(events) == 1 and events[0].input_kind == "speech"
    service.audio.resolve_input_terminals(cid)
    assert _accepted(service, cid) == []


def test_onset_and_partial_transcript_do_not_claim_a_committed_input(observed):
    service, cid = observed
    service.dispatch_pipeline_event(cid, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    service.dispatch_pipeline_event(
        cid, PartialTranscriptionEvent(turn_id="turn_1", turn_revision=0, delta="Synthetic")
    )
    assert not service._state(cid).runtime_config.chat.buffer
    assert _accepted(service, cid) == []


@pytest.mark.parametrize("withdraw", [False, True])
def test_provisional_reopened_speech_never_counts_superseded_revision(observed, withdraw):
    service, cid = observed
    turns = service.speculative_turns = SpeculativeTurnTracker()
    turn, revision = turns.start_turn()
    turns.segment_finalized(100)
    _speech(service, cid, turn_id=turn, revision=revision)
    assert _accepted(service, cid) == []
    new_revision = reopen(turns, turn, revision)
    service.dispatch_pipeline_event(cid, SpeechStartedEvent(turn_id=turn, turn_revision=new_revision, reopened=True))
    turns.segment_finalized(200)
    service.dispatch_pipeline_event(
        cid,
        TranscriptionCompletedEvent(
            turn_id=turn, turn_revision=new_revision, transcript="" if withdraw else "Synthetic revised"
        ),
    )
    assert _accepted(service, cid) == []
    turns.gate(turn, new_revision, commit=True)
    service.audio.resolve_input_terminals(cid)
    events = _accepted(service, cid)
    assert len(events) == (0 if withdraw else 1)
    service.audio.resolve_input_terminals(cid)
    assert _accepted(service, cid) == []


def test_deleted_provisional_speech_does_not_count(observed):
    service, cid = observed
    turns = service.speculative_turns = SpeculativeTurnTracker()
    turn, revision = turns.start_turn()
    turns.segment_finalized(100)
    _speech(service, cid, turn_id=turn, revision=revision)
    assert _accepted(service, cid) == []
    item_id = service._state(cid).current_input_item_id
    service.handle_conversation_item_delete(
        cid, ConversationItemDeleteEvent(type="conversation.item.delete", item_id=item_id)
    )
    # Deletion itself is deferred while generation is pending. Apply it before commit.
    assert service._state(cid).deferred_items
    service.close_pending_responses(cid)
    service.conversation.flush_deferred_items(cid)
    turns.gate(turn, revision, commit=True)
    service.audio.resolve_input_terminals(cid)
    assert _accepted(service, cid) == []


@pytest.mark.parametrize("lifecycle", [False, True])
@pytest.mark.parametrize("samples", [0, 160])
def test_manual_direct_audio_uses_existing_commit_with_or_without_lifecycle(observed, lifecycle, samples):
    service, cid = observed
    _manual(service, cid)
    if lifecycle:
        service.dispatch_pipeline_event(cid, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
        service.dispatch_pipeline_event(cid, SpeechStoppedEvent(turn_id="turn_1", turn_revision=0, duration_s=0.1))
    service.dispatch_pipeline_event(
        cid,
        AudioInputCompletedEvent(
            audio=np.zeros(samples, dtype=np.float32),
            turn_id="turn_1" if lifecycle else None,
            turn_revision=0 if lifecycle else None,
        ),
    )
    events = _accepted(service, cid)
    assert len(events) == bool(samples)
    if events:
        assert events[0].input_kind == "audio"
    assert service.text_prompt_queue.empty()


def test_direct_audio_waits_for_real_speculative_commit(observed):
    service, cid = observed
    turns = service.speculative_turns = SpeculativeTurnTracker()
    turn, revision = turns.start_turn()
    turns.segment_finalized(100)
    service.dispatch_pipeline_event(cid, SpeechStartedEvent(turn_id=turn, turn_revision=revision))
    service.dispatch_pipeline_event(
        cid, AudioInputCompletedEvent(audio=np.zeros(160, dtype=np.float32), turn_id=turn, turn_revision=revision)
    )
    assert not service.text_prompt_queue.empty()
    assert _accepted(service, cid) == []
    turns.gate(turn, revision, commit=True)
    service.audio.resolve_input_terminals(cid)
    assert [e.input_kind for e in _accepted(service, cid)] == ["audio"]


def test_observation_does_not_create_a_speculative_commit_without_input_lifecycle(observed):
    service, cid = observed
    _manual(service, cid)
    turns = service.speculative_turns = SpeculativeTurnTracker()
    turn, revision = turns.start_turn()
    turns.segment_finalized(100)
    service.dispatch_pipeline_event(
        cid, AudioInputCompletedEvent(audio=np.zeros(160, dtype=np.float32), turn_id=turn, turn_revision=revision)
    )
    assert not turns.is_committed(turn, revision)
    assert _accepted(service, cid) == []
    turns.gate(turn, revision, commit=True)
    assert [e.input_kind for e in _accepted(service, cid)] == ["audio"]


@pytest.mark.parametrize("clock", [None, True, -1, float("nan"), float("inf"), 10**1000])
def test_observation_clock_failure_never_undoes_input_and_sequence_exposes_gap(observed, monkeypatch, clock):
    service, cid = observed
    monkeypatch.setattr(observation, "time", SimpleNamespace(time=lambda: clock))
    assert _add(service, cid)[0].type == "conversation.item.created"
    assert len(service._state(cid).runtime_config.chat.buffer) == 1
    assert _accepted(service, cid) == []
    monkeypatch.setattr(observation, "time", SimpleNamespace(time=lambda: 100.0))
    _add(service, cid)
    event = _accepted(service, cid)[0]
    assert (event.sequence, event.dropped_events, event.timestamp_s) == (2, 1, 100.0)


def test_queue_is_bounded_and_drop_count_reaches_retained_events():
    service = RealtimeService(text_prompt_queue=Queue())
    cid = service.register(observe_user_inputs=True)
    try:
        for _ in range(80):
            _add(service, cid)
        events = service.drain_user_input_observations(cid)
        assert len(events) == observation.MAX_PENDING_INPUT_OBSERVATIONS == 64
        assert events[0].sequence == 0 and events[-1].sequence == 63
        assert all(e.dropped_events == 17 for e in events)
        _add(service, cid)
        event = _accepted(service, cid)[0]
        assert event.sequence == 81 and event.dropped_events == 17
    finally:
        service.unregister(cid)
    assert service.drain_user_input_observations(cid) == []
    new = service.register(observe_user_inputs=True)
    try:
        assert new != cid
        assert service.drain_user_input_observations(new)[0].dropped_events == 0
        _add(service, new)
        assert _accepted(service, new)[0].sequence == 1
    finally:
        service.unregister(new)


class _Transport(SessionTransport):
    kind = "websocket"

    def __init__(self, fail_observations=False):
        self.events = []
        self.fail_observations = fail_observations

    async def send_events(self, events):
        if self.fail_observations and any(isinstance(e, observation.UserInputAccepted) for e in events):
            raise RuntimeError("Synthetic credential/content sentinel must not appear in logs")
        self.events.extend(events)

    async def send_audio_chunk(self, service, session_id, pcm, response_key=None):
        raise AssertionError("No audio is sent in this test")

    def discard_pending_audio(self):
        pass

    async def close(self):
        pass


async def test_observation_send_failure_keeps_applied_input_and_ordinary_ack(observed, caplog):
    service, cid = observed
    transport = _Transport(fail_observations=True)
    unit = make_pipeline_unit(service=service)
    unit.session = SessionState(session_id=cid, transport=transport)
    await _dispatch_client_event(unit, cid, {"type": "conversation.item.create", "item": _user()}, transport)
    assert [e.type for e in transport.events] == ["conversation.item.created"]
    assert len(service._state(cid).runtime_config.chat.buffer) == 1
    assert "credential/content sentinel" not in caplog.text
    transport.fail_observations = False
    await _dispatch_client_event(unit, cid, {"type": "conversation.item.create", "item": _user()}, transport)
    assert transport.events[-1].sequence == 2 and transport.events[-1].dropped_events == 1


async def test_released_session_observations_never_leak_to_reused_transport(observed):
    service, cid = observed
    transport = _Transport()
    unit = make_pipeline_unit(service=service)
    unit.session = SessionState(session_id="other-session", transport=transport)
    _add(service, cid)
    await _send_user_input_observations(unit, cid, transport)
    assert transport.events == []
    assert len(_accepted(service, cid)) == 1


async def test_cancelled_observation_send_preserves_input_and_exposes_loss(observed):
    service, cid = observed
    started = asyncio.Event()

    class Blocked(_Transport):
        async def send_events(self, events):
            started.set()
            await asyncio.Event().wait()

    transport = Blocked()
    unit = make_pipeline_unit(service=service)
    unit.session = SessionState(session_id=cid, transport=transport)
    _add(service, cid)
    task = asyncio.create_task(_send_user_input_observations(unit, cid, transport))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(service._state(cid).runtime_config.chat.buffer) == 1
    _add(service, cid)
    event = _accepted(service, cid)[0]
    assert event.sequence == 2 and event.dropped_events == 1


@pytest.mark.parametrize("header", [None, "0", "2", "1,1"])
def test_unnegotiated_websocket_retains_existing_frames(header, monkeypatch):
    unit = make_pipeline_unit()
    app = create_app([unit], Event())
    send_attempts = []
    original_send = WebSocketTransport.send_events

    async def recorded_send(transport, events):
        send_attempts.append([event.type for event in events])
        await original_send(transport, events)

    monkeypatch.setattr(WebSocketTransport, "send_events", recorded_send)
    headers = {observation.USER_INPUT_OBSERVATION_HEADER: header} if header else {}
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=headers) as ws:
            assert ws.receive_json()["type"] == "session.created"
            assert send_attempts == []  # Keep the existing direct initial-send path.
            ws.send_json({"type": "conversation.item.create", "item": _user()})
            assert ws.receive_json()["type"] == "conversation.item.created"
            ws.send_json(
                {"type": "session.update", "session": {"type": "realtime", "metadata": {"observe_user_inputs": True}}}
            )
            assert ws.receive_json()["type"] == "session.updated"
            ws.send_json({"type": "conversation.item.create", "item": _user()})
            assert ws.receive_json()["type"] == "conversation.item.created"
            ws.send_json({"type": "session.update", "session": {"type": "realtime"}})
            assert ws.receive_json()["type"] == "session.updated"
            unit.output_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=unit.session.session_id))


def test_opted_websocket_delivers_capability_and_committed_input_in_wire_order():
    unit = make_pipeline_unit()
    app = create_app([unit], Event())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers={observation.USER_INPUT_OBSERVATION_HEADER: "1"}) as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"
            capability = ws.receive_json()
            assert capability["type"] == "speech_to_speech.user_input.observation_started"
            assert capability["version"] == 1 and capability["sequence"] == 0
            assert capability["session_id"] == created["session"]["id"]
            ws.send_json({"type": "conversation.item.create", "item": _user()})
            assert ws.receive_json()["type"] == "conversation.item.created"
            accepted = ws.receive_json()
            assert accepted["type"] == "speech_to_speech.user_input.accepted"
            assert (accepted["sequence"], accepted["input_kind"], accepted["dropped_events"]) == (1, "text", 0)
            assert "Synthetic" not in json.dumps(accepted)
            assert accepted["session_id"] == capability["session_id"]
            unit.output_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=unit.session.session_id))


def test_superseded_protocol_neutral_audio_never_creates_acceptance(observed):
    service, cid = observed
    _manual(service, cid)
    turns = service.speculative_turns = SpeculativeTurnTracker()
    turn, revision = turns.start_turn()
    turns.segment_finalized(100)
    service.dispatch_pipeline_event(
        cid, AudioInputCompletedEvent(audio=np.zeros(160, dtype=np.float32), turn_id=turn, turn_revision=revision)
    )
    new_revision = reopen(turns, turn, revision)
    turns.gate(turn, new_revision, commit=True)
    assert _accepted(service, cid) == []


def test_sequence_exhaustion_never_reuses_an_identity(observed):
    service, cid = observed
    state = service._state(cid)._user_input_observations
    state.sequence = observation.MAX_INPUT_OBSERVATION_SEQUENCE - 1
    _add(service, cid)
    assert _accepted(service, cid)[0].sequence == observation.MAX_INPUT_OBSERVATION_SEQUENCE
    _add(service, cid)
    assert _accepted(service, cid) == []
    assert state.dropped_events == 1


@pytest.mark.parametrize("direct_audio", [False, True])
def test_opted_wire_reports_input_without_any_generation(direct_audio):
    unit = make_pipeline_unit()
    app = create_app([unit], Event())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers={observation.USER_INPUT_OBSERVATION_HEADER: "1"}) as ws:
            assert ws.receive_json()["type"] == "session.created"
            assert ws.receive_json()["type"] == "speech_to_speech.user_input.observation_started"
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}},
                    },
                }
            )
            assert ws.receive_json()["type"] == "session.updated"
            unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
            unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_1", turn_revision=0, duration_s=0.1))
            unit.text_output_queue.put(
                AudioInputCompletedEvent(audio=np.zeros(160, dtype=np.float32), turn_id="turn_1", turn_revision=0)
                if direct_audio
                else TranscriptionCompletedEvent(transcript="Synthetic", turn_id="turn_1", turn_revision=0)
            )
            types = [ws.receive_json()["type"] for _ in range(5 if direct_audio else 6)]
            assert types == [
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed",
                "conversation.item.created",
                *([] if direct_audio else ["conversation.item.input_audio_transcription.completed"]),
                "speech_to_speech.user_input.accepted",
            ]
            assert unit.text_prompt_queue.empty()
            unit.output_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=unit.session.session_id))


def test_kind_snapshot_survives_image_consumption_after_real_append(observed, monkeypatch):
    from speech_to_speech.api.openai_realtime.handlers import conversation

    service, cid = observed
    original = conversation.add_supported_item

    def consume(chat, item):
        original(chat, item)
        item.content = []  # The LM may strip consumed media from the retained shared item.

    monkeypatch.setattr(conversation, "add_supported_item", consume)
    _add(service, cid, _user([{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]))
    assert [e.input_kind for e in _accepted(service, cid)] == ["image"]


async def test_partial_inline_dispatch_failure_still_drains_real_acceptance(observed, monkeypatch):
    service, cid = observed
    transport = _Transport()
    unit = make_pipeline_unit(service=service)
    unit.session = SessionState(session_id=cid, transport=transport)
    original = service.conversation._append_item
    calls = 0

    def apply(conn_id, item):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ChatItemError("Synthetic partial mutation")
        original(conn_id, item)

    monkeypatch.setattr(service.conversation, "_append_item", apply)
    with pytest.raises(ChatItemError):
        await _dispatch_client_event(
            unit, cid, {"type": "response.create", "response": {"input": [_user(), _user()]}}, transport
        )
    assert len(service._state(cid).runtime_config.chat.buffer) == 1
    assert len(transport.events) == 1 and transport.events[0].type == "speech_to_speech.user_input.accepted"
