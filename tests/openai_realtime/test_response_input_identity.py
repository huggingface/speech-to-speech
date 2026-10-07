"""Response ownership must follow supplied input, even during newer speech."""

import asyncio
from queue import Queue
from threading import Event

import numpy as np
import pytest
import torch
from openai.types.realtime import ConversationItemCreateEvent, ResponseCreateEvent
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig, _ToolCallCoordinator
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.chat import make_user_audio_message
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AudioInputCompletedEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import AssistantToolCallPart
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker, TurnPhase
from speech_to_speech.VAD.vad_handler import VADHandler
from tests.test_vad_iterator import _FakeVADModel


@pytest.mark.asyncio
@pytest.mark.parametrize("origin_kind", ["client_text", "speech", "direct_audio"])
async def test_late_client_tool_followup_does_not_commit_unfinished_speech(monkeypatch, runtime_config, origin_kind):
    tracker = SpeculativeTurnTracker()
    prompts, events = Queue(), Queue()
    listening = Event()
    listening.set()
    service = RealtimeService(text_prompt_queue=prompts, should_listen=listening, speculative_turns=tracker)
    conn_id = service.register()
    service._state(conn_id).runtime_config = runtime_config
    # Keep the actual VAD iterator/handler; replace only the neural probabilities.
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: (_FakeVADModel([0.9] * 4 + [0.1] * 4 + [0.9] * 4), None))
    vad = VADHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "should_listen": listening,
            "speculative_turns": tracker,
            "text_output_queue": events,
            "min_speech_ms": 64,
            "min_speech_continuation_ms": 64,
            "min_silence_ms": 64,
            "speech_pad_ms": 0,
            "speculative_reopen_ms": 0,
            "smart_turn": False,
        },
    )
    tool_started, tool_release = asyncio.Event(), asyncio.Event()
    sent = []

    async def tool_executor(name, arguments):
        tool_started.set()
        await tool_release.wait()
        return "sunny"

    class Connection:
        async def send(self, payload):
            sent.append(payload)
            if payload["type"] == "conversation.item.create":
                replies = service.handle_conversation_item_create(
                    conn_id, ConversationItemCreateEvent.model_validate(payload)
                )
            else:
                replies = [service.handle_response_create(conn_id, ResponseCreateEvent.model_validate(payload))]
            for reply in replies:
                if reply is not None:
                    coordinator.handle_event(reply)

    coordinator = _ToolCallCoordinator(
        Connection(),
        RealtimeAudioClientConfig(
            tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            tool_executor=tool_executor,
        ),
    )

    def speech_chunks(count):
        audio = (np.ones(512, dtype=np.int16) * 1000).tobytes()
        segments = []
        for _ in range(count):
            segments.extend(vad.process((audio, runtime_config)))
            while not events.empty():
                for reply in service.dispatch_pipeline_event(conn_id, events.get_nowait()):
                    coordinator.handle_event(reply)
        return segments

    try:
        if origin_kind == "client_text":
            service.handle_conversation_item_create(
                conn_id,
                ConversationItemCreateEvent.model_validate(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Weather?"}],
                        },
                    }
                ),
            )
            created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
            coordinator.handle_event(created)
        else:
            turn_id, revision, _ = tracker.speech_started(0)
            tracker.segment_finalized(100)
            supplied = (
                TranscriptionCompletedEvent(transcript="Weather?", turn_id=turn_id, turn_revision=revision)
                if origin_kind == "speech"
                else AudioInputCompletedEvent(
                    audio=np.zeros(1600, dtype=np.float32),
                    audio_duration_s=0.1,
                    turn_id=turn_id,
                    turn_revision=revision,
                )
            )
            service.dispatch_pipeline_event(conn_id, supplied)
        origin = prompts.get_nowait()
        if origin_kind == "direct_audio":
            audio_item = make_user_audio_message("AAAA")
            audio_item.id = origin.input_item_id
            runtime_config.chat.add_provisional_generation_items(origin.response_key, [audio_item])
        call = RealtimeConversationItemFunctionCall(
            type="function_call", id="fc_call", call_id="call_weather", name="lookup", arguments="{}"
        )
        runtime_config.chat.add_provisional_generation_items(origin.response_key, [call])
        for reply in service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=origin.response_key,
                turn_id=origin.turn_id,
                turn_revision=origin.turn_revision,
                parts=[AssistantToolCallPart(tool=call.model_dump())],
            ),
        ):
            coordinator.handle_event(reply)
        await asyncio.wait_for(tool_started.wait(), 1)
        for reply in service.finish_response(conn_id, response_key=origin.response_key):
            coordinator.handle_event(reply)
        speech_chunks(4)
        unfinished_id, unfinished_revision = tracker.current_turn()
        assert unfinished_revision == 0
        assert tracker.phase == TurnPhase.LISTENING
        tool_release.set()
        for _ in range(100):
            if any(p["type"] == "response.create" for p in sent):
                break
            await asyncio.sleep(0.01)
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        followup = prompts.get_nowait()
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="Sunny",
                response_key=followup.response_key,
                turn_id=followup.turn_id,
                turn_revision=followup.turn_revision,
            ),
        )
        committed = tracker.is_committed(unfinished_id, 0)
        phase_during_speech = tracker.phase
        finalized = speech_chunks(4)
        assert finalized
        phase_after_segment = tracker.phase
        speech_chunks(4)
        print(
            f"followup={(followup.turn_id, followup.turn_revision)} committed={committed} resumed={tracker.current_turn()}"
        )
        assert tracker.current_turn() == (unfinished_id, 1)
        assert not committed, "tool reply committed speech it never consumed"
        assert phase_during_speech == TurnPhase.LISTENING
        assert phase_after_segment == TurnPhase.SOFT_ENDED
        assert (followup.turn_id, followup.turn_revision) == (origin.turn_id, origin.turn_revision)
    finally:
        await coordinator.close()
        service.unregister(conn_id)


@pytest.mark.parametrize("associated", [False, True])
@pytest.mark.parametrize(
    ("newer_turn", "prefetch"),
    # A newer transcript discards a prefetch, so a question only meets explicit follow-ups.
    [("listening", False), ("listening", True), ("closed", False), ("closed", True), ("question", False)],
)
def test_tool_continuation_turn_after_newer_speech(runtime_config, prefetch, associated, newer_turn):
    tracker, prompts = SpeculativeTurnTracker(), Queue()
    service = RealtimeService(text_prompt_queue=prompts, speculative_turns=tracker)
    conn_id = service.register()
    service._state(conn_id).runtime_config = runtime_config
    if associated:
        origin_id, revision, _ = tracker.speech_started(0)
        tracker.segment_finalized(100)
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="Weather?",
                turn_id=origin_id,
                turn_revision=revision,
                speech_stopped_at_s=123.0,
            ),
        )
    else:
        origin_id, revision = None, None
        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    origin = prompts.get_nowait()
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_origin", call_id="call_origin", name="lookup", arguments="{}"
    )
    runtime_config.chat.add_provisional_generation_items(origin.response_key, [call])
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=origin.response_key,
            turn_id=origin_id,
            turn_revision=revision,
            parts=[AssistantToolCallPart(tool=call.model_dump())],
        ),
    )
    # With interruption disabled, an accepted origin can finish while newer
    # input remains open. This tests both explicit and prefetched requests.
    unfinished_id, _, _ = tracker.speech_started(200)
    if not prefetch:
        service.finish_response(conn_id, response_key=origin.response_key)
    if newer_turn != "listening":
        # "closed": the newer turn transcribes empty, so nothing answers it and it closes.
        # "question": the model refuses it while the tool call has no output.
        newer = {"turn_id": unfinished_id, "turn_revision": 0}
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(interrupt_response=False, **newer))
        tracker.segment_finalized(300)
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(duration_s=0.1, **newer))
        transcript = "And hotels?" if newer_turn == "question" else ""
        service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript=transcript, **newer))
        if newer_turn == "question":
            refused = prompts.get_nowait()
            message = "Cannot generate a response while function call outputs are pending."
            service.dispatch_pipeline_event(
                conn_id, ResponseFailedEvent(message=message, response_key=refused.response_key, **newer)
            )
            service.finish_response(conn_id, response_key=refused.response_key)
    service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent.model_validate(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call.call_id, "output": "sunny"},
            }
        ),
    )
    if prefetch:
        from speech_to_speech.pipeline.events import ResponseGenerationDoneEvent

        service.dispatch_pipeline_event(
            conn_id, ResponseGenerationDoneEvent(response_key=origin.response_key, call_ids=[call.call_id])
        )
    else:
        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    request = prompts.get_nowait()
    if newer_turn == "closed" and associated:
        # No one is speaking, so the late result moves to the closed turn
        # instead of being dropped as stale.
        assert tracker.phase == TurnPhase.CLOSED
        expected = (unfinished_id, 0, None)
    elif newer_turn == "question":
        # The follow-up's chat holds the unanswered question, so it answers that turn.
        expected = (unfinished_id, 0, None)
    else:
        expected = (origin_id, revision, 123.0 if associated else None)
    assert (request.turn_id, request.turn_revision, request.speech_stopped_at_s) == expected
    if newer_turn == "listening":
        assert not tracker.is_committed(unfinished_id, 0)
        assert tracker.phase == TurnPhase.LISTENING
    if prefetch:
        # Prefetch output is held until the client claims the response.
        service.finish_response(conn_id, response_key=origin.response_key)
        service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    events = service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            text="Sunny",
            response_key=request.response_key,
            turn_id=request.turn_id,
            turn_revision=request.turn_revision,
        ),
    )
    if newer_turn != "listening":
        assert "response.output_audio_transcript.delta" in [event.type for event in events]
        service.unregister(conn_id)
        return
    assert not tracker.is_committed(unfinished_id, 0)
    assert tracker.phase == TurnPhase.LISTENING
    tracker.segment_finalized(300, output_hold_ms=800, processing_delay_ms=600)
    assert tracker.processing_deadline(unfinished_id, 0) is not None
    assert tracker.speech_started(400) == (unfinished_id, 1, True)
    service.unregister(conn_id)


def test_response_input_follows_supplied_revision_without_advancing_tracker(runtime_config):
    tracker, prompts = SpeculativeTurnTracker(), Queue()
    service = RealtimeService(text_prompt_queue=prompts, speculative_turns=tracker)
    conn_id = service.register()
    service._state(conn_id).runtime_config = runtime_config
    tracker.speech_started(0)
    tracker.segment_finalized(100)
    service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(
            transcript="Find a hotel",
            turn_id="turn_1",
            turn_revision=0,
        ),
    )
    service.close_pending_responses(conn_id)
    assert service.response_input_turn(conn_id)[:2] == ("turn_1", 0)
    tracker.speech_started(200)
    # Speaking revision 1 has not replaced the supplied revision 0 yet.
    assert service.response_input_turn(conn_id)[:2] == ("turn_1", 0)
    assert tracker.current_turn() == ("turn_1", 1)
    service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    prompts.get_nowait()  # automatic revision 0 request
    stale = prompts.get_nowait()
    assert (
        service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                text="obsolete",
                response_key=stale.response_key,
                turn_id=stale.turn_id,
                turn_revision=stale.turn_revision,
            ),
        )
        == []
    )
    assert tracker.phase == TurnPhase.LISTENING
    tracker.segment_finalized(300)
    service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(
            transcript="Find a hotel near the station",
            turn_id="turn_1",
            turn_revision=1,
        ),
    )
    assert service.response_input_turn(conn_id)[:2] == ("turn_1", 1)
    assert len([i for i in runtime_config.chat.buffer if getattr(i, "role", None) == "user"]) == 1
    service.unregister(conn_id)


def test_direct_audio_chat_item_retains_its_input_identity(runtime_config):
    from tests.test_chat_completions_backend import _make_handler

    tracker, prompts = SpeculativeTurnTracker(), Queue()
    service = RealtimeService(text_prompt_queue=prompts, speculative_turns=tracker)
    conn_id = service.register()
    service._state(conn_id).runtime_config = runtime_config
    tracker.speech_started(0)
    tracker.segment_finalized(100)
    service.dispatch_pipeline_event(
        conn_id,
        AudioInputCompletedEvent(
            audio=np.zeros(1600, dtype=np.float32),
            audio_duration_s=0.1,
            turn_id="turn_1",
            turn_revision=0,
            speech_stopped_at_s=123.0,
        ),
    )
    request = prompts.get_nowait()
    from tests.llm_history import drive_llm

    list(drive_llm(_make_handler(stream=False), request))
    assert runtime_config.chat.buffer[0].id == request.input_item_id
    tracker.speech_started(200)
    assert service.response_input_turn(conn_id) == ("turn_1", 0, 123.0)
    assert tracker.phase == TurnPhase.LISTENING
    service.unregister(conn_id)
