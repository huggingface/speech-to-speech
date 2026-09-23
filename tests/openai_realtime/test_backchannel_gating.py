from __future__ import annotations

import time
from queue import Queue
from threading import Event as ThreadingEvent

from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.server import create_app
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import (
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import AudioOutput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def _setup_app():
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    cancel_scope = CancelScope()
    speculative_turns = SpeculativeTurnTracker()
    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
        cancel_scope=cancel_scope,
        speculative_turns=speculative_turns,
    )
    input_queue: Queue = Queue()
    output_queue: Queue = Queue()
    text_output_queue: Queue = Queue()
    stop_event = ThreadingEvent()
    response_playing = ThreadingEvent()
    unit = PipelineUnit(
        index=0,
        service=service,
        cancel_scope=cancel_scope,
        should_listen=should_listen,
        response_playing=response_playing,
        input_queue=input_queue,
        output_queue=output_queue,
        text_output_queue=text_output_queue,
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )
    app = create_app(pool=[unit], stop_event=stop_event)
    return (
        app,
        service,
        output_queue,
        text_output_queue,
        text_prompt_queue,
        response_playing,
        cancel_scope,
    )


def test_backchannel_speech_resumes_response() -> None:
    """A short backchannel ('mm-hmm') during active playback must not cancel the response or trigger an LLM prompt."""
    (
        app,
        service,
        output_queue,
        text_output_queue,
        text_prompt_queue,
        response_playing,
        cancel_scope,
    ) = _setup_app()

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            conn_id = list(service._conns.keys())[0]
            service.response._ensure_response(conn_id)
            response_playing.set()

            # Active assistant output chunk in queue
            output_queue.put(AudioOutput(audio=b"\x01\x02" * 100, response_key="resp_1"))

            # User vocalizes a brief backchannel: speech starts
            text_output_queue.put(SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
            # Client receives speech_started
            msg1 = ws.receive_json()
            assert msg1["type"] == "input_audio_buffer.speech_started"

            time.sleep(0.05)
            # Response must NOT be cancelled
            assert not cancel_scope.discarding

            # User speech ends (short duration)
            text_output_queue.put(
                SpeechStoppedEvent(
                    turn_id="turn_1",
                    turn_revision=0,
                    duration_s=0.35,
                )
            )
            # Transcription completes as a backchannel
            text_output_queue.put(
                TranscriptionCompletedEvent(
                    transcript="mm-hmm",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )

            time.sleep(0.1)

            # Verification:
            # 1. Response generation was NOT cancelled
            assert not cancel_scope.discarding
            # 2. No new LLM response request was queued for the backchannel
            assert text_prompt_queue.empty()
            # 3. Backchannel turn was committed in speculative tracker so it never reopens/accumulates
            assert service.speculative_turns.is_committed("turn_1", 0)


def test_substantive_interruption_cancels_response() -> None:
    """A real interruption ('Stop talking') must cancel the active response and queue a new LLM prompt."""
    (
        app,
        service,
        output_queue,
        text_output_queue,
        text_prompt_queue,
        response_playing,
        cancel_scope,
    ) = _setup_app()

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            conn_id = list(service._conns.keys())[0]
            service.response._ensure_response(conn_id)
            response_playing.set()

            # User interrupts with substantive command
            text_output_queue.put(SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
            ws.receive_json()  # speech_started

            text_output_queue.put(
                SpeechStoppedEvent(
                    turn_id="turn_1",
                    turn_revision=0,
                    duration_s=0.6,
                )
            )
            text_output_queue.put(
                TranscriptionCompletedEvent(
                    transcript="Stop talking right now",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )

            time.sleep(0.1)

            # Verification:
            # 1. Hard cancel executed
            assert cancel_scope.discarding
            # 2. New LLM prompt queued with the user's interruption text
            assert not text_prompt_queue.empty()
            req = text_prompt_queue.get_nowait()
            assert req.turn_id == "turn_1"


def test_watchdog_auto_cancels_on_long_speech() -> None:
    """Continuous speech exceeding 1.2s must auto-promote to hard cancel before STT completes."""
    (
        app,
        service,
        output_queue,
        text_output_queue,
        text_prompt_queue,
        response_playing,
        cancel_scope,
    ) = _setup_app()

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            conn_id = list(service._conns.keys())[0]
            service.response._ensure_response(conn_id)
            response_playing.set()

            # User starts speaking
            text_output_queue.put(SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
            ws.receive_json()  # speech_started

            # Simulate passage of 1.3 seconds while speaking
            st = service._conns[conn_id]
            st.tentative_interruption_started_at = time.monotonic() - 1.3

            time.sleep(0.05)

            # Verification: Watchdog promoted to hard cancel
            assert cancel_scope.discarding
