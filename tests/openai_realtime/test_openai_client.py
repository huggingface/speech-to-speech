"""Tests using the real OpenAI Python SDK client connected to our local server.

``AsyncOpenAI.realtime.connect()`` (non-beta) establishes a WebSocket
connection and returns parsed event objects from ``openai.types.realtime``.
These tests start our FastAPI app on a local port with uvicorn, then drive
a real SDK client against it — exactly as the production client does.

The pipeline side (audio output, text events) is driven through the queues,
while the client side uses ``conn.send()`` and ``async for event in conn``.

We use ``client.realtime.connect()`` (non-beta), **not**
``client.realtime.connect()``.  The non-beta path expects the GA type
strings our server emits (e.g. ``response.output_audio.delta``), whereas
the beta path expects the older ``response.audio.delta`` variants.
"""

import asyncio
import base64
import json
import socket
import sys
import threading
import time
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from types import SimpleNamespace

import numpy as np
import pytest
import uvicorn
from openai import AsyncOpenAI
from openai.types.realtime import RealtimeConversationItemFunctionCall

import speech_to_speech.api.openai_realtime.audio_client as audio_client_module
from speech_to_speech.api.openai_realtime.audio_client import (
    RealtimeAudioClientConfig,
    ToolResult,
    listen_and_play_realtime,
)
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantToolCallReadyEvent,
    AudioInputCompletedEvent,
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    AssistantTextPart,
    AssistantToolCallPart,
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pcm_bytes(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


class _ServerEnv:
    """Wraps a running uvicorn server + all pipeline queues."""

    def __init__(self):
        self.text_prompt_queue: Queue = Queue()
        self.should_listen = ThreadingEvent()
        self.should_listen.set()
        self.service = RealtimeService(
            text_prompt_queue=self.text_prompt_queue,
            should_listen=self.should_listen,
        )
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.text_output_queue: Queue = Queue()
        self.stop_event = ThreadingEvent()
        self.response_playing = ThreadingEvent()
        self.cancel_scope = CancelScope()
        self.unit = PipelineUnit(
            index=0,
            service=self.service,
            cancel_scope=self.cancel_scope,
            should_listen=self.should_listen,
            response_playing=self.response_playing,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            text_output_queue=self.text_output_queue,
            text_prompt_queue=self.text_prompt_queue,
            handlers=[],
        )
        self.app = create_app(pool=[self.unit], stop_event=self.stop_event)
        self.port = _free_port()
        self._server_thread: threading.Thread | None = None

    def start(self):
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._server = server
        self._server_thread = threading.Thread(target=server.run, daemon=True)
        self._server_thread.start()
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("Server did not start in time")

    def stop(self):
        self.stop_event.set()
        self._server.should_exit = True
        if self._server_thread:
            self._server_thread.join(timeout=5)

    def make_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            websocket_base_url=f"ws://127.0.0.1:{self.port}/v1",
        )


@pytest.fixture
def server_env():
    env = _ServerEnv()
    env.start()
    yield env
    env.stop()


async def _recv(conn, timeout: float = 3.0):
    """Receive next event with a timeout to avoid hanging tests."""
    return await asyncio.wait_for(conn.recv(), timeout=timeout)


# Our server uses the openai.types.realtime type strings (e.g.
# "response.output_audio.done").  The production client code matches on
# event.type using both GA and legacy names for compatibility.  These
# constants match the Literal values from openai.types.realtime.
SESSION_CREATED = "session.created"
SPEECH_STARTED = "input_audio_buffer.speech_started"
SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
TRANSCRIPTION_DELTA = "conversation.item.input_audio_transcription.delta"
TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
ITEM_CREATED = "conversation.item.created"
RESPONSE_CREATED = "response.created"
RESPONSE_DONE = "response.done"
AUDIO_DELTA = "response.output_audio.delta"
AUDIO_DONE = "response.output_audio.done"
TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
TRANSCRIPT_DONE = "response.output_audio_transcript.done"
FUNCTION_CALL_DONE = "response.function_call_arguments.done"
OUTPUT_ITEM_ADDED = "response.output_item.added"
OUTPUT_ITEM_DONE = "response.output_item.done"
ERROR = "error"


# ===================================================================
# 1. Connection and session.created
# ===================================================================


class TestSDKConnection:
    @pytest.mark.asyncio
    async def test_connect_receives_session_created(self, server_env):
        """SDK connect yields session.created as the first event."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            event = await _recv(conn)
            assert event.type == SESSION_CREATED
            assert event.event_id.startswith("event_")
            assert event.session is not None


# ===================================================================
# 2. Session update
# ===================================================================


class TestSDKSessionUpdate:
    @pytest.mark.asyncio
    async def test_session_update_applies_config(self, server_env):
        """conn.session.update() applies config server-side."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)  # session.created

            await conn.send(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": "You are a helpful robot",
                        "audio": {
                            "input": {
                                "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "interrupt_response": True,
                                },
                            },
                            "output": {
                                "voice": "alloy",
                            },
                        },
                        "tools": [{"type": "function", "name": "get_weather"}],
                        "tool_choice": "auto",
                    },
                }
            )
            await asyncio.sleep(0.2)

            cid = server_env.service.connection_ids[0]
            s = server_env.service._state(cid).runtime_config.session
            assert s.audio.output.voice == "alloy"
            assert s.instructions == "You are a helpful robot"
            assert s.audio.input.turn_detection.type == "server_vad"
            assert s.tools is not None
            assert s.tool_choice == "auto"


# ===================================================================
# 3. Full voice conversation turn
# ===================================================================


class TestSDKVoiceTurn:
    @pytest.mark.asyncio
    async def test_full_voice_turn(self, server_env):
        """
        Pipeline-driven voice turn through the real SDK:
          speech_started → partial transcription → speech_stopped →
          transcription_completed → audio response → transcript → done
        """
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)  # session.created

            # -- User speech --
            server_env.text_output_queue.put(SpeechStartedEvent())
            event = await _recv(conn)
            assert event.type == SPEECH_STARTED
            assert event.audio_start_ms == 0
            item_id = event.item_id

            server_env.text_output_queue.put(PartialTranscriptionEvent(delta="hello there"))
            server_env.text_output_queue.put(PartialTranscriptionEvent(delta="hello there friend"))
            event = await _recv(conn)
            assert event.type == TRANSCRIPTION_DELTA
            assert event.delta == "hello"
            assert event.item_id == item_id

            server_env.text_output_queue.put(SpeechStoppedEvent(duration_s=1.9))
            event = await _recv(conn)
            assert event.type == SPEECH_STOPPED
            assert event.audio_end_ms == 0
            assert event.item_id == item_id

            server_env.text_output_queue.put(TranscriptionCompletedEvent(transcript="hello"))
            event = await _recv(conn)
            assert event.type == TRANSCRIPTION_COMPLETED
            assert event.transcript == "hello"
            assert event.usage.seconds == 1.9

            # -- Server audio response --
            server_env.output_queue.put(_pcm_bytes(256))
            event = await _recv(conn)
            assert event.type == RESPONSE_CREATED
            assert event.response.status == "in_progress"
            assert event.response.object == "realtime.response"
            conversation_id = event.response.conversation_id

            event = await _recv(conn)
            assert event.type == AUDIO_DELTA
            decoded = base64.b64decode(event.delta)
            assert len(decoded) == len(_pcm_bytes(256))

            server_env.text_output_queue.put(AssistantOutputEvent(text="Hi there!"))
            event = await _recv(conn)
            assert event.type == TRANSCRIPT_DELTA
            assert event.delta == "Hi there!"

            server_env.output_queue.put(PIPELINE_END)
            event = await _recv(conn)
            assert event.type == AUDIO_DONE

            event = await _recv(conn)
            assert event.type == TRANSCRIPT_DONE
            assert event.transcript == "Hi there!"

            event = await _recv(conn)
            assert event.type == RESPONSE_DONE
            assert event.response.status == "completed"
            assert event.response.conversation_id == conversation_id


# ===================================================================
# 3b. Packaged local client parity
# ===================================================================


class TestPackagedAudioClient:
    @pytest.mark.asyncio
    async def test_direct_audio_reopen_cancels_revision_zero_over_loopback(
        self,
        server_env,
        monkeypatch,
        capsys,
    ):
        """The embedded client uses the public WebSocket revision/cancellation lifecycle."""

        tracker = SpeculativeTurnTracker()
        server_env.service.speculative_turns = tracker
        client_stop = ThreadingEvent()
        received_events = []

        class FakeInputStream:
            def __init__(self, *, callback, **_kwargs):
                self.callback = callback

            def start(self):
                # Two microphone callbacks become the original segment and its
                # resumed continuation in the deterministic pipeline driver below.
                self.callback(_pcm_bytes(512), 512, None, None)
                self.callback(_pcm_bytes(512), 512, None, None)

            def stop(self):
                pass

            def close(self):
                pass

        class FakeOutputStream:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(
            sys.modules,
            "sounddevice",
            SimpleNamespace(RawInputStream=FakeInputStream, RawOutputStream=FakeOutputStream),
        )

        original_handle_server_event = audio_client_module.handle_server_event

        def record_server_event(event, **kwargs):
            received_events.append(event)
            original_handle_server_event(event, **kwargs)
            if event.type == TRANSCRIPT_DELTA and event.delta == "fresh revision one":
                client_stop.set()

        monkeypatch.setattr(audio_client_module, "handle_server_event", record_server_event)

        async def queue_get(queue: Queue, timeout: float = 3.0):
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                try:
                    return queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)
            raise AssertionError("Timed out waiting for the loopback pipeline queue")

        async def wait_until(predicate, timeout: float = 3.0):
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("Timed out waiting for the loopback pipeline state")

        async def drive_direct_audio_reopen():
            first_chunk, first_config = await queue_get(server_env.input_queue)
            assert len(first_chunk) == 1024
            first_generation = server_env.cancel_scope.generation

            tracker.observe("turn_1", 0)
            tracker.start_reopen_grace("turn_1", 0, 5.0)
            server_env.text_output_queue.put(SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
            server_env.text_output_queue.put(
                SpeechStoppedEvent(
                    duration_s=0.032,
                    audio_end_ms=32,
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
            server_env.text_output_queue.put(
                AudioInputCompletedEvent(
                    audio=np.zeros(512, dtype=np.float32),
                    audio_sample_rate=16000,
                    audio_duration_s=0.032,
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
            first_request = await queue_get(server_env.text_prompt_queue)

            second_chunk, second_config = await queue_get(server_env.input_queue)
            assert len(second_chunk) == 1024
            assert second_config is first_config

            candidate = tracker.begin_reopen_candidate("turn_1", 0)
            assert candidate == 1
            assert tracker.confirm_reopen_candidate("turn_1", 0, candidate)
            server_env.text_output_queue.put(SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True))
            await wait_until(lambda: server_env.cancel_scope.generation == first_generation + 1)
            reopened_generation = server_env.cancel_scope.generation

            server_env.text_output_queue.put(
                SpeechStoppedEvent(
                    duration_s=0.064,
                    audio_end_ms=64,
                    turn_id="turn_1",
                    turn_revision=1,
                )
            )
            server_env.text_output_queue.put(
                AudioInputCompletedEvent(
                    audio=np.zeros(1024, dtype=np.float32),
                    audio_sample_rate=16000,
                    audio_duration_s=0.064,
                    turn_id="turn_1",
                    turn_revision=1,
                )
            )
            second_request = await queue_get(server_env.text_prompt_queue)

            server_env.text_output_queue.put(
                AssistantOutputEvent(
                    text="stale revision zero",
                    turn_id="turn_1",
                    turn_revision=0,
                    cancel_generation=first_generation,
                )
            )
            server_env.text_output_queue.put(
                AssistantOutputEvent(
                    text="fresh revision one",
                    turn_id="turn_1",
                    turn_revision=1,
                    cancel_generation=first_generation + 1,
                )
            )
            await wait_until(client_stop.is_set)
            return (first_request, second_request), first_generation, reopened_generation

        pipeline_task = asyncio.create_task(drive_direct_audio_reopen())
        client_task = asyncio.create_task(
            listen_and_play_realtime(
                RealtimeAudioClientConfig(
                    url=f"ws://127.0.0.1:{server_env.port}/v1/realtime",
                    api_key="local",
                ),
                stop_event=client_stop,
            )
        )
        pipeline_result, _ = await asyncio.wait_for(
            asyncio.gather(pipeline_task, client_task),
            timeout=5.0,
        )
        requests, first_generation, reopened_generation = pipeline_result

        assert all(isinstance(request, GenerateResponseRequest) for request in requests)
        assert [request.turn_revision for request in requests] == [0, 1]
        assert reopened_generation == first_generation + 1
        transcript_deltas = [event.delta for event in received_events if event.type == TRANSCRIPT_DELTA]
        assert transcript_deltas == ["fresh revision one"]
        capsys.readouterr()

    @pytest.mark.asyncio
    async def test_tool_loop_runs_through_sdk_server_and_follow_up_response(
        self,
        server_env,
        monkeypatch,
        capsys,
    ):
        """The SDK client starts tool follow-up generation while TTS is blocked."""

        class FakeStream:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(
            sys.modules,
            "sounddevice",
            SimpleNamespace(RawInputStream=FakeStream, RawOutputStream=FakeStream),
        )

        calls = []
        first_tool_started = asyncio.Event()
        second_tool_started = asyncio.Event()
        follow_up_created = asyncio.Event()
        tts_started = [ThreadingEvent(), ThreadingEvent()]
        release_tts = [ThreadingEvent(), ThreadingEvent()]
        tts_invocations = 0
        received_events = []

        async def executor(name, arguments):
            calls.append((name, arguments))
            if arguments["index"] == 7:
                first_tool_started.set()
                return {"value": "first"}
            second_tool_started.set()
            return ToolResult({"value": "second"}, create_response=False)

        class BlockingTTS(BaseHandler):
            def process(self, item):
                nonlocal tts_invocations
                if isinstance(item, TTSInput):
                    invocation = tts_invocations
                    tts_invocations += 1
                    tts_started[invocation].set()
                    while not release_tts[invocation].wait(0.01):
                        if self.stop_event.is_set():
                            return
                    yield _pcm_bytes(256)
                elif isinstance(item, EndOfResponse):
                    yield AUDIO_RESPONSE_DONE

        client_stop = ThreadingEvent()
        config = RealtimeAudioClientConfig(
            url=f"ws://127.0.0.1:{server_env.port}/v1/realtime",
            api_key="local",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"index": {"type": "integer"}},
                        "required": ["index"],
                    },
                }
            ],
            tool_executor=executor,
        )

        async def wait_until(predicate, timeout: float = 3.0):
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("Timed out waiting for the packaged tool loop")

        original_handle_server_event = audio_client_module.handle_server_event

        def record_server_event(event, **kwargs):
            received_events.append(event)
            original_handle_server_event(event, **kwargs)
            metadata = getattr(getattr(event, "response", None), "metadata", None)
            if event.type == RESPONSE_CREATED and isinstance(metadata, dict) and "s2s_local_tool_create_id" in metadata:
                follow_up_created.set()

        monkeypatch.setattr(audio_client_module, "handle_server_event", record_server_event)

        lm_output_queue: Queue = Queue()
        tts_input_queue: Queue = Queue()
        lm_output_processor = LMOutputProcessor(
            server_env.stop_event,
            lm_output_queue,
            tts_input_queue,
            setup_kwargs={"text_output_queue": server_env.text_output_queue},
        )
        blocking_tts = BlockingTTS(
            server_env.stop_event,
            tts_input_queue,
            server_env.output_queue,
        )
        processor_thread = threading.Thread(target=lm_output_processor.run, daemon=True)
        tts_thread = threading.Thread(target=blocking_tts.run, daemon=True)
        processor_thread.start()
        tts_thread.start()

        client_task = asyncio.create_task(listen_and_play_realtime(config, stop_event=client_stop))
        try:
            await wait_until(
                lambda: (
                    bool(server_env.service.connection_ids)
                    and bool(
                        server_env.service._state(server_env.service.connection_ids[0]).runtime_config.session.tools
                    )
                )
            )
            conn_id = server_env.service.connection_ids[0]
            chat = server_env.service._state(conn_id).runtime_config.chat
            chat.add_item(
                RealtimeConversationItemFunctionCall(
                    type="function_call",
                    call_id="call_integration",
                    name="lookup",
                    arguments='{"index": 7}',
                )
            )

            response_key = "response_blocked_tts"
            lm_output_queue.put(
                LLMResponseChunk(
                    response_key=response_key,
                    parts=[
                        AssistantTextPart(text="One moment."),
                        AssistantToolCallPart(
                            tool={
                                "type": "function_call",
                                "call_id": "call_integration",
                                "name": "lookup",
                                "arguments": '{"index": 7}',
                            }
                        ),
                    ],
                )
            )
            lm_output_queue.put(EndOfResponse(response_key=response_key))

            assert await asyncio.to_thread(tts_started[0].wait, 1.0)
            await asyncio.wait_for(first_tool_started.wait(), timeout=1.0)
            await wait_until(lambda: not server_env.text_prompt_queue.empty())
            follow_up = server_env.text_prompt_queue.get_nowait()

            assert isinstance(follow_up, GenerateResponseRequest)
            assert follow_up.prefetch_transaction is not None
            assert calls == [("lookup", {"index": 7})]
            assert [item.type for item in chat.buffer[-2:]] == ["function_call", "function_call_output"]
            assert json.loads(chat.buffer[-1].output) == {"value": "first"}
            assert not release_tts[0].is_set()
            assert not any(event.type == RESPONSE_DONE for event in received_events)

            second_call = RealtimeConversationItemFunctionCall(
                type="function_call",
                id="fc_second_integration",
                call_id="call_second_integration",
                name="lookup",
                arguments='{"index": 8}',
            )
            assert chat.add_provisional_generation_items(follow_up.response_key, [second_call]) is not None
            lm_output_queue.put(
                LLMResponseChunk(
                    response_key=follow_up.response_key,
                    prefetch_transaction=follow_up.prefetch_transaction,
                    parts=[
                        AssistantTextPart(text="One more check."),
                        AssistantToolCallPart(
                            tool={
                                "type": "function_call",
                                "id": second_call.id,
                                "call_id": second_call.call_id,
                                "name": second_call.name,
                                "arguments": second_call.arguments,
                            }
                        ),
                    ],
                )
            )
            lm_output_queue.put(
                EndOfResponse(
                    response_key=follow_up.response_key,
                )
            )
            await wait_until(
                lambda: bool(
                    server_env.unit.session
                    and any(
                        isinstance(item, AssistantToolCallReadyEvent) and item.part.tool.call_id == second_call.call_id
                        for item in server_env.unit.session.pending_text_output_items
                    )
                )
            )
            assert not second_tool_started.is_set()

            release_tts[0].set()
            await asyncio.wait_for(follow_up_created.wait(), timeout=2.0)
            assert await asyncio.to_thread(tts_started[1].wait, 1.0)
            await asyncio.wait_for(second_tool_started.wait(), timeout=1.0)
            await wait_until(lambda: sum(event.type == OUTPUT_ITEM_DONE for event in received_events) == 2)

            event_types = [event.type for event in received_events]
            assert event_types.index(FUNCTION_CALL_DONE) < event_types.index(AUDIO_DONE)
            assert event_types.index(AUDIO_DONE) < event_types.index(RESPONSE_DONE)
            assert event_types.count(OUTPUT_ITEM_ADDED) == 2
            assert event_types.count(OUTPUT_ITEM_DONE) == 2
            assert event_types.count(FUNCTION_CALL_DONE) == 2
            assert event_types.count(RESPONSE_CREATED) == 2
            assert calls == [("lookup", {"index": 7}), ("lookup", {"index": 8})]
            assert not release_tts[1].is_set()
        finally:
            for release in release_tts:
                release.set()
            client_stop.set()
            await asyncio.wait_for(client_task, timeout=3.0)
            lm_output_queue.put(PIPELINE_END)
            processor_thread.join(timeout=1.0)
            tts_thread.join(timeout=1.0)
            assert not processor_thread.is_alive()
            assert not tts_thread.is_alive()
            capsys.readouterr()


# ===================================================================
# 4. Interruption (barge-in)
# ===================================================================


class TestSDKBargeIn:
    @pytest.mark.asyncio
    async def test_speech_interrupts_active_response(self, server_env):
        """User speech during audio streaming cancels with turn_detected."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)  # session.created

            server_env.output_queue.put(_pcm_bytes(256))
            event = await _recv(conn)
            assert event.type == RESPONSE_CREATED
            await _recv(conn)  # audio delta

            server_env.text_output_queue.put(SpeechStartedEvent())

            events = []
            for _ in range(3):
                events.append(await _recv(conn))

            types = [e.type for e in events]
            assert AUDIO_DONE in types
            assert RESPONSE_DONE in types
            assert SPEECH_STARTED in types

            done = next(e for e in events if e.type == RESPONSE_DONE)
            assert done.response.status == "cancelled"
            assert done.response.status_details.reason == "turn_detected"

    @pytest.mark.asyncio
    async def test_stale_assistant_text_flushed_on_interruption(self, server_env):
        """Stale assistant_text queued during interruption is flushed, not reopened as a new response."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)  # session.created

            server_env.output_queue.put(_pcm_bytes(256))
            event = await _recv(conn)
            assert event.type == RESPONSE_CREATED
            await _recv(conn)  # audio delta

            server_env.text_output_queue.put(SpeechStartedEvent())
            server_env.text_output_queue.put(AssistantOutputEvent(text="stale response text"))

            events = []
            for _ in range(3):
                events.append(await _recv(conn))

            types = [e.type for e in events]
            assert AUDIO_DONE in types
            assert RESPONSE_DONE in types
            assert SPEECH_STARTED in types

            done = next(e for e in events if e.type == RESPONSE_DONE)
            assert done.response.status == "cancelled"

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(conn.recv(), timeout=0.5)


# ===================================================================
# 4b. Phantom speech & interruption state
# ===================================================================


class TestSDKPhantomSpeech:
    @pytest.mark.asyncio
    async def test_phantom_speech_does_not_block_pipeline(self, server_env):
        """speech_started + speech_stopped(duration=0) doesn't hang; a normal turn follows."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)  # session.created

            server_env.text_output_queue.put(SpeechStartedEvent())
            event = await _recv(conn)
            assert event.type == SPEECH_STARTED

            server_env.text_output_queue.put(SpeechStoppedEvent())
            event = await _recv(conn)
            assert event.type == SPEECH_STOPPED

            server_env.text_output_queue.put(SpeechStartedEvent())
            event = await _recv(conn)
            assert event.type == SPEECH_STARTED

            server_env.text_output_queue.put(SpeechStoppedEvent(duration_s=2.0))
            event = await _recv(conn)
            assert event.type == SPEECH_STOPPED

            server_env.output_queue.put(_pcm_bytes(256))
            event = await _recv(conn)
            assert event.type == RESPONSE_CREATED
            await _recv(conn)  # audio delta

            server_env.output_queue.put(AUDIO_RESPONSE_DONE)
            event = await _recv(conn)
            assert event.type == AUDIO_DONE
            event = await _recv(conn)
            assert event.type == RESPONSE_DONE
            assert event.response.status == "completed"


class TestSDKInterruptionState:
    @pytest.mark.asyncio
    async def test_interruption_resets_pipeline_state(self, server_env):
        """After interruption, response_playing is cleared and cancel_scope
        enters discarding mode until __RESPONSE_DONE__ arrives."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)  # session.created

            assert not server_env.response_playing.is_set()
            assert not server_env.cancel_scope.discarding

            server_env.output_queue.put(_pcm_bytes(256))
            await _recv(conn)  # response.created
            await _recv(conn)  # audio delta
            assert server_env.response_playing.is_set()

            server_env.text_output_queue.put(SpeechStartedEvent())
            events = []
            for _ in range(3):
                events.append(await _recv(conn))

            types = [e.type for e in events]
            assert SPEECH_STARTED in types
            assert RESPONSE_DONE in types

            await asyncio.sleep(0.1)
            assert not server_env.response_playing.is_set()
            assert server_env.cancel_scope.discarding


# ===================================================================
# 5. Tool calling
# ===================================================================


class TestSDKToolCalling:
    @pytest.mark.asyncio
    async def test_tool_call_events(self, server_env):
        """Tool calls produce events with name, call_id, arguments."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)

            server_env.text_output_queue.put(
                AssistantOutputEvent(
                    text="Checking weather",
                    tools=[
                        {
                            "type": "function_call",
                            "call_id": "call_xyz",
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        }
                    ],
                )
            )
            server_env.output_queue.put(AUDIO_RESPONSE_DONE)

            event = await _recv(conn)
            assert event.type == RESPONSE_CREATED

            event = await _recv(conn)
            assert event.type == TRANSCRIPT_DELTA
            assert event.delta == "Checking weather"

            event = await _recv(conn)
            assert event.type == OUTPUT_ITEM_ADDED
            assert event.item.call_id == "call_xyz"

            event = await _recv(conn)
            assert event.type == FUNCTION_CALL_DONE
            assert event.name == "get_weather"
            assert event.call_id == "call_xyz"
            assert json.loads(event.arguments) == {"city": "Tokyo"}

            event = await _recv(conn)
            assert event.type == OUTPUT_ITEM_DONE
            assert event.item.call_id == "call_xyz"
            assert event.item.status == "completed"

            event = await _recv(conn)
            assert event.type == TRANSCRIPT_DONE

            event = await _recv(conn)
            assert event.type == RESPONSE_DONE

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_output_index(self, server_env):
        """Multiple tool calls have incrementing output_index."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)

            server_env.text_output_queue.put(
                AssistantOutputEvent(
                    text="",
                    tools=[
                        {"type": "function_call", "call_id": "c1", "name": "tool_a", "arguments": "{}"},
                        {"type": "function_call", "call_id": "c2", "name": "tool_b", "arguments": '{"x": 1}'},
                    ],
                )
            )
            server_env.output_queue.put(AUDIO_RESPONSE_DONE)

            created = await _recv(conn)
            added_1 = await _recv(conn)
            e1 = await _recv(conn)
            done_1 = await _recv(conn)
            added_2 = await _recv(conn)
            e2 = await _recv(conn)
            done_2 = await _recv(conn)
            response_terminal = await _recv(conn)
            assert created.type == RESPONSE_CREATED
            assert added_1.type == OUTPUT_ITEM_ADDED
            assert e1.type == FUNCTION_CALL_DONE
            assert done_1.type == OUTPUT_ITEM_DONE
            assert added_2.type == OUTPUT_ITEM_ADDED
            assert e2.type == FUNCTION_CALL_DONE
            assert done_2.type == OUTPUT_ITEM_DONE
            assert response_terminal.type == RESPONSE_DONE
            assert e1.output_index == 0
            assert e2.output_index == 1


# ===================================================================
# 6. Text input via SDK
# ===================================================================


class TestSDKTextInput:
    @pytest.mark.asyncio
    async def test_send_conversation_item_create(self, server_env):
        """Sending conversation.item.create produces an item.created event."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)

            await conn.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "msg_sdk_1",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello from SDK"}],
                    },
                }
            )

            event = await _recv(conn)
            assert event.type == ITEM_CREATED
            assert event.item.role == "user"
            assert event.item.content[0].text == "Hello from SDK"
            assert event.previous_item_id is None

    @pytest.mark.asyncio
    async def test_text_input_previous_item_id_chain(self, server_env):
        """Sequential text items chain via previous_item_id."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)

            await conn.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "msg_a",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "first"}],
                    },
                }
            )
            e1 = await _recv(conn)
            assert e1.previous_item_id is None

            await conn.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "msg_b",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "second"}],
                    },
                }
            )
            e2 = await _recv(conn)
            assert e2.previous_item_id == e1.item.id


# ===================================================================
# 7. Error handling
# ===================================================================


class TestSDKErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_event_returns_error(self, server_env):
        """Unknown event type returns an error event."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)

            await conn.send({"type": "bogus.nonexistent", "event_id": "client_bogus_1"})
            event = await _recv(conn)
            assert event.type == ERROR
            assert event.error is not None
            assert event.error.event_id == "client_bogus_1"

    @pytest.mark.asyncio
    async def test_duplicate_response_create_error(self, server_env):
        """response.create while response is active returns error."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)

            server_env.output_queue.put(_pcm_bytes(256))
            await _recv(conn)  # response.created
            await _recv(conn)  # audio delta

            await conn.send({"type": "response.create", "event_id": "client_create_1"})
            event = await _recv(conn)
            assert event.type == ERROR
            assert event.error.type == "conversation_already_has_active_response"
            assert event.error.event_id == "client_create_1"
            assert event.event_id != "client_create_1"


# ===================================================================
# 8. Response cancel
# ===================================================================


class TestSDKResponseCancel:
    @pytest.mark.asyncio
    async def test_cancel_active_response(self, server_env):
        """response.cancel produces done events with cancelled status."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)

            server_env.output_queue.put(_pcm_bytes(256))
            await _recv(conn)  # response.created
            await _recv(conn)  # audio delta

            await conn.send({"type": "response.cancel"})

            event = await _recv(conn)
            assert event.type == AUDIO_DONE

            event = await _recv(conn)
            assert event.type == RESPONSE_DONE
            assert event.response.status == "cancelled"
            assert event.response.status_details.reason == "client_cancelled"


# ===================================================================
# 9. Multi-turn conversation_id consistency
# ===================================================================


class TestSDKMultiTurn:
    @pytest.mark.asyncio
    async def test_two_turns_same_conversation(self, server_env):
        """Two voice turns share the same conversation_id."""
        client = server_env.make_client()
        async with client.realtime.connect(model="test") as conn:
            await _recv(conn)  # session.created

            # Turn 1
            server_env.text_output_queue.put(SpeechStartedEvent())
            await _recv(conn)

            server_env.text_output_queue.put(SpeechStoppedEvent())
            await _recv(conn)

            server_env.text_output_queue.put(TranscriptionCompletedEvent(transcript="hi"))
            await _recv(conn)

            server_env.output_queue.put(_pcm_bytes(128))
            t1_created = await _recv(conn)
            assert t1_created.type == RESPONSE_CREATED
            await _recv(conn)  # audio delta

            # Barge-in
            server_env.text_output_queue.put(SpeechStartedEvent())
            events = []
            for _ in range(3):
                events.append(await _recv(conn))

            t1_done = next(e for e in events if e.type == RESPONSE_DONE)

            # Simulate pipeline acknowledging cancellation so discard guard clears
            server_env.output_queue.put(AUDIO_RESPONSE_DONE)
            await asyncio.sleep(0.15)

            # Turn 2
            server_env.text_output_queue.put(SpeechStoppedEvent())
            await _recv(conn)

            server_env.text_output_queue.put(TranscriptionCompletedEvent(transcript="bye"))
            await _recv(conn)

            server_env.output_queue.put(_pcm_bytes(128))
            t2_created = await _recv(conn)
            assert t2_created.type == RESPONSE_CREATED
            await _recv(conn)  # audio delta

            server_env.output_queue.put(PIPELINE_END)
            await _recv(conn)  # audio done
            t2_done = await _recv(conn)
            assert t2_done.type == RESPONSE_DONE

            assert t1_done.response.conversation_id == t2_done.response.conversation_id
