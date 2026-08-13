import asyncio
import json
import signal
import sys
from threading import Event
from types import SimpleNamespace

import pytest

import speech_to_speech.api.openai_realtime.audio_client as audio_client_module
from speech_to_speech.api.openai_realtime.audio_client import (
    PlaybackBuffer,
    RealtimeAudioClientConfig,
    ToolResult,
    _FriendlyEventRenderer,
    _ToolCallCoordinator,
    _ToolCoordinatorError,
    build_session_update,
    handle_server_event,
    load_realtime_tool_module,
    normalize_realtime_url,
    run_realtime_audio_client,
)

TOOL_DEFINITION = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a value.",
    "parameters": {"type": "object", "properties": {"index": {"type": "integer"}}},
}


async def noop_tool_executor(_name, _arguments):
    return None


async def done_tool_executor(_name, _arguments):
    return "done"


class RecordingConnection:
    def __init__(self):
        self.sent = []

    async def send(self, event):
        self.sent.append(event)


async def wait_until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


def response_created(response_id, *, metadata=None):
    return SimpleNamespace(
        type="response.created",
        response=SimpleNamespace(id=response_id, metadata=metadata or {}),
    )


def tool_call(call_id, *, response_id="response_1", output_index=0, name="lookup", arguments="{}"):
    return SimpleNamespace(
        type="response.function_call_arguments.done",
        response_id=response_id,
        output_index=output_index,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def output_item_added(call_id, *, response_id="response_1", output_index=0, name="lookup"):
    return SimpleNamespace(
        type="response.output_item.added",
        response_id=response_id,
        output_index=output_index,
        item=function_call(call_id, name=name, arguments=""),
    )


def function_call(call_id, *, name="lookup", arguments="{}"):
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def response_done(response_id="response_1", status="completed", output=()):
    return SimpleNamespace(
        type="response.done",
        response=SimpleNamespace(id=response_id, status=status, output=list(output)),
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "ws://127.0.0.1:8765/v1/realtime",
            ("http://127.0.0.1:8765/v1", "ws://127.0.0.1:8765/v1"),
        ),
        (
            "https://voice.example/openai/v1/realtime/",
            ("https://voice.example/openai/v1", "wss://voice.example/openai/v1"),
        ),
    ],
)
def test_full_realtime_url_is_normalized_for_openai_sdk(url, expected):
    assert normalize_realtime_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "127.0.0.1:8765/v1/realtime",
        "ws://127.0.0.1:8765/v1",
        "ws://127.0.0.1:8765/v1/realtime?token=secret",
    ],
)
def test_realtime_url_rejects_noncanonical_endpoints(url):
    with pytest.raises(ValueError, match="--url"):
        normalize_realtime_url(url)


def test_audio_client_api_key_precedence_and_loopback_fallback(monkeypatch):
    client_kwargs = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            client_kwargs.append(kwargs)

    monkeypatch.setattr(audio_client_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    audio_client_module._make_client(RealtimeAudioClientConfig())
    audio_client_module._make_client(RealtimeAudioClientConfig(url="ws://voice.example/v1/realtime"))

    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    audio_client_module._make_client(RealtimeAudioClientConfig(url="wss://voice.example/v1/realtime"))
    audio_client_module._make_client(RealtimeAudioClientConfig())
    audio_client_module._make_client(RealtimeAudioClientConfig(api_key="explicit-secret"))

    assert client_kwargs[0]["api_key"] == "local"
    assert "api_key" not in client_kwargs[1]
    assert "api_key" not in client_kwargs[2]
    assert client_kwargs[3]["api_key"] == "local"
    assert client_kwargs[4]["api_key"] == "explicit-secret"


def test_audio_client_sends_realtime_session_configuration():
    event = build_session_update(
        RealtimeAudioClientConfig(
            instructions="Be concise",
            voice="alloy",
        )
    )

    assert event == {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "Be concise",
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": True,
                    }
                },
                "output": {"voice": "alloy"},
            },
        },
    }


def test_audio_client_advertises_only_configured_tools():
    config = RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=noop_tool_executor)

    session = build_session_update(config)["session"]

    assert session["tools"] == [TOOL_DEFINITION]
    assert session["tool_choice"] == "auto"


def test_audio_client_rejects_tools_without_an_executor():
    with pytest.raises(ValueError, match="tool_executor"):
        build_session_update(RealtimeAudioClientConfig(tools=[TOOL_DEFINITION]))


def test_audio_client_defers_executor_awaitable_validation_until_invocation():
    session = build_session_update(
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=lambda _name, _arguments: None)
    )["session"]

    assert session["tools"] == [TOOL_DEFINITION]


def test_audio_client_loads_explicit_tool_module_contract(monkeypatch):
    async def executor(_name, _arguments):
        return None

    monkeypatch.setitem(
        sys.modules,
        "test_voice_tools",
        SimpleNamespace(TOOLS=[TOOL_DEFINITION], execute_tool=executor, CREATE_RESPONSE=False),
    )

    tools, loaded_executor, create_response = load_realtime_tool_module("test_voice_tools")

    assert tools == [TOOL_DEFINITION]
    assert loaded_executor is executor
    assert create_response is False


@pytest.mark.parametrize("rate", [8000, 44100, 48000])
def test_audio_client_rejects_unsupported_explicit_pcm_rates(rate):
    with pytest.raises(ValueError, match="Unsupported rate"):
        build_session_update(RealtimeAudioClientConfig(send_rate=rate))


def test_audio_client_clears_unplayed_audio_on_barge_in(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()
    playback.append(b"\x01\x02" * 100)

    handle_server_event(
        SimpleNamespace(type="input_audio_buffer.speech_started"),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )

    assert playback.buffered_bytes == 0
    assert not playback.is_active()
    capsys.readouterr()


def test_audio_client_does_not_allocate_transcript_state_for_direct_audio_turns(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for index in range(130):
        handle_server_event(
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id=f"item_{index}"),
            playback=playback,
            renderer=renderer,
            print_json=False,
        )

    assert renderer.user_transcript_by_item == {}
    capsys.readouterr()


def test_audio_client_clears_unplayed_audio_when_response_is_cancelled(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()
    playback.append(b"\x01\x02" * 100)

    handle_server_event(
        SimpleNamespace(type="response.done", response=SimpleNamespace(status="cancelled")),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )

    assert playback.buffered_bytes == 0
    assert not playback.is_active()
    capsys.readouterr()


def test_audio_client_streams_assistant_transcript_without_reprinting_done(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for event in (
        SimpleNamespace(type="response.output_audio_transcript.delta", delta="Hello"),
        SimpleNamespace(type="response.output_audio_transcript.delta", delta=" there."),
        SimpleNamespace(type="response.output_audio_transcript.done", transcript="Hello there."),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert capsys.readouterr().out == "ASSISTANT: Hello there.\n"


def test_audio_client_tracks_interleaved_transcripts_per_output_item(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    def transcript_event(type, *, item_id, output_index, delta=None, transcript=None):
        return SimpleNamespace(
            type=type,
            response_id="response_1",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            delta=delta,
            transcript=transcript,
        )

    for event in (
        transcript_event("response.output_audio_transcript.delta", item_id="item_a", output_index=0, delta="first"),
        transcript_event("response.output_audio_transcript.delta", item_id="item_b", output_index=1, delta="second"),
        transcript_event("response.output_audio_transcript.done", item_id="item_a", output_index=0, transcript="first"),
        transcript_event(
            "response.output_audio_transcript.done", item_id="item_b", output_index=1, transcript="second"
        ),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert capsys.readouterr().out == "ASSISTANT: first\nASSISTANT: second\n"


def test_audio_client_separates_done_only_transcript_from_live_stream(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    def transcript_event(type, *, item_id, delta=None, transcript=None):
        return SimpleNamespace(
            type=type,
            response_id="response_1",
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta=delta,
            transcript=transcript,
        )

    for event in (
        transcript_event("response.output_audio_transcript.delta", item_id="item_b", delta="second"),
        transcript_event("response.output_audio_transcript.done", item_id="item_a", transcript="legacy first"),
        transcript_event("response.output_audio_transcript.done", item_id="item_b", transcript="second"),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert capsys.readouterr().out == "ASSISTANT: second\nASSISTANT: legacy first\n"


def test_audio_client_separates_alternating_assistant_and_user_partial_text(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    def assistant_event(type, *, delta=None, transcript=None):
        return SimpleNamespace(
            type=type,
            response_id="response_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            delta=delta,
            transcript=transcript,
        )

    for event in (
        assistant_event("response.output_audio_transcript.delta", delta="assistant"),
        SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta="user partial"),
        assistant_event("response.output_audio_transcript.delta", delta="continues"),
        assistant_event("response.output_audio_transcript.done", transcript="assistant continues"),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    user_line = "USER: user partial"
    assert capsys.readouterr().out == (
        f"ASSISTANT: assistant\n\r{user_line}\r{' ' * len(user_line)}\rASSISTANT: continues\n"
    )


def test_audio_client_accumulates_incremental_user_transcription_deltas(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for event in (
        SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item_1"),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_1",
            delta="user",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_1",
            delta=" partial",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            item_id="item_1",
            transcript="user partial",
        ),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert renderer.user_transcript_by_item == {}
    assert "USER: user partial" in capsys.readouterr().out


def test_audio_client_tracks_overlapping_user_transcriptions_by_item(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for event in (
        SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item_1"),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_1",
            delta="hel",
        ),
        SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item_2"),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_2",
            delta="wor",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_1",
            delta="lo",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            item_id="item_1",
            transcript="hello",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_2",
            delta="ld",
        ),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert renderer.user_transcript_by_item == {"item_2": "world"}
    output = capsys.readouterr().out
    assert "USER: hello" in output
    assert "USER: world" in output


def test_audio_client_retains_unterminated_transcripts_until_completion(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for index in range(130):
        handle_server_event(
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta",
                item_id=f"item_{index}",
                delta=str(index),
            ),
            playback=playback,
            renderer=renderer,
            print_json=False,
        )

    assert len(renderer.user_transcript_by_item) == 130
    assert renderer.user_transcript_by_item["item_0"] == "0"
    assert renderer.user_transcript_by_item["item_1"] == "1"
    assert renderer.user_transcript_by_item["item_129"] == "129"

    handle_server_event(
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_0",
            delta=" more",
        ),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    assert renderer.user_transcript_by_item["item_0"] == "0 more"

    handle_server_event(
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            item_id="item_0",
            transcript="",
        ),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    assert "item_0" not in renderer.user_transcript_by_item
    capsys.readouterr()


def test_audio_client_response_done_preserves_other_response_transcripts(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    def transcript_event(type, *, response_id, delta=None, transcript=None):
        return SimpleNamespace(
            type=type,
            response_id=response_id,
            item_id=f"item_{response_id}",
            output_index=0,
            content_index=0,
            delta=delta,
            transcript=transcript,
        )

    for event in (
        transcript_event("response.output_audio_transcript.delta", response_id="response_a", delta="first"),
        transcript_event("response.output_audio_transcript.delta", response_id="response_b", delta="second"),
        transcript_event("response.output_audio_transcript.done", response_id="response_a", transcript="first"),
        SimpleNamespace(type="response.done", response=SimpleNamespace(id="response_a", status="completed")),
        transcript_event("response.output_audio_transcript.done", response_id="response_b", transcript="second"),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert capsys.readouterr().out == ("ASSISTANT: first\nASSISTANT: second\nASSISTANT: <response completed>\n")


def test_audio_client_prints_transcript_from_legacy_done_only_server(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    handle_server_event(
        SimpleNamespace(type="response.output_audio_transcript.done", transcript="Hello there."),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )

    assert capsys.readouterr().out == "ASSISTANT: Hello there.\n"


def test_audio_client_keeps_tool_event_off_live_transcript_line(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for event in (
        SimpleNamespace(type="response.output_audio_transcript.delta", delta="Checking."),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            name="lookup",
            call_id="call_1",
            arguments="{}",
        ),
        SimpleNamespace(type="response.output_audio_transcript.done", transcript="Checking."),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert capsys.readouterr().out == "ASSISTANT: Checking.\nTOOL: lookup call_id=call_1 arguments={}\n"


async def test_audio_client_executes_tools_from_completed_response_output():
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        return "done"

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))

    await wait_until(lambda: len(conn.sent) == 2)

    assert calls == [("lookup", {})]
    assert [event["type"] for event in conn.sent] == ["conversation.item.create", "response.create"]
    await coordinator.close()


async def test_audio_client_executes_multiple_tools_once_and_flushes_in_response_output_order():
    gates = [asyncio.Event(), asyncio.Event()]
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        await gates[arguments["index"]].wait()
        return {"result": arguments["index"]}

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )
    coordinator.handle_event(response_created("response_1"))
    coordinator.handle_event(
        response_done(
            output=[
                function_call("call_1", arguments='{"index": 0}'),
                function_call("call_2", arguments='{"index": 1}'),
            ]
        )
    )

    await wait_until(lambda: len(calls) == 2)
    gates[1].set()
    await asyncio.sleep(0)
    assert conn.sent == []
    gates[0].set()
    await wait_until(lambda: len(conn.sent) == 3)

    assert calls == [("lookup", {"index": 0}), ("lookup", {"index": 1})]
    assert [event["type"] for event in conn.sent] == [
        "conversation.item.create",
        "conversation.item.create",
        "response.create",
    ]
    assert [event["item"]["call_id"] for event in conn.sent[:2]] == ["call_1", "call_2"]
    assert [json.loads(event["item"]["output"]) for event in conn.sent[:2]] == [
        {"result": 0},
        {"result": 1},
    ]
    await coordinator.close()


async def test_audio_client_executes_immediately_but_delivers_in_output_index_order():
    gates = {"call_0": asyncio.Event(), "call_1": asyncio.Event()}
    started = []

    async def executor(_name, arguments):
        call_id = f"call_{arguments['index']}"
        started.append(call_id)
        await gates[call_id].wait()
        return {"result": arguments["index"]}

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )
    coordinator.handle_event(output_item_added("call_0", output_index=0))
    coordinator.handle_event(output_item_added("call_1", output_index=1))
    coordinator.handle_event(tool_call("call_1", output_index=1, arguments='{"index": 1}'))
    coordinator.handle_event(tool_call("call_0", output_index=0, arguments='{"index": 0}'))
    await wait_until(lambda: len(started) == 2)

    gates["call_1"].set()
    await asyncio.sleep(0)
    assert conn.sent == []
    gates["call_0"].set()
    await wait_until(lambda: len(conn.sent) == 2)

    assert [event["item"]["call_id"] for event in conn.sent] == ["call_0", "call_1"]
    assert all(event["type"] == "conversation.item.create" for event in conn.sent)

    coordinator.handle_event(response_done(output=[function_call("call_0"), function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 3)

    assert conn.sent[-1]["type"] == "response.create"
    await coordinator.close()


async def test_audio_client_uses_terminal_output_order_when_item_added_is_absent():
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )
    coordinator.handle_event(tool_call("call_1", output_index=1))
    coordinator.handle_event(tool_call("call_0", output_index=0))
    await wait_until(lambda: len(coordinator._tool_batches["response_1"].results) == 2)

    assert conn.sent == []

    coordinator.handle_event(response_done(output=[function_call("call_0"), function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 3)

    assert [event["item"]["call_id"] for event in conn.sent[:2]] == ["call_0", "call_1"]
    assert conn.sent[-1]["type"] == "response.create"
    await coordinator.close()


async def test_audio_client_cancellation_during_blocked_delivery_releases_counters_once():
    class BlockingConnection(RecordingConnection):
        def __init__(self):
            super().__init__()
            self.send_started = asyncio.Event()
            self.release_send = asyncio.Event()
            self.block_next_output = True

        async def send(self, event):
            if event["type"] == "conversation.item.create" and self.block_next_output:
                self.block_next_output = False
                self.send_started.set()
                await self.release_send.wait()
            self.sent.append(event)

    conn = BlockingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )
    coordinator.handle_event(output_item_added("call_cancelled", output_index=0))
    coordinator.handle_event(tool_call("call_cancelled", output_index=0))
    await asyncio.wait_for(conn.send_started.wait(), timeout=1.0)
    cancelled_batch = coordinator._tool_batches["response_1"]

    coordinator.handle_event(response_done(status="cancelled"))
    assert cancelled_batch.pending_deliveries == 0
    assert coordinator._pending_tool_flushes == 0

    conn.release_send.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert cancelled_batch.pending_deliveries == 0
    assert coordinator._pending_tool_flushes == 0

    coordinator.handle_event(output_item_added("call_next", response_id="response_2", output_index=0))
    coordinator.handle_event(tool_call("call_next", response_id="response_2", output_index=0))
    coordinator.handle_event(response_done(response_id="response_2", output=[function_call("call_next")]))
    await wait_until(lambda: any(event["type"] == "response.create" for event in conn.sent))

    assert coordinator._pending_tool_flushes == 0
    await coordinator.close()


async def test_audio_client_uses_per_result_follow_up_policy_for_mixed_batch():
    async def executor(_name, arguments):
        return ToolResult(
            {"result": arguments["index"]},
            create_response=arguments["index"] == 1,
        )

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(
            tools=[TOOL_DEFINITION],
            tool_executor=executor,
            tool_response_create=False,
        ),
    )
    coordinator.handle_event(
        response_done(
            output=[
                function_call("call_1", arguments='{"index": 0}'),
                function_call("call_2", arguments='{"index": 1}'),
            ]
        )
    )

    await wait_until(lambda: len(conn.sent) == 3)

    assert [event["type"] for event in conn.sent] == [
        "conversation.item.create",
        "conversation.item.create",
        "response.create",
    ]
    assert conn.sent[-1]["event_id"] == "tool_1"
    await coordinator.close()


async def test_audio_client_per_result_policy_can_disable_all_follow_ups():
    async def executor(_name, _arguments):
        return ToolResult("done", create_response=False)

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))

    await wait_until(lambda: len(conn.sent) == 1)

    assert conn.sent[0]["type"] == "conversation.item.create"
    await coordinator.close()


@pytest.mark.parametrize("executor_kind", ["async-call", "decorated"])
async def test_audio_client_accepts_callable_executors_returning_awaitables(executor_kind):
    calls = []

    if executor_kind == "async-call":

        class AsyncCallable:
            async def __call__(self, name, arguments):
                calls.append((name, arguments))
                return "done"

        executor = AsyncCallable()
    else:

        async def async_handler(name, arguments):
            calls.append((name, arguments))
            return "done"

        def executor(name, arguments):
            return async_handler(name, arguments)

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))

    await wait_until(lambda: len(conn.sent) == 2)

    assert calls == [("lookup", {})]
    assert conn.sent[0]["item"]["output"] == "done"
    await coordinator.close()


async def test_audio_client_returns_error_when_executor_result_is_not_awaitable(capsys):
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(
            tools=[TOOL_DEFINITION],
            tool_executor=lambda _name, _arguments: "not-awaitable",
        ),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))

    await wait_until(lambda: len(conn.sent) == 2)

    output = json.loads(conn.sent[0]["item"]["output"])
    assert "must return an awaitable" in output["error"]
    assert "must return an awaitable" in capsys.readouterr().out
    await coordinator.close()


async def test_audio_client_returns_unknown_malformed_and_handler_failures_and_forces_recovery(capsys):
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        raise RuntimeError("lookup failed")

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(
            tools=[TOOL_DEFINITION],
            tool_executor=executor,
            tool_response_create=False,
        ),
    )
    coordinator.handle_event(response_created("response_1"))
    coordinator.handle_event(
        response_done(
            output=[
                function_call("call_unknown", name="not_declared"),
                function_call("call_malformed", arguments="{"),
                function_call("call_failed"),
            ]
        )
    )

    await wait_until(lambda: len(conn.sent) == 4)

    assert calls == [("lookup", {})]
    outputs = [json.loads(event["item"]["output"]) for event in conn.sent[:3]]
    assert all("error" in output for output in outputs)
    errors = capsys.readouterr().out
    assert "unknown tool" in errors
    assert "not valid JSON" in errors
    assert "lookup failed" in errors
    await coordinator.close()


async def test_audio_client_validates_arguments_against_declared_schema_and_forces_recovery(capsys):
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        return "ok"

    strict_tool = {
        **TOOL_DEFINITION,
        "parameters": {
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
            "additionalProperties": False,
        },
    }
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(
            tools=[strict_tool],
            tool_executor=executor,
            tool_response_create=False,
        ),
    )
    coordinator.handle_event(
        response_done(output=[function_call("call_1", arguments='{"index": "wrong", "extra": true}')])
    )

    await wait_until(lambda: len(conn.sent) == 2)

    assert calls == []
    output = json.loads(conn.sent[0]["item"]["output"])
    assert "arguments do not match the declared schema" in output["error"]
    assert "arguments do not match the declared schema" in capsys.readouterr().out
    await coordinator.close()


async def test_audio_client_ignores_repeated_argument_events_and_uses_terminal_output_once():
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        return "ok"

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(
            tools=[TOOL_DEFINITION],
            tool_executor=executor,
            tool_response_create=False,
        ),
    )
    coordinator.handle_event(tool_call("call_1"))
    coordinator.handle_event(tool_call("call_1"))
    coordinator.handle_event(response_done(output=[function_call("call_1")]))

    await wait_until(lambda: len(conn.sent) == 1)

    coordinator.handle_event(response_created("response_2"))
    coordinator.handle_event(tool_call("call_1", response_id="response_2"))
    coordinator.handle_event(response_done("response_2", output=[function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 2)

    assert calls == [("lookup", {}), ("lookup", {})]
    await coordinator.close()


@pytest.mark.parametrize("status", ["cancelled", "incomplete"])
async def test_audio_client_does_not_execute_tools_from_unsuccessful_responses(status):
    calls = []
    release = asyncio.Event()

    async def executor(name, arguments):
        calls.append((name, arguments))
        await release.wait()
        return "done"

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )
    coordinator.handle_event(tool_call("call_1"))
    await wait_until(lambda: calls == [("lookup", {})])

    coordinator.handle_event(response_done(status=status, output=[function_call("call_1")]))
    await asyncio.sleep(0)
    await coordinator.close()

    assert calls == [("lookup", {})]
    assert conn.sent == []


async def test_audio_client_waits_for_an_active_response_before_tool_follow_up():
    release = asyncio.Event()

    async def executor(_name, _arguments):
        await release.wait()
        return "result"

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )
    coordinator.handle_event(response_created("response_1"))
    coordinator.handle_event(response_done(output=[function_call("call_1")]))
    coordinator.handle_event(response_created("response_2"))
    release.set()

    await wait_until(lambda: len(conn.sent) == 1)
    assert conn.sent[0]["type"] == "conversation.item.create"

    coordinator.handle_event(response_done("response_2"))
    await wait_until(lambda: len(conn.sent) == 2)
    assert conn.sent[1]["type"] == "response.create"
    await coordinator.close()


async def test_audio_client_one_follow_up_covers_all_queued_tool_outputs():
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )

    coordinator.handle_event(response_created("response_1"))
    coordinator.handle_event(response_done("response_1", output=[function_call("call_1")]))
    coordinator.handle_event(response_created("response_2"))

    await wait_until(lambda: len(conn.sent) == 1)
    assert conn.sent[0]["item"]["call_id"] == "call_1"

    coordinator.handle_event(response_done("response_2", output=[function_call("call_2")]))

    await wait_until(lambda: len(conn.sent) == 3)
    assert [event["type"] for event in conn.sent] == [
        "conversation.item.create",
        "conversation.item.create",
        "response.create",
    ]
    assert coordinator._queued_follow_ups == 2
    create_event = conn.sent[-1]

    coordinator.handle_event(
        response_created(
            "response_tool_1",
            metadata={"s2s_local_tool_create_id": create_event["event_id"]},
        )
    )
    assert coordinator._queued_follow_ups == 0

    coordinator.handle_event(response_done("response_tool_1"))
    await asyncio.sleep(0.05)
    assert len(conn.sent) == 3
    await coordinator.close()


async def test_audio_client_waits_for_all_tool_flushes_before_follow_up():
    releases = [asyncio.Event(), asyncio.Event()]
    calls = []

    async def executor(_name, arguments):
        index = arguments["index"]
        calls.append(index)
        await releases[index].wait()
        return {"result": index}

    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=executor),
    )

    coordinator.handle_event(response_created("response_1"))
    coordinator.handle_event(response_done("response_1", output=[function_call("call_1", arguments='{"index": 0}')]))
    coordinator.handle_event(response_created("response_2"))
    coordinator.handle_event(response_done("response_2", output=[function_call("call_2", arguments='{"index": 1}')]))

    await wait_until(lambda: len(calls) == 2)
    releases[0].set()
    await wait_until(lambda: len(conn.sent) >= 1)
    await asyncio.sleep(0.01)
    assert [event["type"] for event in conn.sent] == ["conversation.item.create"]

    releases[1].set()
    await wait_until(lambda: len(conn.sent) == 3)
    assert [event["type"] for event in conn.sent] == [
        "conversation.item.create",
        "conversation.item.create",
        "response.create",
    ]
    assert [event["item"]["call_id"] for event in conn.sent[:2]] == ["call_1", "call_2"]
    await coordinator.close()


async def test_audio_client_waits_for_response_lifecycle_after_follow_up_collision():
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 2)
    create_id = conn.sent[-1]["event_id"]

    coordinator.handle_event(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                type="conversation_already_has_active_response",
                code=None,
                event_id=create_id,
            ),
        )
    )
    await asyncio.sleep(0.01)
    assert len(conn.sent) == 2
    assert coordinator._queued_follow_ups == 1

    coordinator.handle_event(response_created("response_implicit"))
    coordinator.handle_event(response_done("response_implicit"))
    await wait_until(lambda: len(conn.sent) == 3)
    assert conn.sent[-1]["type"] == "response.create"
    await coordinator.close()


async def test_audio_client_retries_when_collision_arrives_after_response_finished():
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 2)
    create_id = conn.sent[-1]["event_id"]

    coordinator.handle_event(response_created("response_implicit"))
    coordinator.handle_event(response_done("response_implicit"))
    coordinator.handle_event(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                type="invalid_request_error",
                code="conversation_already_has_active_response",
                event_id=create_id,
            ),
        )
    )

    await wait_until(lambda: len(conn.sent) == 3)
    assert conn.sent[-1]["type"] == "response.create"
    await coordinator.close()


async def test_audio_client_surfaces_correlated_non_collision_follow_up_rejection():
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 2)
    create_id = conn.sent[-1]["event_id"]

    coordinator.handle_event(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                type="invalid_request_error",
                code="invalid_value",
                message="Invalid response metadata",
                event_id=create_id,
            ),
        )
    )

    with pytest.raises(_ToolCoordinatorError, match="invalid_value"):
        await coordinator.wait_for_failure()
    assert coordinator._pending_create_id is None
    await coordinator.close()


async def test_audio_client_ignores_uncorrelated_error_while_follow_up_is_pending():
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 2)
    create_event = conn.sent[-1]

    coordinator.handle_event(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                type="invalid_request_error",
                code="invalid_value",
                message="Unrelated error",
                event_id="another_client_event",
            ),
        )
    )

    assert coordinator._pending_create_id == create_event["event_id"]
    coordinator.handle_event(
        response_created(
            "response_follow_up",
            metadata={"s2s_local_tool_create_id": create_event["event_id"]},
        )
    )
    assert coordinator._pending_create_id is None
    await coordinator.close()


async def test_audio_client_can_disable_tool_follow_up_response():
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(
            tools=[TOOL_DEFINITION],
            tool_executor=done_tool_executor,
            tool_response_create=False,
        ),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))

    await wait_until(lambda: len(conn.sent) == 1)
    assert conn.sent[0]["type"] == "conversation.item.create"
    await coordinator.close()


def test_audio_client_does_not_duplicate_partial_transcript_on_cancel(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()
    playback.append(b"\x01\x02" * 100)

    for event in (
        SimpleNamespace(type="response.output_audio_transcript.delta", delta="partial"),
        SimpleNamespace(type="response.output_audio_transcript.done", transcript="partial"),
        SimpleNamespace(type="response.done", response=SimpleNamespace(status="cancelled")),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert capsys.readouterr().out == "ASSISTANT: partial\nASSISTANT: <response cancelled>\n"
    assert playback.buffered_bytes == 0


async def test_audio_streams_are_cleaned_up_when_output_start_fails(monkeypatch):
    events = []

    class FakeStream:
        def __init__(self, name, *, fail_start=False):
            self.name = name
            self.fail_start = fail_start

        def start(self):
            events.append(f"{self.name}.start")
            if self.fail_start:
                raise RuntimeError("output start failed")

        def stop(self):
            events.append(f"{self.name}.stop")

        def close(self):
            events.append(f"{self.name}.close")

    input_stream = FakeStream("input")
    output_stream = FakeStream("output", fail_start=True)
    fake_sounddevice = SimpleNamespace(
        RawInputStream=lambda **_kwargs: input_stream,
        RawOutputStream=lambda **_kwargs: output_stream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    with pytest.raises(RuntimeError, match="output start failed"):
        await audio_client_module._run_audio_session(
            SimpleNamespace(),
            RealtimeAudioClientConfig(),
            Event(),
        )

    assert events == [
        "input.start",
        "output.start",
        "input.stop",
        "output.close",
        "input.close",
    ]


async def test_open_input_stream_is_closed_when_output_construction_fails(monkeypatch):
    events = []

    class FakeInputStream:
        def close(self):
            events.append("input.close")

    def fail_output_stream(**_kwargs):
        raise RuntimeError("output construction failed")

    fake_sounddevice = SimpleNamespace(
        RawInputStream=lambda **_kwargs: FakeInputStream(),
        RawOutputStream=fail_output_stream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    with pytest.raises(RuntimeError, match="output construction failed"):
        await audio_client_module._run_audio_session(
            SimpleNamespace(),
            RealtimeAudioClientConfig(),
            Event(),
        )

    assert events == ["input.close"]


async def test_audio_client_startup_and_shutdown_use_public_realtime_connection(monkeypatch):
    sent = []
    audio_session_calls = []

    class FakeConnection:
        async def send(self, event):
            sent.append(event)

    class FakeConnectContext:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, *_args):
            return None

    class FakeClient:
        def __init__(self):
            self.realtime = SimpleNamespace(connect=lambda **_kwargs: FakeConnectContext())
            self.closed = False

        async def close(self):
            self.closed = True

    fake_client = FakeClient()
    stop_event = Event()

    async def fake_audio_session(conn, config, received_stop_event):
        audio_session_calls.append((conn, config, received_stop_event))
        received_stop_event.set()

    monkeypatch.setattr(audio_client_module, "_make_client", lambda _config: fake_client)
    monkeypatch.setattr(audio_client_module, "_run_audio_session", fake_audio_session)
    config = RealtimeAudioClientConfig()

    await audio_client_module.listen_and_play_realtime(
        config,
        stop_event=stop_event,
    )

    assert sent == [build_session_update(config)]
    assert len(audio_session_calls) == 1
    assert audio_session_calls[0][2] is stop_event
    assert fake_client.closed is True


def test_talk_client_uses_signal_driven_shutdown(monkeypatch):
    installed_handlers = {}
    restored_handlers = []
    received_stop_event = None

    def fake_getsignal(sig):
        return f"previous-{sig.name}"

    def fake_signal(sig, handler):
        if callable(handler):
            installed_handlers[sig] = handler
        else:
            restored_handlers.append((sig, handler))

    async def fake_listen(_config, *, stop_event):
        nonlocal received_stop_event
        received_stop_event = stop_event
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(audio_client_module.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(audio_client_module.signal, "signal", fake_signal)
    monkeypatch.setattr(audio_client_module, "listen_and_play_realtime", fake_listen)

    run_realtime_audio_client(RealtimeAudioClientConfig())

    assert received_stop_event is not None and received_stop_event.is_set()
    assert set(installed_handlers) == {signal.SIGINT, signal.SIGTERM}
    assert restored_handlers == [
        (signal.SIGINT, "previous-SIGINT"),
        (signal.SIGTERM, "previous-SIGTERM"),
    ]
