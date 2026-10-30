import asyncio
import json
import logging
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
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
from speech_to_speech.api.openai_realtime.echo_canceller import EchoCanceller
from speech_to_speech.api.openai_realtime.tool_followup import (
    TOOL_FOLLOWUP_COVERED,
    TOOL_FOLLOWUP_METADATA_KEY,
    TOOL_INPUT_METADATA_KEY,
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


def response_done(response_id="response_1", status="completed", output=(), metadata=None):
    return SimpleNamespace(
        type="response.done",
        response=SimpleNamespace(id=response_id, status=status, output=list(output), metadata=metadata),
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
            "extensions": ["speech_to_speech.input_audio_transcription.snapshot"],
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


def test_playback_buffer_waits_for_startup_audio_before_playing():
    playback = PlaybackBuffer(1000, startup_buffer_ms=100)
    playback.append(b"\x01" * 100)
    first_callback = bytearray(100)

    playback.write(first_callback)

    assert first_callback == b"\x00" * 100
    assert playback.buffered_bytes == 100

    playback.append(b"\x02" * 100)
    second_callback = bytearray(100)
    playback.write(second_callback)

    assert second_callback == b"\x01" * 100
    assert playback.buffered_bytes == 100


def test_playback_buffer_flushes_completed_audio_shorter_than_startup_buffer():
    playback = PlaybackBuffer(1000, startup_buffer_ms=100)
    playback.append(b"\x03" * 100)
    playback.finish()
    callback = bytearray(200)

    playback.write(callback)

    assert callback[:100] == b"\x03" * 100
    assert callback[100:] == b"\x00" * 100
    assert playback.buffered_bytes == 0


def test_playback_buffer_starts_immediately_by_default():
    playback = PlaybackBuffer(1000)
    playback.append(b"\x04" * 100)
    callback = bytearray(100)

    playback.write(callback)

    assert callback == b"\x04" * 100
    assert playback.buffered_bytes == 0


def _reference_speech(np, rate):
    import soundfile
    import soxr

    speech, source_rate = soundfile.read(
        Path(__file__).parents[2] / "src/speech_to_speech/TTS/ref_audio.wav", dtype="float32"
    )
    return (soxr.resample(speech.mean(axis=1), source_rate, rate)[: 12 * rate] * 12000).astype(np.int16)


@pytest.mark.parametrize("mode", ["echo", "overlap", "no_playback"])
@pytest.mark.parametrize("block,device_delay_ms", [(1024, 256), (8192, 1000)])
@pytest.mark.parametrize("rate", [16000, 24000])
def test_echo_canceller_removes_played_audio_from_microphone_blocks(mode, block, device_delay_ms, rate):
    np = pytest.importorskip("numpy")
    from scipy.signal import correlate, correlation_lags

    played = _reference_speech(np, rate)
    echo = np.zeros_like(played)
    delay = rate * device_delay_ms // 1000 + 320
    echo[delay:] = (played[:-delay] * 0.3).astype(np.int16)
    near = np.zeros_like(played)
    if mode == "overlap":
        near[6 * rate : 8 * rate] = played[10 * rate : 12 * rate]
    if mode == "no_playback":
        echo[:] = 0
        near = played.copy()
    microphone = (echo.astype(np.int32) + near).astype(np.int16)
    canceller = EchoCanceller(rate, rate)
    cleaned = bytearray()
    for start in range(0, len(played), block):
        if mode != "no_playback":
            jitter = 0.002 * (start // block % 2) if block == 1024 else 0
            canceller.render(played[start : start + block].tobytes(), (start + delay - 320) / rate + jitter)
        cleaned += canceller.capture(microphone[start : start + block].tobytes(), start / rate)

    assert len(cleaned) == len(played) // (rate // 100) * (rate // 100) * 2
    output = np.frombuffer(bytes(cleaned), dtype=np.int16).astype(float)
    if mode != "echo":
        # The binding's high-pass filter and processing delay change phase.
        # Require the near speech to survive, allowing up to 10 ms alignment.
        actual = output[6 * rate : 8 * rate]
        expected = near[6 * rate : 8 * rate].astype(float)
        scores = correlate(actual, expected, method="fft")
        lags = correlation_lags(len(actual), len(expected))
        correlation = scores[np.abs(lags) < rate // 100].max() / (np.linalg.norm(actual) * np.linalg.norm(expected))
        assert correlation > 0.6
        assert np.linalg.norm(actual) > 0.5 * np.linalg.norm(expected)
    else:
        tail = slice(8 * rate, len(output))
        reduction_db = 10 * np.log10(np.mean(echo[tail].astype(float) ** 2) / (np.mean(output[tail] ** 2) + 1e-9))
        assert reduction_db > 30


@pytest.mark.parametrize("case", ["late_microphone_start", "lost_microphone_block", "unreported_microphone_gap"])
def test_echo_canceller_realigns_after_microphone_discontinuity(case):
    np = pytest.importorskip("numpy")

    rate, block = 16000, 1024
    played = _reference_speech(np, rate)
    echo = np.zeros_like(played)
    echo[320:] = (played[:-320] * 0.3).astype(np.int16)
    # Silence the speaker played before the first microphone callback.
    skew = 5 * block if case == "late_microphone_start" else 0
    speaker = np.concatenate([np.zeros(skew, dtype=np.int16), played])
    lost = 4 * rate // block if case != "late_microphone_start" else -2
    canceller = EchoCanceller(rate, rate)
    for start in range(0, skew + block, block):
        canceller.render(speaker[start : start + block].tobytes(), start / rate)
    cleaned = bytearray()
    for index, start in enumerate(range(0, len(played), block)):
        canceller.render(
            speaker[start + skew + block : start + skew + 2 * block].tobytes(),
            (start + skew + block) / rate,
        )
        if index == lost:
            continue
        cleaned += canceller.capture(
            echo[start : start + block].tobytes(),
            (start + skew) / rate,
            lost_input=case == "lost_microphone_block" and index == lost + 1,
        )

    output = np.frombuffer(bytes(cleaned), dtype=np.int16).astype(float)
    tail = 4 * rate
    reduction_db = 10 * np.log10(np.mean(echo[-tail:].astype(float) ** 2) / (np.mean(output[-tail:] ** 2) + 1e-9))
    assert reduction_db > 30


def test_echo_canceller_rejects_unsupported_rates():
    with pytest.raises(ValueError, match="22050 Hz"):
        EchoCanceller(22050, 16000)
    with pytest.raises(ValueError, match="matching"):
        EchoCanceller(16000, 48000)


def test_echo_canceller_passes_raw_audio_after_processor_failure(caplog):
    calls = []

    def fail(near, far):
        calls.append((len(near), len(far)))
        raise RuntimeError("native failure")

    canceller = EchoCanceller(16000, 16000)
    canceller._apm = SimpleNamespace(process=fail)
    block = bytes(range(256)) * 8

    canceller.render(block, playback_time_s=0)
    assert canceller.capture(block, capture_time_s=0) == block
    assert canceller.capture(block, capture_time_s=1024 / 16000) == block
    assert calls == [(960, 960)]
    assert [record.getMessage() for record in caplog.records if record.levelname == "ERROR"] == [
        "Echo cancellation failed; sending raw microphone audio for the rest of the session"
    ]


@pytest.mark.parametrize("buffer_ms", [-1, float("inf"), float("nan")])
def test_audio_client_rejects_invalid_playback_buffer(buffer_ms):
    with pytest.raises(ValueError, match="playback_buffer_ms"):
        RealtimeAudioClientConfig(playback_buffer_ms=buffer_ms)


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


def test_conversation_only_renders_one_exchange_and_keeps_playback_and_barge_in(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer(conversation_only=True)
    for event in (
        SimpleNamespace(type="session.created"),
        SimpleNamespace(type="input_audio_buffer.speech_started"),
        SimpleNamespace(type="speech_to_speech.input_audio_transcription.snapshot", transcript="How"),
        SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta="How are you?"),
        SimpleNamespace(type="input_audio_buffer.committed"),
        SimpleNamespace(type="conversation.item.created"),
        SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="How are you?"),
        response_created("response_1"),
        SimpleNamespace(type="response.content_part.added"),
        SimpleNamespace(type="response.output_audio_transcript.delta", delta="I'm"),
        SimpleNamespace(type="speech_to_speech.input_audio_transcription.snapshot", transcript="background"),
        SimpleNamespace(type="rate_limits.updated"),
        tool_call("call_1"),
        SimpleNamespace(type="response.output_audio_transcript.delta", delta=" doing well."),
        SimpleNamespace(type="response.output_audio_transcript.done", transcript="I'm doing well."),
        SimpleNamespace(type="response.output_audio.delta", delta="AQI="),
        SimpleNamespace(type="response.output_audio.done"),
        SimpleNamespace(type="response.content_part.done"),
        response_done(),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert capsys.readouterr().out == "USER: How are you?\nASSISTANT: I'm doing well.\n"
    assert playback.buffered_bytes == 2
    handle_server_event(
        SimpleNamespace(type="input_audio_buffer.speech_started"),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    assert playback.buffered_bytes == 0


def test_conversation_only_keeps_errors_visible(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer(conversation_only=True)
    for event in (
        SimpleNamespace(type="error", error=SimpleNamespace(type="server_error", message="Try again")),
        response_done(status="failed"),
        SimpleNamespace(type="conversation.item.input_audio_transcription.failed", item_id="item_1", error=None),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)
    assert capsys.readouterr().out == (
        "ERROR: server_error: Try again\n"
        "ERROR: response failed\n"
        "ERROR: transcription_error: Input audio transcription failed\n"
    )


@pytest.mark.parametrize(
    "settings,conversation_only",
    [
        ({}, True),
        ({"verbose": True}, False),
        ({"print_json": True}, False),
        ({"verbose": True, "print_json": True}, False),
    ],
)
def test_audio_client_defaults_to_clean_output_with_explicit_diagnostics(settings, conversation_only):
    assert RealtimeAudioClientConfig(**settings).conversation_only is conversation_only


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


def test_audio_client_renders_snapshots_and_replaces_on_completion(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for event in (
        SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item_1"),
        SimpleNamespace(
            type="speech_to_speech.input_audio_transcription.snapshot",
            item_id="item_1",
            transcript="hello brave",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_1",
            delta="hello",
        ),
        SimpleNamespace(
            type="speech_to_speech.input_audio_transcription.snapshot",
            item_id="item_1",
            transcript="hello brave new",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            item_id="item_1",
            transcript="hello brave new world",
        ),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert renderer.user_transcript_by_item == {}
    assert renderer.user_snapshot_by_item == {}
    out = capsys.readouterr().out
    assert "USER: hello brave new world" in out


def test_audio_client_clears_snapshots_and_deltas_on_transcription_failed(capsys):
    playback = PlaybackBuffer(16000)
    renderer = _FriendlyEventRenderer()

    for event in (
        SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item_1"),
        SimpleNamespace(
            type="speech_to_speech.input_audio_transcription.snapshot",
            item_id="item_1",
            transcript="hello brave",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta",
            item_id="item_1",
            delta="hello",
        ),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.failed",
            item_id="item_1",
            error=SimpleNamespace(
                type="transcription_error",
                message="transcription timed out",
            ),
        ),
    ):
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    assert renderer.user_transcript_by_item == {}
    assert renderer.user_snapshot_by_item == {}
    assert renderer.live_user_width == 0
    out = capsys.readouterr().out
    assert "ERROR: transcription_error: transcription timed out" in out


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


async def test_audio_client_returns_unknown_malformed_and_handler_failures_and_forces_recovery(capsys, caplog):
    caplog.set_level(logging.DEBUG)
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        raise RuntimeError("executor-secret-8675309")

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
    assert "executor-secret-8675309" in errors
    assert "executor-secret-8675309" not in "\n".join(record.getMessage() for record in caplog.records)
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


@pytest.mark.parametrize(
    "turn_end",
    ["answered", "missed_output", "empty_answer", "cancelled_answer", "refused_before_output", "refused_after_output"],
)
async def test_audio_client_holds_tool_follow_up_until_user_turn_ends(turn_end):
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
    coordinator.handle_event(SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item_2"))

    def refuse():
        # The model refuses a question asked while the call has no output.
        # The turn stays open, and speech can resume without another start.
        coordinator.handle_event(response_created("response_refused"))
        coordinator.handle_event(response_done("response_refused", status="failed"))

    if turn_end == "refused_before_output":
        refuse()
    release.set()
    # The output reaches the conversation, but nothing talks over the user.
    await wait_until(lambda: len(conn.sent) == 1)
    if turn_end == "refused_after_output":
        refuse()
    coordinator.handle_event(SimpleNamespace(type="input_audio_buffer.committed", item_id="item_1"))
    await asyncio.sleep(0.01)
    assert [event["type"] for event in conn.sent] == ["conversation.item.create"]

    # The user item commits once the turn cannot reopen.
    coordinator.handle_event(SimpleNamespace(type="input_audio_buffer.committed", item_id="item_2"))
    if turn_end in {"answered", "missed_output", "empty_answer", "cancelled_answer"}:
        output = [SimpleNamespace(type="message")] if turn_end != "empty_answer" else []
        coordinator.handle_event(response_created("response_2"))
        coordinator.handle_event(
            response_done(
                "response_2",
                output=output,
                status="cancelled" if turn_end == "cancelled_answer" else "completed",
                metadata={TOOL_INPUT_METADATA_KEY: '["call_1"]'} if turn_end != "missed_output" else None,
            )
        )
    await asyncio.sleep(0.01)
    # Only an answer the user heard makes the follow-up unnecessary.
    assert [event["type"] for event in conn.sent][1:] == ([] if turn_end == "answered" else ["response.create"])
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
    assert len(coordinator._queued_tool_results) == 2
    create_event = conn.sent[-1]

    coordinator.handle_event(
        response_created(
            "response_tool_1",
            metadata={"s2s_local_tool_create_id": create_event["event_id"]},
        )
    )
    assert len(coordinator._queued_tool_results) == 0

    coordinator.handle_event(response_done("response_tool_1"))
    await asyncio.sleep(0.05)
    assert len(conn.sent) == 3
    await coordinator.close()


@pytest.mark.parametrize("settlement", ["accepted", "covered"])
async def test_audio_client_tool_follow_up_preserves_batch_beyond_metadata_limit(settlement):
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn, RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor)
    )
    call_ids = {f"call_{index:02d}_" + "x" * 24 for index in range(20)}
    try:
        coordinator.handle_event(response_done(output=[function_call(call_id) for call_id in sorted(call_ids)]))
        await wait_until(lambda: len(conn.sent) == len(call_ids) + 1)
        create = conn.sent[-1]
        marker = create["response"]["metadata"][TOOL_FOLLOWUP_METADATA_KEY]
        marked_ids = set(json.loads(marker))
        assert len(marker) <= 512
        assert marked_ids < call_ids
        assert {event["item"]["call_id"] for event in conn.sent[:-1]} == call_ids

        if settlement == "covered":
            # Only the IDs that fit in the marker have already been answered.
            coordinator.handle_event(
                SimpleNamespace(
                    type="error",
                    error=SimpleNamespace(type=TOOL_FOLLOWUP_COVERED, code=None, event_id=create["event_id"]),
                )
            )
            await wait_until(lambda: len(conn.sent) == len(call_ids) + 2)
            create = conn.sent[-1]
            assert set(json.loads(create["response"]["metadata"][TOOL_FOLLOWUP_METADATA_KEY])) == (
                call_ids - marked_ids
            )

        # A server without the acknowledgement extension echoes request metadata.
        # Acceptance covers every submitted result, even if the marker is partial.
        coordinator.handle_event(response_created("response_tool", metadata=create["response"]["metadata"]))
        assert not coordinator._queued_tool_results
        coordinator.handle_event(
            response_done(
                "response_tool",
                output=[SimpleNamespace(type="message")],
                metadata=create["response"]["metadata"],
            )
        )
        await asyncio.sleep(0.01)
        assert len(conn.sent) == len(call_ids) + (2 if settlement == "covered" else 1)
    finally:
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


@pytest.mark.parametrize("user_turn", [False, True])
async def test_audio_client_waits_for_response_lifecycle_after_follow_up_collision(user_turn):
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn,
        RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor),
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 2)
    create_id = conn.sent[-1]["event_id"]
    if user_turn:
        coordinator.handle_event(SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item_2"))

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
    assert len(coordinator._queued_tool_results) == 1

    if user_turn:
        coordinator.handle_event(SimpleNamespace(type="input_audio_buffer.committed", item_id="item_2"))
    coordinator.handle_event(response_created("response_implicit"))
    coordinator.handle_event(
        response_done(
            "response_implicit",
            output=[SimpleNamespace(type="message")],
            metadata={TOOL_INPUT_METADATA_KEY: '["call_1"]'} if user_turn else None,
        )
    )
    if user_turn:
        # The answer to the turn started after the output, so it used it.
        await asyncio.sleep(0.01)
        assert len(conn.sent) == 2
    else:
        await wait_until(lambda: len(conn.sent) == 3)
        assert conn.sent[-1]["type"] == "response.create"
    await coordinator.close()


@pytest.mark.parametrize("rejection", ["tool_followup_already_answered", "tool_followup_user_turn_open"])
async def test_audio_client_handles_tool_followup_settlement(rejection):
    conn = RecordingConnection()
    coordinator = _ToolCallCoordinator(
        conn, RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=done_tool_executor)
    )
    coordinator.handle_event(response_done(output=[function_call("call_1")]))
    await wait_until(lambda: len(conn.sent) == 2)
    create_id = conn.sent[-1]["event_id"]
    coordinator.handle_event(
        SimpleNamespace(type="error", error=SimpleNamespace(type=rejection, code=None, event_id=create_id))
    )
    await asyncio.sleep(0.01)
    assert len(conn.sent) == 2
    if rejection == "tool_followup_user_turn_open":
        coordinator.handle_event(SimpleNamespace(type="input_audio_buffer.committed", item_id="item_2"))
        await wait_until(lambda: len(conn.sent) == 3)
    else:
        assert len(coordinator._queued_tool_results) == 0
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


def test_audio_stream_clock_mapping_uses_the_shortest_read(monkeypatch):
    readings = iter([0, 10, 20, 22, 30, 34, 40, 46, 50, 58])
    stream_times = iter([101, 121, 131, 141, 151])

    class Stream:
        @property
        def time(self):
            return next(stream_times)

    monkeypatch.setattr(audio_client_module.time, "monotonic", lambda: next(readings))

    assert audio_client_module._stream_clock_offset(Stream()) == -100


@pytest.mark.parametrize("echo_cancellation", [False, True])
@pytest.mark.parametrize("windows", [False, True])
def test_audio_streams_are_cleaned_up_when_startup_fails(monkeypatch, echo_cancellation, windows):
    events = []
    stream_options = []

    class FakeStream:
        def __init__(self, name, *, fail_start=False):
            self.name = name
            self.fail_start = fail_start
            self.time = 0.0

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

    def make_stream(stream, **kwargs):
        stream_options.append(kwargs)
        return stream

    wasapi_settings = object()
    fake_sounddevice = SimpleNamespace(
        RawInputStream=lambda **kwargs: make_stream(input_stream, **kwargs),
        RawOutputStream=lambda **kwargs: make_stream(output_stream, **kwargs),
        query_hostapis=lambda: [
            {"name": "MME"},
            {"name": "Windows WASAPI", "default_input_device": 9, "default_output_device": 8},
        ],
        query_devices=lambda device, kind: {"hostapi": 1},
        WasapiSettings=lambda *, auto_convert: wasapi_settings if auto_convert else None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    monkeypatch.setattr(
        audio_client_module, "sys", SimpleNamespace(platform="win32" if windows else "linux"), raising=False
    )

    config = RealtimeAudioClientConfig(playback_buffer_ms=240, echo_cancellation=echo_cancellation)
    with pytest.raises(RuntimeError, match="output start failed"):
        with audio_client_module._open_audio_streams(config, PlaybackBuffer(16000, 240), audio_client_module.Queue()):
            pass

    assert events == ["input.start", "output.start", "input.stop", "output.close", "input.close"]
    if windows and echo_cancellation:
        assert [options["device"] for options in stream_options] == [9, 8]
        assert all(options["extra_settings"] is wasapi_settings for options in stream_options)
    else:
        assert all(options["device"] is None and "extra_settings" not in options for options in stream_options)


@pytest.mark.parametrize("devices", [(9, 8), (1, 8), (9, 3)])
def test_windows_echo_cancellation_preserves_explicit_wasapi_devices(monkeypatch, devices):
    monkeypatch.setattr(audio_client_module, "sys", SimpleNamespace(platform="win32"), raising=False)
    opened = []

    def stream(**kwargs):
        opened.append(kwargs)
        return SimpleNamespace(time=0.0, start=lambda: None, stop=lambda: None, close=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            query_hostapis=lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}],
            query_devices=lambda device, kind: {"hostapi": 1 if device in (9, 8) else 0},
            WasapiSettings=lambda **kwargs: kwargs,
            RawInputStream=stream,
            RawOutputStream=stream,
        ),
    )
    config = RealtimeAudioClientConfig(echo_cancellation=True, input_device=devices[0], output_device=devices[1])
    if devices == (9, 8):
        with audio_client_module._open_audio_streams(config, PlaybackBuffer(16000), audio_client_module.Queue()):
            pass
        assert [options["device"] for options in opened] == [9, 8]
    else:
        with pytest.raises(ValueError, match="requires WASAPI input and output"):
            with audio_client_module._open_audio_streams(config, PlaybackBuffer(16000), audio_client_module.Queue()):
                pass
        assert not opened


async def test_audio_worker_reports_startup_failure_and_exits():
    config = RealtimeAudioClientConfig(echo_cancellation=True, input_device=-9999)
    worker = audio_client_module._AudioWorker(config)
    with worker:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(worker.wait_for_failure(), timeout=15)
    assert not worker._process.is_alive()


def test_audio_worker_can_stop_before_opening_devices():
    worker = audio_client_module._AudioWorker(RealtimeAudioClientConfig(echo_cancellation=True, input_device=-9999))
    worker._stop.set()
    with worker:
        worker.playback.append(bytes(1024 * 1024))
        worker.playback.clear()
    assert worker._process.exitcode == 0
    worker._commands._thread.join(timeout=2)
    assert not worker._commands._thread.is_alive()


def _produce_microphone_audio(mic_queue):
    for _ in range(128):
        mic_queue.put_nowait(bytes(16384))
    Event().wait(15)


def test_audio_worker_death_unblocks_microphone_reader():
    worker = audio_client_module._AudioWorker(RealtimeAudioClientConfig(echo_cancellation=True))
    worker._process = audio_client_module.multiprocessing.get_context("spawn").Process(
        target=_produce_microphone_audio, args=(worker.mic_queue,)
    )

    def drain():
        try:
            while True:
                worker.mic_queue.get(timeout=0.1)
        except (EOFError, OSError):
            pass

    with worker:
        assert worker.mic_queue.get(timeout=10) == bytes(16384)
        worker._process.terminate()
        worker._process.join(timeout=5)
        reader = Thread(target=drain, daemon=True)
        reader.start()
        reader.join(timeout=2)
        assert not reader.is_alive()


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
            RealtimeAudioClientConfig(echo_cancellation=False),
            Event(),
        )

    assert events == ["input.close"]


@pytest.mark.parametrize("startup_failure", [False, True])
async def test_audio_client_startup_and_shutdown_use_public_realtime_connection(monkeypatch, startup_failure):
    sent = []
    audio_session_calls = []

    class FakeConnection:
        async def send(self, event):
            sent.append(event)

        async def recv(self):
            await asyncio.Future()

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
    if not startup_failure:
        monkeypatch.setattr(audio_client_module, "_run_audio_session", fake_audio_session)
    config = RealtimeAudioClientConfig(echo_cancellation=startup_failure, input_device=-9999)

    if startup_failure:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(audio_client_module.listen_and_play_realtime(config, stop_event=stop_event), 15)
    else:
        await audio_client_module.listen_and_play_realtime(config, stop_event=stop_event)
        assert len(audio_session_calls) == 1
        assert audio_session_calls[0][2] is stop_event

    assert sent == [build_session_update(config)]
    assert fake_client.closed is True


async def test_audio_session_cancellation_closes_streams_and_stops_receiving(monkeypatch):
    receiving = asyncio.Event()
    closed = []

    @contextmanager
    def streams(*_args):
        try:
            yield
        finally:
            closed.append("streams")

    class Connection:
        async def recv(self):
            receiving.set()
            try:
                await asyncio.Future()
            finally:
                closed.append("receiver")

    monkeypatch.setattr(audio_client_module, "_open_audio_streams", streams)
    stop_event = Event()
    session = asyncio.create_task(
        audio_client_module._run_audio_session(
            Connection(), RealtimeAudioClientConfig(echo_cancellation=False), stop_event
        )
    )
    await asyncio.wait_for(receiving.wait(), timeout=1)
    session.cancel()
    with pytest.raises(asyncio.CancelledError):
        await session
    assert stop_event.is_set()
    assert closed == ["receiver", "streams"]


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


def test_clean_conversation_has_colors_literal_text_and_no_blank_lines_between_turns():
    from io import StringIO

    from rich.console import Console
    from rich.text import Text

    output = StringIO()
    terminal = Console(file=output, force_terminal=True, color_system="standard", no_color=False)
    renderer = _FriendlyEventRenderer(conversation_only=True, console=terminal)
    playback = PlaybackBuffer(16000)
    events = [
        SimpleNamespace(type="session.created"),
        SimpleNamespace(type="session.updated"),
        SimpleNamespace(type="session.updated"),
    ]
    for user, assistant in (("Hello [bold]", "Hi there!"), ("Thanks", "You're welcome.")):
        events.extend(
            [
                SimpleNamespace(type="input_audio_buffer.speech_started"),
                SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript=user),
                response_created("response_1"),
                SimpleNamespace(type="response.output_audio_transcript.delta", delta=assistant),
                SimpleNamespace(type="response.output_audio_transcript.done", transcript=assistant),
                SimpleNamespace(type="response.output_audio.done"),
                response_done(),
            ]
        )
    for event in events:
        handle_server_event(event, playback=playback, renderer=renderer, print_json=False)

    colored = output.getvalue()
    assert "\x1b[33mUSER: Hello [bold]" in colored
    assert "\x1b[32mASSISTANT: " in colored
    assert Text.from_ansi(colored).plain == (
        "Ready — start speaking.\nUSER: Hello [bold]\nASSISTANT: Hi there!\nUSER: Thanks\nASSISTANT: You're welcome.\n"
    )


@pytest.mark.parametrize("echo_cancellation", [False, True])
async def test_ready_waits_for_audio_devices_and_session_acknowledgement(monkeypatch, capsys, echo_cancellation):
    starts = []
    stop = Event()
    audio_sent = asyncio.Event()

    class Stream:
        def __init__(self, name):
            self.name = name

        def start(self):
            starts.append(self.name)

        def stop(self):
            pass

        def close(self):
            pass

    class Connection:
        count = 0

        async def send(self, _event):
            audio_sent.set()

        async def recv(self):
            assert starts == ["microphone", "speakers"]
            self.count += 1
            if self.count == 1:
                assert capsys.readouterr().out == ""
                return SimpleNamespace(type="session.created")
            if self.count == 2:
                assert capsys.readouterr().out == ""
                return SimpleNamespace(type="session.updated")
            assert capsys.readouterr().out == "Ready — start speaking.\n"
            stop.set()
            await asyncio.Future()

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            RawInputStream=lambda **_: Stream("microphone"),
            RawOutputStream=lambda **_: Stream("speakers"),
        ),
    )

    class Worker:
        def __init__(self):
            self.mic_queue = audio_client_module.Queue()
            self.playback = PlaybackBuffer(16000)
            self._ready = Event()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        async def wait_for_failure(self):
            await asyncio.Future()

    worker = Worker()
    monkeypatch.setattr(audio_client_module, "_AudioWorker", lambda _config: worker)
    conn = Connection()
    session = asyncio.create_task(
        audio_client_module._run_audio_session(
            conn, RealtimeAudioClientConfig(echo_cancellation=echo_cancellation), stop
        )
    )
    if echo_cancellation:
        # Input can arrive before the output stream has finished opening.
        worker.mic_queue.put(bytes(320))
        await asyncio.wait_for(audio_sent.wait(), 2)
        assert conn.count == 0
        assert capsys.readouterr().out == ""
        starts.extend(["microphone", "speakers"])
        worker._ready.set()
    await asyncio.wait_for(session, 2)


def test_voice_status_is_rewritten_in_place_below_the_conversation():
    from io import StringIO

    from rich.console import Console

    output = StringIO()
    terminal = Console(file=output, force_terminal=True, no_color=True, width=40)
    renderer = _FriendlyEventRenderer(conversation_only=True, console=terminal)
    renderer.show_status("Voice [1/38]: default")
    renderer.show_status("Voice [2/38]: Zoey — switching")
    renderer.render_assistant_text_delta(
        SimpleNamespace(delta="Hi.", response_id="r", item_id="i", output_index=0, content_index=0)
    )
    renderer.show_status("Voice [3/38]: Juniper — next reply")
    renderer.finish_live_assistant_text()
    renderer.clear_status()

    clear = "\r\x1b[2K"
    assert output.getvalue() == (
        "Voice [1/38]: default"
        f"{clear}Voice [2/38]: Zoey — switching"
        f"{clear}ASSISTANT: Hi.\n"
        f"Voice [3/38]: Juniper — next reply{clear}"
    )


def test_voice_status_prints_lines_without_a_terminal(capsys):
    renderer = _FriendlyEventRenderer(conversation_only=False)
    renderer.show_status("Voice [2/38]: Zoey — selected")
    assert capsys.readouterr().out == "Voice [2/38]: Zoey — selected\n"
