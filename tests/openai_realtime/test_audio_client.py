import signal
import sys
from threading import Event
from types import SimpleNamespace

import pytest

import speech_to_speech.api.openai_realtime.audio_client as audio_client_module
from speech_to_speech.api.openai_realtime.audio_client import (
    PlaybackBuffer,
    RealtimeAudioClientConfig,
    _FriendlyEventRenderer,
    build_session_update,
    handle_server_event,
    normalize_realtime_url,
    run_realtime_audio_client,
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
