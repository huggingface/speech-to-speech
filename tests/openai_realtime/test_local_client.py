from threading import Event
from types import SimpleNamespace

import pytest

import speech_to_speech.api.openai_realtime.local_client as local_client_module
from speech_to_speech.api.openai_realtime.local_client import (
    PlaybackBuffer,
    RealtimeAudioClientConfig,
    _FriendlyEventRenderer,
    build_session_update,
    handle_server_event,
)


def test_local_client_sends_realtime_session_configuration():
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
def test_local_client_rejects_unsupported_explicit_pcm_rates(rate):
    with pytest.raises(ValueError, match="Unsupported rate"):
        build_session_update(RealtimeAudioClientConfig(send_rate=rate))


def test_local_client_clears_unplayed_audio_on_barge_in(capsys):
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


def test_local_client_clears_unplayed_audio_when_response_is_cancelled(capsys):
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


async def test_local_client_startup_and_shutdown_use_public_realtime_connection(monkeypatch):
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

    async def fake_audio_session(conn, config, received_stop_event, *, prompt_for_stop):
        audio_session_calls.append((conn, config, received_stop_event, prompt_for_stop))
        received_stop_event.set()

    monkeypatch.setattr(local_client_module, "_make_client", lambda _config: fake_client)
    monkeypatch.setattr(local_client_module, "_run_audio_session", fake_audio_session)
    config = RealtimeAudioClientConfig()

    await local_client_module.listen_and_play_realtime(
        config,
        stop_event=stop_event,
        prompt_for_stop=False,
    )

    assert sent == [build_session_update(config)]
    assert len(audio_session_calls) == 1
    assert audio_session_calls[0][2] is stop_event
    assert audio_session_calls[0][3] is False
    assert fake_client.closed is True
