import asyncio
import os
from threading import Event
from types import SimpleNamespace

import pytest

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig
from speech_to_speech.api.openai_realtime.voice_controls import ArrowDecoder, VoiceSelection, arrow_steps


class Connection:
    def __init__(self):
        self.sent = []

    async def send(self, event):
        self.sent.append(event)


def selector():
    connection = Connection()
    config = RealtimeAudioClientConfig(voice="Zoey", voice_choices=["Zoey", "Maeve", "Hugh"])
    shown = []
    return VoiceSelection(connection, config, shown.append), connection, config, shown


def updated(voice):
    return SimpleNamespace(
        type="session.updated", session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice=voice)))
    )


def test_decoder_handles_split_keys_application_mode_and_ignores_text():
    decoder = ArrowDecoder()
    assert decoder.feed(b"text\x1b") == []
    assert decoder.feed(b"[") == []
    assert decoder.feed(b"C\x1b[D\x1bOA\x1bOB") == [1, -1, -1, 1]
    assert decoder.feed(b"lettersABCD\x1b[1;5C\x1b[3~") == []
    assert decoder.feed(b"\x1b[" + b"1" * 100) == []
    assert decoder.pending == b""


@pytest.mark.parametrize("choices", [[""], ["  "], [1]])
def test_invalid_voice_choices_are_rejected(choices):
    with pytest.raises(ValueError, match="voice_choices"):
        RealtimeAudioClientConfig(voice_choices=choices)


def test_voice_choices_preserve_initial_voice_and_remove_duplicates():
    config = RealtimeAudioClientConfig(voice="Zoey", voice_choices=["Maeve", "Maeve", "Hugh"])
    assert config.voice_choices == ["Zoey", "Maeve", "Hugh"]
    assert RealtimeAudioClientConfig(voice_choices=["Zoey", "Maeve"]).voice == "Zoey"


async def test_voice_update_uses_same_session_without_touching_conversation_settings():
    control, connection, config, shown = selector()
    await control.select(1)
    assert connection.sent == [
        {
            "type": "session.update",
            "event_id": "local_voice_1",
            "session": {"type": "realtime", "audio": {"output": {"voice": "Maeve"}}},
        }
    ]
    assert config.voice == "Zoey"
    await control.handle_event(updated("Maeve"))
    assert config.voice == "Maeve" and "selected" in shown[-1]


async def test_arrows_wrap_and_coalesce_changes_until_current_response_finishes():
    control, connection, _, shown = selector()
    await control.handle_event(SimpleNamespace(type="response.created", response=SimpleNamespace(id="current")))
    await control.select(-1)
    await control.select(-1)
    assert connection.sent == []
    assert "Maeve" in shown[-1] and "next reply" in shown[-1]
    await control.handle_event(SimpleNamespace(type="response.done", response=SimpleNamespace(id="older")))
    assert connection.sent == []
    await control.handle_event(SimpleNamespace(type="response.done", response=SimpleNamespace(id="current")))
    assert len(connection.sent) == 1
    assert connection.sent[0]["session"]["audio"]["output"]["voice"] == "Maeve"


async def test_rejected_update_keeps_previous_voice_and_ignores_outdated_ack():
    control, connection, config, shown = selector()
    await control.select(1)
    await control.select(1)
    await control.handle_event(updated("Maeve"))
    assert config.voice == "Maeve" and control.index == 2
    await control.handle_event(
        SimpleNamespace(type="error", error=SimpleNamespace(event_id=connection.sent[-1]["event_id"]))
    )
    assert config.voice == "Maeve" and control.index == 1
    assert "rejected" in shown[-1]


@pytest.mark.skipif(os.name != "posix", reason="POSIX terminal mode")
@pytest.mark.parametrize("cancel", [False, True])
async def test_real_terminal_arrows_restore_echo_and_keep_ctrl_c(cancel):
    import termios

    master, slave = os.openpty()
    with os.fdopen(slave) as stream:
        previous = termios.tcgetattr(stream.fileno())
        keys = arrow_steps(stream)
        task = asyncio.create_task(anext(keys))
        try:
            await asyncio.sleep(0)
            active = termios.tcgetattr(stream.fileno())
            assert not active[3] & (termios.ICANON | termios.ECHO)
            assert active[3] & termios.ISIG == previous[3] & termios.ISIG
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                os.write(master, b"\x1b[")
                await asyncio.sleep(0.01)
                assert not task.done()
                os.write(master, b"C")
                assert await asyncio.wait_for(task, 1) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await keys.aclose()
            os.close(master)
        assert termios.tcgetattr(stream.fileno()) == previous


async def test_redirected_input_has_no_keyboard_controls(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("\x1b[C")
    with path.open() as stream:
        assert [step async for step in arrow_steps(stream)] == []


async def test_live_audio_session_applies_arrow_on_existing_connection_after_reply(monkeypatch):
    import sys

    from speech_to_speech.api.openai_realtime import audio_client

    ready = asyncio.Event()
    closed = []
    events = asyncio.Queue()
    stop = Event()
    events.put_nowait(SimpleNamespace(type="response.created", response=SimpleNamespace(id="reply")))

    class AudioConnection(Connection):
        async def recv(self):
            event = await events.get()
            if event.type == "response.created":
                ready.set()
            if event.type == "session.updated":
                stop.set()
            return event

        async def send(self, event):
            await super().send(event)
            events.put_nowait(updated(event["session"]["audio"]["output"]["voice"]))

    async def keys():
        try:
            await ready.wait()
            yield 1
            events.put_nowait(
                SimpleNamespace(type="response.done", response=SimpleNamespace(id="reply", status="completed"))
            )
            await asyncio.Future()
        finally:
            closed.append("keyboard")

    class Stream:
        def start(self):
            pass

        def stop(self):
            closed.append("stream stopped")

        def close(self):
            closed.append("stream closed")

    monkeypatch.setattr(audio_client, "arrow_steps", keys)
    monkeypatch.setattr(audio_client, "has_terminal_arrows", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            RawInputStream=lambda **kwargs: Stream(),
            RawOutputStream=lambda **kwargs: Stream(),
        ),
    )
    config = RealtimeAudioClientConfig(voice="Zoey", voice_choices=["Zoey", "Maeve"])
    conn = AudioConnection()
    await asyncio.wait_for(audio_client._run_audio_session(conn, config, stop), 2)
    assert config.voice == "Maeve"
    assert len(conn.sent) == 1 and conn.sent[0]["type"] == "session.update"
    assert "keyboard" in closed and closed.count("stream closed") == 2
