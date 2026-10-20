import asyncio
import os
from threading import Event
from types import SimpleNamespace

import pytest

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig
from speech_to_speech.api.openai_realtime.voice_controls import (
    ArrowDecoder,
    LanguageSelection,
    VoiceSelection,
    arrow_steps,
)


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


def language_updated(language):
    return SimpleNamespace(
        type="session.updated",
        session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(language=language))),
    )


async def test_language_update_waits_for_reply_and_preserves_instructions_and_voice():
    connection = Connection()
    config = RealtimeAudioClientConfig(
        voice="Zoey", language="en", language_choices=["en", "fr", "de"], instructions="Be brief."
    )
    shown = []
    control = LanguageSelection(connection, config, shown.append)
    await control.handle_event(SimpleNamespace(type="response.created", response=SimpleNamespace(id="current")))
    await control.select(1)
    assert connection.sent == [] and "French" in shown[-1]
    await control.handle_event(SimpleNamespace(type="response.done", response=SimpleNamespace(id="current")))
    assert connection.sent == [
        {
            "type": "session.update",
            "event_id": "local_language_1",
            "session": {"type": "realtime", "audio": {"output": {"language": "fr"}}},
        }
    ]
    assert config.language == "en"
    await control.handle_event(language_updated("fr"))
    assert config.language == "fr" and config.voice == "Zoey" and config.instructions == "Be brief."
    await control.select(1)
    await control.handle_event(SimpleNamespace(type="error", error=SimpleNamespace(event_id="local_language_2")))
    assert config.language == "fr" and control.index == 1 and "rejected" in shown[-1]


def test_decoder_handles_split_keys_application_mode_and_ignores_text():
    decoder = ArrowDecoder()
    assert decoder.feed(b"text\x1b") == []
    assert decoder.feed(b"[") == []
    assert decoder.feed(b"C\x1b[D\x1bOA\x1bOB") == ["right", "left", "up", "down"]
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
    # Linux hangs up the slave when its master closes, so inspect restoration first.
    with os.fdopen(master, "wb", buffering=0), os.fdopen(slave) as stream:
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
                assert await asyncio.wait_for(task, 1) == "right"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await keys.aclose()
        # macOS sets PENDIN when canonical mode resumes; clear pending input
        # before comparing the restored configuration, with both PTY ends open.
        termios.tcflush(stream.fileno(), termios.TCIFLUSH)
        assert termios.tcgetattr(stream.fileno()) == previous


async def test_redirected_input_has_no_keyboard_controls(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("\x1b[C")
    with path.open() as stream:
        assert [step async for step in arrow_steps(stream)] == []


@pytest.mark.parametrize("key", ["right", "down"])
async def test_live_audio_session_applies_arrow_on_existing_connection_after_reply(monkeypatch, key):
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
            audio = event["session"]["audio"]
            events.put_nowait(
                updated(audio["output"]["voice"]) if "voice" in audio["output"] else language_updated("fr")
            )

    async def keys():
        try:
            await ready.wait()
            yield key
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
    config = RealtimeAudioClientConfig(
        voice="Zoey", voice_choices=["Zoey", "Maeve"], language="en", language_choices=["en", "fr"]
    )
    conn = AudioConnection()
    await asyncio.wait_for(audio_client._run_audio_session(conn, config, stop), 2)
    assert config.voice == ("Maeve" if key == "right" else "Zoey")
    assert config.language == ("fr" if key == "down" else "en")
    assert len(conn.sent) == 1 and conn.sent[0]["type"] == "session.update"
    assert "keyboard" in closed and closed.count("stream closed") == 2


def test_terminal_selector_updates_one_line_and_waits_during_streamed_text():
    from io import StringIO

    from rich.console import Console

    from speech_to_speech.api.openai_realtime.audio_client import _FriendlyEventRenderer
    from speech_to_speech.api.openai_realtime.voice_controls import SelectionDisplay

    output = StringIO()
    console = Console(file=output, force_terminal=True, width=100, color_system=None)
    display = SelectionDisplay(console)
    renderer = _FriendlyEventRenderer(conversation_only=True, console=console)
    renderer.selection_display = display
    display.update("Voice [1/2]: Trevor — switching")
    display.update("Voice [1/2]: Trevor — selected")
    assert "\n" not in output.getvalue()
    assert "\r\x1b[2K" in output.getvalue()
    renderer.render_assistant_text_delta(SimpleNamespace(delta="Hello"))
    during_reply = output.getvalue()
    display.update("Voice [2/2]: Vyrmathrax the Ancient — selected")
    assert output.getvalue() == during_reply
    renderer.render_assistant_text_delta(SimpleNamespace(delta=" there!"))
    assert output.getvalue().endswith("ASSISTANT: Hello there!")
    renderer.finish_live_assistant_text()
    assert output.getvalue().endswith("Voice [2/2]: Vyrmathrax the Ancient — selected")
    display.clear()
    assert not display.visible


def test_redirected_selector_omits_transient_switching_and_uses_readable_names():
    from io import StringIO

    from rich.console import Console

    from speech_to_speech.api.openai_realtime.voice_controls import SelectionDisplay

    output = StringIO()
    display = SelectionDisplay(Console(file=output, force_terminal=False))
    config = RealtimeAudioClientConfig(voice_choices=["Vyrmathrax_the_Ancient"])
    control = VoiceSelection(Connection(), config, display.update)
    control.announce("switching")
    control.announce("selected")
    assert output.getvalue() == "Voice [1/1]: Vyrmathrax the Ancient — selected\n"
