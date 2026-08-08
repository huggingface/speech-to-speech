from io import StringIO

from rich.console import Console

from speech_to_speech import conversation_console
from speech_to_speech.STT import parakeet_tdt_handler
from speech_to_speech.TTS import kokoro_handler


def test_conversation_console_suppresses_private_content(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(conversation_console, "console", Console(file=output))

    conversation_console.configure_conversation_text_output(enabled=False)
    conversation_console.console.print("private-user-sentinel")
    conversation_console.console.print("private-assistant-sentinel")

    assert output.getvalue() == ""


def test_conversation_console_preserves_default_output(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(conversation_console, "console", Console(file=output))

    conversation_console.configure_conversation_text_output(enabled=True)
    conversation_console.console.print("visible-sentinel")

    assert "visible-sentinel" in output.getvalue()


def test_private_output_setting_reaches_local_runtime_handlers():
    try:
        conversation_console.configure_conversation_text_output(enabled=False)

        assert parakeet_tdt_handler.console is conversation_console.console
        assert kokoro_handler.console is conversation_console.console
        assert parakeet_tdt_handler.console.quiet is True
        assert kokoro_handler.console.quiet is True
    finally:
        conversation_console.configure_conversation_text_output(enabled=True)
