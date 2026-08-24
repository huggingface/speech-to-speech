"""Conversation content is retained in protocol events, but omitted from logs by default."""

from __future__ import annotations

import logging
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    LLMResponseChunk,
    PartialTranscription,
    Transcription,
)
from speech_to_speech.pipeline.transcript_logging import (
    log_transcripts_enabled,
    set_log_transcripts,
    transcript_for_log,
    warn_if_log_transcripts_enabled,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.TTS.facebookmms_handler import FacebookMMSTTSHandler

SENTINEL = "Meet me at Rue Saint-Antoine at nine tomorrow"
TOOL_SENTINEL = "account-number-8675309"
ERROR_SENTINEL = "tts-error-8675309"


@pytest.fixture(autouse=True)
def reset_transcript_gate():
    set_log_transcripts(False)
    yield
    set_log_transcripts(False)


def _notifier() -> TranscriptionNotifier:
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=Queue(), should_listen=Event())
    return notifier


def _logged_text(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def _assert_content_visibility(logged: str, enabled: bool, *values: str) -> None:
    for value in values:
        assert (value in logged) is enabled


@pytest.mark.parametrize(
    "message",
    [
        Transcription(text=SENTINEL, language_code="fr", speech_stopped_at_s=1.0),
        Transcription(text=SENTINEL, language_code=None, speech_stopped_at_s=1.0),
        PartialTranscription(text=SENTINEL),
        PartialTranscription(text=SENTINEL * 10),
    ],
)
def test_stt_content_is_not_logged_by_default(caplog, message):
    caplog.set_level(logging.DEBUG)

    list(_notifier().process(message))

    assert SENTINEL not in _logged_text(caplog)


def test_stt_metadata_is_retained(caplog):
    caplog.set_level(logging.DEBUG)

    list(_notifier().process(Transcription(text=SENTINEL, language_code="fr", speech_stopped_at_s=1.0)))

    logged = _logged_text(caplog)
    assert "Transcription completed" in logged
    assert "fr" in logged
    assert str(len(SENTINEL)) in logged


def test_transcription_protocol_event_still_carries_text():
    queue: Queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=queue, should_listen=Event())

    list(notifier.process(PartialTranscription(text=SENTINEL)))

    assert queue.get_nowait().delta == SENTINEL


@pytest.mark.parametrize(
    "message",
    [
        Transcription(text=SENTINEL, language_code="fr", speech_stopped_at_s=1.0),
        PartialTranscription(text=SENTINEL * 5),
    ],
)
def test_stt_content_is_logged_in_full_when_opted_in(caplog, message):
    caplog.set_level(logging.DEBUG)
    set_log_transcripts(True)

    list(_notifier().process(message))

    assert message.text in _logged_text(caplog)


@pytest.mark.parametrize("enabled", [False, True])
def test_llm_text_and_tool_arguments_follow_the_gate(caplog, enabled):
    caplog.set_level(logging.DEBUG)
    tool = ResponseFunctionToolCall(
        type="function_call",
        id="fc_test",
        call_id="call_test",
        name="remember",
        arguments=f'{{"note": "{TOOL_SENTINEL}"}}',
        status="completed",
    )
    chunk = LLMResponseChunk(parts=[AssistantTextPart(text=SENTINEL), AssistantToolCallPart(tool=tool)])
    processor = object.__new__(LMOutputProcessor)
    processor.setup()
    set_log_transcripts(enabled)

    list(processor.process(chunk))

    _assert_content_visibility(_logged_text(caplog), enabled, SENTINEL, TOOL_SENTINEL)


@pytest.mark.parametrize("enabled", [False, True])
def test_realtime_validation_errors_follow_the_gate(caplog, enabled):
    caplog.set_level(logging.DEBUG)
    raw = {
        "type": "conversation.item.create",
        "item": {"type": "function_call", "arguments": TOOL_SENTINEL},
    }
    set_log_transcripts(enabled)

    assert RealtimeService().parse_client_event(raw) is None

    _assert_content_visibility(_logged_text(caplog), enabled, TOOL_SENTINEL)


@pytest.mark.parametrize("enabled", [False, True])
def test_tts_exceptions_follow_the_gate(caplog, enabled):
    class FailingTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise ValueError(ERROR_SENTINEL)

    caplog.set_level(logging.DEBUG)
    handler = object.__new__(FacebookMMSTTSHandler)
    handler.language = "en"
    handler.tokenizer = FailingTokenizer()
    set_log_transcripts(enabled)

    assert handler.generate_audio("harmless input") is None

    logged = _logged_text(caplog)
    _assert_content_visibility(logged, enabled, ERROR_SENTINEL)
    assert "ValueError" in logged


def test_gate_is_off_by_default():
    assert log_transcripts_enabled() is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("hello", "chars=5"), (None, "chars=0"), ("", "chars=0")],
)
def test_transcript_for_log_reports_length_by_default(value, expected):
    assert transcript_for_log(value) == expected


def test_transcript_for_log_returns_stringified_content_when_opted_in():
    set_log_transcripts(True)

    assert transcript_for_log("hello") == "hello"
    assert transcript_for_log(42) == "42"


def test_no_warning_when_the_gate_is_off(caplog):
    caplog.set_level(logging.DEBUG)

    warn_if_log_transcripts_enabled()

    assert caplog.records == []


def test_warning_is_emitted_when_opted_in(caplog):
    caplog.set_level(logging.DEBUG)
    set_log_transcripts(True)

    warn_if_log_transcripts_enabled()

    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    message = caplog.records[0].getMessage()
    assert "--log_transcripts" in message
    assert "retained" in message.lower()


def test_startup_wires_the_gate_and_warns_before_processing(monkeypatch):
    from speech_to_speech import s2s_pipeline

    events: list[str] = []
    args = SimpleNamespace(
        module_kwargs=SimpleNamespace(
            log_level="debug",
            log_transcripts=True,
            num_pipelines=1,
            enable_live_transcription=False,
        )
    )
    manager = SimpleNamespace(
        start=lambda: events.append("start"),
        wait=lambda: events.append("wait"),
        stop=lambda: events.append("stop"),
    )

    monkeypatch.setattr(s2s_pipeline, "parse_arguments", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(s2s_pipeline, "setup_logger", lambda _level: events.append("logger"))
    monkeypatch.setattr(s2s_pipeline, "prepare_all_args", lambda _args: events.append("prepare"))
    monkeypatch.setattr(
        s2s_pipeline,
        "build_pipeline",
        lambda _args, _stop_event: events.append("build") or manager,
    )
    monkeypatch.setattr(s2s_pipeline.signal, "signal", lambda *_args: None)
    original_warning = s2s_pipeline.warn_if_log_transcripts_enabled

    def record_warning():
        assert log_transcripts_enabled() is True
        events.append("warning")
        original_warning()

    monkeypatch.setattr(s2s_pipeline, "warn_if_log_transcripts_enabled", record_warning)

    s2s_pipeline.run_pipeline_command("serve", [])

    assert events == ["logger", "warning", "prepare", "build", "start", "wait"]


def test_cli_exposes_the_flag_defaulting_to_off():
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments

    field = ModuleArguments.__dataclass_fields__["log_transcripts"]

    assert field.default is False
    assert "retained" in field.metadata["help"].lower()
