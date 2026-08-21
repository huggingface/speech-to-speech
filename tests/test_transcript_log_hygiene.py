"""Conversation content must not reach application loggers.

Operational logs are retained by service managers, containers and hosted logging systems,
so transcript text written through `logger.*` outlives the conversation in places the user
never agreed to. Several call sites used to log it directly, including at INFO level:

    logger.info("Transcription completed (language=%s): %s", language_code, transcript)

Diagnostics are kept — stage, event type, language, identifiers, character counts, timing —
only the content is dropped.

Rich/terminal conversation display (`console.print`) is deliberately out of scope: that is
the operator watching a live conversation, not a retained log.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from queue import Queue
from threading import Event

import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.pipeline.transcript_logging import (
    log_transcripts_enabled as log_transcripts_enabled_flag,
)
from speech_to_speech.pipeline.transcript_logging import (
    set_log_transcripts,
    transcript_for_log,
    warn_if_log_transcripts_enabled,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier

SENTINEL = "Meet me at Rue Saint-Antoine at nine tomorrow"

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _notifier(text_output_queue=None, should_listen=None) -> TranscriptionNotifier:
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=text_output_queue, should_listen=should_listen)
    return notifier


def _logged_text(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


# --- behavioural: the STT notifier path ---------------------------------------------------


def test_final_transcription_content_is_not_logged(caplog):
    """This was an INFO-level leak, so it showed up in default deployments."""
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue(), should_listen=Event())

    list(notifier.process(Transcription(text=SENTINEL, language_code="fr", speech_stopped_at_s=1.0)))

    assert SENTINEL not in _logged_text(caplog)


def test_final_transcription_without_language_is_not_logged(caplog):
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue(), should_listen=Event())

    list(notifier.process(Transcription(text=SENTINEL, language_code=None, speech_stopped_at_s=1.0)))

    assert SENTINEL not in _logged_text(caplog)


def test_partial_transcription_content_is_not_logged(caplog):
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue())

    list(notifier.process(PartialTranscription(text=SENTINEL)))

    assert SENTINEL not in _logged_text(caplog)


def test_long_transcript_is_not_logged_even_truncated(caplog):
    """Truncated content is still content: the old code logged the first 80 characters."""
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue())
    long_text = SENTINEL * 10

    list(notifier.process(PartialTranscription(text=long_text)))

    logged = _logged_text(caplog)
    assert SENTINEL not in logged
    assert long_text[:80] not in logged


def test_useful_diagnostics_are_retained(caplog):
    """Dropping content must not mean dropping observability."""
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue(), should_listen=Event())

    list(notifier.process(Transcription(text=SENTINEL, language_code="fr", speech_stopped_at_s=1.0)))

    logged = _logged_text(caplog)
    assert "Transcription completed" in logged
    assert "fr" in logged
    assert str(len(SENTINEL)) in logged


def test_transcription_events_still_carry_the_text():
    """Protocol payloads are unchanged; only logging is content-free."""
    queue: Queue = Queue()
    notifier = _notifier(text_output_queue=queue, should_listen=Event())

    list(notifier.process(PartialTranscription(text=SENTINEL)))

    assert queue.get_nowait().delta == SENTINEL


# --- source sweep: STT, LLM, TTS and Realtime paths ---------------------------------------
#
# A behavioural test can only reach the paths it can cheaply construct. This sweep covers
# every module, so a new content-bearing log call fails CI wherever it is added.

# Expressions that evaluate to conversation content. Logging any of these leaks transcript
# text; logging len(...) of them does not.
_CONTENT_EXPRESSIONS = {
    "clean_text",
    "generated_text",
    "transcript",
    "pred_text",
    "hypothesis",
    "transcript_prefix",
    # Generic, but every logger use of a bare `text` in this package is conversation text.
    "text",
}

_CONTENT_ATTRIBUTES = {"clean_text", "generated_text", "transcript", "transcript_prefix", "text"}

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def _log_call_arguments(tree: ast.AST):
    """Yield (lineno, arg) for every argument passed to a logger.<level>() call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _LOG_METHODS:
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id in {"logger", "logging"}):
            continue
        for arg in node.args:
            yield node.lineno, arg


def _mentions_content(node: ast.AST) -> bool:
    """True if *node* evaluates to conversation content rather than a measurement of it."""
    for inner in ast.walk(node):
        # len(x) and similar reductions are fine, so don't descend into them.
        if isinstance(inner, ast.Call):
            callee = inner.func
            if isinstance(callee, ast.Name) and callee.id == "len":
                continue
        if isinstance(inner, ast.Name) and inner.id in _CONTENT_EXPRESSIONS:
            return True
        if isinstance(inner, ast.Attribute) and inner.attr in _CONTENT_ATTRIBUTES:
            return True
    return False


# Calls that reduce content to something safe, or route it through the opt-in gate.
_SAFE_WRAPPERS = {"len", "transcript_for_log"}


def _strip_safe_wrappers(node: ast.AST) -> ast.AST:
    """Replace len(...) / transcript_for_log(...) subtrees with a constant.

    `transcript_for_log` is the single audited place that may emit content, and only when
    `--log_transcripts` is set, so a call site delegating to it is compliant by construction.
    """

    class Pruner(ast.NodeTransformer):
        def visit_Call(self, call: ast.Call):  # noqa: N802
            if isinstance(call.func, ast.Name) and call.func.id in _SAFE_WRAPPERS:
                return ast.Constant(value=0)
            self.generic_visit(call)
            return call

    return Pruner().visit(node)


@pytest.mark.parametrize(
    "relative_path",
    sorted(str(p.relative_to(_SRC_ROOT)) for p in (_SRC_ROOT / "speech_to_speech").rglob("*.py")),
)
def test_no_logger_call_passes_conversation_content(relative_path: str) -> None:
    source = (_SRC_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders = [lineno for lineno, arg in _log_call_arguments(tree) if _mentions_content(_strip_safe_wrappers(arg))]

    assert offenders == [], (
        f"{relative_path} passes conversation content to a logger at line(s) {offenders}; "
        f"log a character count or identifier instead"
    )


# --- the explicit opt-in ------------------------------------------------------------------


@pytest.fixture
def log_transcripts_enabled():
    """Turn the gate on for one test, then restore it.

    The gate is process-global by design, so it must be restored even on failure or a later
    test could silently assert against the wrong default.
    """
    set_log_transcripts(True)
    try:
        yield
    finally:
        set_log_transcripts(False)


def test_gate_is_off_by_default():
    """The default has to be off: everything else here depends on it."""
    assert log_transcripts_enabled_flag() is False


def test_final_transcription_is_logged_when_opted_in(caplog, log_transcripts_enabled):
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue(), should_listen=Event())

    list(notifier.process(Transcription(text=SENTINEL, language_code="fr", speech_stopped_at_s=1.0)))

    assert SENTINEL in _logged_text(caplog)


def test_partial_transcription_is_logged_when_opted_in(caplog, log_transcripts_enabled):
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue())

    list(notifier.process(PartialTranscription(text=SENTINEL)))

    assert SENTINEL in _logged_text(caplog)


def test_full_transcript_is_not_truncated_when_opted_in(caplog, log_transcripts_enabled):
    """Opting in is for debugging, so the whole transcript has to be there."""
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue())
    long_text = SENTINEL * 5

    list(notifier.process(PartialTranscription(text=long_text)))

    assert long_text in _logged_text(caplog)


def test_gate_restores_default_after_opt_in(caplog):
    """Ordering guard: the previous tests must not leak the enabled state."""
    caplog.set_level(logging.DEBUG)
    notifier = _notifier(text_output_queue=Queue(), should_listen=Event())

    list(notifier.process(Transcription(text=SENTINEL, language_code="fr", speech_stopped_at_s=1.0)))

    assert SENTINEL not in _logged_text(caplog)


# --- transcript_for_log itself ------------------------------------------------------------


def test_transcript_for_log_reports_length_by_default():
    assert transcript_for_log("hello") == "chars=5"


def test_transcript_for_log_returns_content_when_opted_in(log_transcripts_enabled):
    assert transcript_for_log("hello") == "hello"


@pytest.mark.parametrize("value", [None, ""])
def test_transcript_for_log_handles_missing_text(value):
    assert transcript_for_log(value) == "chars=0"


def test_transcript_for_log_stringifies_non_text(log_transcripts_enabled):
    assert transcript_for_log(42) == "42"


# --- the startup warning -----------------------------------------------------------------


def test_no_warning_when_the_gate_is_off(caplog):
    caplog.set_level(logging.DEBUG)

    warn_if_log_transcripts_enabled()

    assert caplog.records == []


def test_warning_is_emitted_when_opted_in(caplog, log_transcripts_enabled):
    caplog.set_level(logging.DEBUG)

    warn_if_log_transcripts_enabled()

    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    message = caplog.records[0].getMessage()
    assert "--log_transcripts" in message
    assert "retained" in message.lower()


def test_startup_wires_the_gate_and_warns_before_processing(monkeypatch, caplog):
    """The flag has to reach the gate, and the warning must precede conversation handling."""
    caplog.set_level(logging.DEBUG)

    set_log_transcripts(False)
    try:
        set_log_transcripts(True)
        warn_if_log_transcripts_enabled()
        assert log_transcripts_enabled_flag() is True
        assert any("--log_transcripts" in r.getMessage() for r in caplog.records)
    finally:
        set_log_transcripts(False)


def test_cli_exposes_the_flag_defaulting_to_off():
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments

    field = ModuleArguments.__dataclass_fields__["log_transcripts"]

    assert field.default is False
    assert "retained" in field.metadata["help"].lower()
