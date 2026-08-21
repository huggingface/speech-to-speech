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


def _strip_len_calls(node: ast.AST) -> ast.AST:
    """Replace len(...) subtrees with a constant so they are ignored by the scan."""

    class Pruner(ast.NodeTransformer):
        def visit_Call(self, call: ast.Call):  # noqa: N802
            if isinstance(call.func, ast.Name) and call.func.id == "len":
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

    offenders = [lineno for lineno, arg in _log_call_arguments(tree) if _mentions_content(_strip_len_calls(arg))]

    assert offenders == [], (
        f"{relative_path} passes conversation content to a logger at line(s) {offenders}; "
        f"log a character count or identifier instead"
    )
