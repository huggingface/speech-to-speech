"""A cancelled streaming turn must not wait for the provider to speak again.

The provider socket can stay busy without producing an event: SSE comment
keep-alives keep the read alive while the SDK ignores them, so neither the
cancellation check nor the read timeout in the turn gets a chance to run.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
import speech_to_speech.LLM.chat_completions_language_model as ccm
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk
from tests.llm_history import drive_llm

# A cancelled turn that still waits on the provider is what this file is about;
# give the regression plenty of room, but never let a broken build hang CI.
CANCEL_DEADLINE_S = 5.0


class _KeepAliveThenSilentStream:
    """One chunk, then keep-alive traffic that never becomes an event.

    ``close`` releases the pending read, as the real SDK does; nothing else
    does, so a turn that never closes the response cannot finish.
    """

    def __init__(self):
        self.released = threading.Event()
        self.closed = threading.Event()
        self.reading = threading.Event()

    def __iter__(self):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hallo.", tool_calls=None), finish_reason=None)],
            usage=None,
        )
        # Reached only when the consumer asks for the next event, i.e. while the
        # real read sits waiting for another SSE line.
        self.reading.set()
        self.released.wait(60)

    def close(self):
        self.closed.set()
        self.released.set()


class _FakeCompletions:
    def __init__(self, stream):
        self._stream = stream

    def create(self, **kwargs):
        return self._stream


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=None)

    def with_options(self, **kwargs):
        return self


def _make_handler(stream):
    original = base_mod.OpenAI

    def build(*args, **kwargs):
        client = _FakeClient()
        client.chat.completions = _FakeCompletions(stream)
        return client

    base_mod.OpenAI = build
    try:
        return ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=dict(
                model_name="test-model",
                base_url="http://fake/v1",
                api_key="k",
                stream=True,
                disable_thinking=True,
                compact_history=False,
            ),
        )
    finally:
        base_mod.OpenAI = original


def _request():
    chat = Chat(10)
    chat.add_item(make_user_message("Hallo"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Du bist ein Roboter.")
    return GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        response=None,
        language_code="de",
        turn_id="turn_1",
        turn_revision=0,
    )


def _drive_in_background(handler, request):
    """Run one turn on its own thread, as the pipeline does, and record it."""
    result: dict[str, object] = {}
    finished = threading.Event()

    def run():
        outputs = []
        try:
            for output in drive_llm(handler, request):
                outputs.append(output)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertions
            result["error"] = exc
        else:
            result["outputs"] = outputs
        finally:
            finished.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, finished, result


def test_cancel_releases_a_stream_reading_keep_alives():
    stream = _KeepAliveThenSilentStream()
    ccm.Stream = _KeepAliveThenSilentStream

    scope = CancelScope()
    handler = _make_handler(stream)
    handler.cancel_scope = scope
    request = _request()

    thread, finished, result = _drive_in_background(handler, request)
    assert stream.reading.wait(10), "the turn never started waiting for the next provider event"

    scope.cancel()
    assert finished.wait(CANCEL_DEADLINE_S), "a cancelled turn stayed blocked on the provider read"
    thread.join(1)
    assert "error" not in result, result.get("error")

    outputs = result["outputs"]
    assert isinstance(outputs[-1], EndOfResponse)
    assert outputs[-1].error is None
    assert stream.closed.is_set(), "the response must be released when the turn is cancelled"
    assert [getattr(item, "role", None) for item in request.runtime_config.chat.buffer] == ["user"]


def test_completed_stream_still_delivers_every_event():
    class _Chunks:
        def __init__(self):
            self.closed = threading.Event()

        def __iter__(self):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hallo.", tool_calls=None), finish_reason=None)],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=" Wie geht es dir?", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
            )

        def close(self):
            self.closed.set()

    stream = _Chunks()
    ccm.Stream = _Chunks
    handler = _make_handler(stream)
    text = ""
    end = None
    for output in drive_llm(handler, _request()):
        if isinstance(output, LLMResponseChunk):
            text += output.text
        elif isinstance(output, EndOfResponse):
            end = output
    assert "Hallo" in text and "Wie geht es dir" in text
    assert end is not None and end.error is None
    assert stream.closed.is_set(), "the response is released once the turn completes"
