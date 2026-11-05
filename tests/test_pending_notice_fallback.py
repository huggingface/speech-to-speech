"""A provider or chat template that rejects the pending tool calls system note gets it folded into the user turn."""

from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError
from openai.types.realtime import RealtimeConversationItemFunctionCall

from speech_to_speech.LLM.chat import PENDING_NOTICE_PREFIX, Chat, fold_pending_notice, make_user_message
from speech_to_speech.LLM.language_model import _messages_with_pending_notice
from tests.test_chat_completions_backend import _make_handler as make_chat_completions_handler
from tests.test_responses_api_language_model import _make_handler as make_responses_handler

NOTICE = f"{PENDING_NOTICE_PREFIX} were already made and are still running."


def _chat_with_pending_call() -> Chat:
    chat = Chat(20)
    chat.add_item(make_user_message("what is 12 times 12?"))
    chat.add_ordered_function_call(
        RealtimeConversationItemFunctionCall(type="function_call", call_id="call_1", name="slow", arguments="{}")
    )
    return chat


def _bad_request() -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "http://fake/v1"))
    return BadRequestError("system message must come first", response=response, body=None)


def _has_trailing_notice(messages) -> bool:
    last = messages[-1]
    return last.get("role") == "system" and str(last.get("content")).startswith(PENDING_NOTICE_PREFIX)


def test_fold_moves_the_note_into_the_last_user_message():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
        {"role": "system", "content": NOTICE},
    ]
    folded = fold_pending_notice(messages)
    assert [m["role"] for m in folded] == ["user", "assistant", "user"]
    assert folded[-1]["content"] == f"second\n\n{NOTICE}"
    assert messages[-1]["role"] == "system"


@pytest.mark.parametrize(
    ("part", "expected"),
    [({"type": "input_text", "text": "hi"}, "input_text"), ({"type": "text", "text": "hi"}, "text")],
)
def test_fold_appends_a_part_of_the_matching_type(part, expected):
    folded = fold_pending_notice([{"role": "user", "content": [part]}, {"role": "system", "content": NOTICE}])
    assert folded[-1]["content"][-1] == {"type": expected, "text": NOTICE}


def test_fold_without_a_user_message_adds_one_and_ignores_other_system_messages():
    assert fold_pending_notice([{"role": "system", "content": NOTICE}]) == [{"role": "user", "content": NOTICE}]
    ordinary = [{"role": "user", "content": "hi"}, {"role": "system", "content": "something else"}]
    assert fold_pending_notice(ordinary) == ordinary


def test_chat_completions_retries_once_with_the_note_folded_and_stays_folded():
    handler = make_chat_completions_handler()
    sent = []

    def create(**kwargs):
        sent.append(kwargs["messages"])
        if _has_trailing_notice(kwargs["messages"]):
            raise _bad_request()
        return "ok"

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    chat = _chat_with_pending_call()

    assert handler._request(handler._serialize(chat), {}) == "ok"
    assert len(sent) == 2 and _has_trailing_notice(sent[0]) and not _has_trailing_notice(sent[1])
    assert PENDING_NOTICE_PREFIX in str(sent[1][-1]["content"])

    sent.clear()
    assert handler._request(handler._serialize(chat), {}) == "ok"
    assert len(sent) == 1 and not _has_trailing_notice(sent[0])


def test_responses_api_retries_once_with_the_note_folded():
    handler = make_responses_handler()
    sent = []

    def create(**kwargs):
        sent.append(kwargs["input"])
        if _has_trailing_notice(kwargs["input"]):
            raise _bad_request()
        return "ok"

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    chat = _chat_with_pending_call()

    assert handler._request(handler._serialize(chat), {}) == "ok"
    assert len(sent) == 2 and not _has_trailing_notice(sent[1])
    assert PENDING_NOTICE_PREFIX in str(sent[1][-1]["content"])


def test_an_unrelated_bad_request_is_not_retried():
    handler = make_chat_completions_handler()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise _bad_request()

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    chat = Chat(20)
    chat.add_item(make_user_message("hello"))

    with pytest.raises(BadRequestError):
        handler._request(handler._serialize(chat), {})
    assert len(calls) == 1


def test_a_400_that_the_retry_does_not_fix_leaves_the_note_as_a_system_message():
    handler = make_chat_completions_handler()

    def create(**kwargs):
        raise _bad_request()

    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    chat = _chat_with_pending_call()

    with pytest.raises(BadRequestError):
        handler._request(handler._serialize(chat), {})

    assert _has_trailing_notice(handler._serialize(chat))


class _StrictTemplate:
    """Like the Mistral and Gemma templates: a system message is only allowed first."""

    def apply_chat_template(self, messages, **kwargs):
        if any(m["role"] == "system" for m in messages[1:]):
            raise ValueError("System message must be at the beginning.")
        return "rendered"


def test_local_model_folds_the_note_when_the_template_rejects_it():
    chat = _chat_with_pending_call()

    folded = _messages_with_pending_notice(chat, _StrictTemplate())
    assert [m["role"] for m in folded] == ["user"]
    assert PENDING_NOTICE_PREFIX in folded[-1]["content"]

    accepting = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: "rendered")
    assert _has_trailing_notice(_messages_with_pending_notice(chat, accepting))
    assert _has_trailing_notice(_messages_with_pending_notice(chat))
