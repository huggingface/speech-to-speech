"""End-to-end coverage for ``--llm_current_date``.

When enabled, the backend computes ``Today is <Weekday>, <Month> <D>, <YYYY>.``
per request (in ``--llm_timezone``) and the prompt builders put it on the first
line of the system prompt.

The OpenAI client and the clock are faked, so this runs with no network and no GPU.
"""

from __future__ import annotations

import queue
import re
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
import speech_to_speech.LLM.utils as llm_utils
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.arguments_classes.chat_completions_language_model_arguments import (
    ChatCompletionsLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.language_model_arguments import LanguageModelHandlerArguments
from speech_to_speech.backend_registry import LLM_BACKENDS
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.language_model import LanguageModelHandler
from speech_to_speech.LLM.voice_prompt import VOICE_SYSTEM_PROMPT_LEAD
from speech_to_speech.pipeline.messages import GenerateResponseRequest


class _FakeCompletions:
    def __init__(self):
        self.next_result = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_result


class _FakeClient:
    def __init__(self, *a, **k):
        self.chat = SimpleNamespace(completions=_FakeCompletions())

    def with_options(self, **kwargs):
        return self


def _freeze_clock(monkeypatch, utc_now: datetime) -> None:
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return utc_now.astimezone(tz) if tz is not None else utc_now

    monkeypatch.setattr(llm_utils, "datetime", _FrozenDatetime)


def _make_handler(monkeypatch, **setup_kwargs):
    monkeypatch.setattr(base_mod, "OpenAI", _FakeClient)
    return ChatCompletionsApiModelHandler(
        threading.Event(),
        queue.Queue(),
        queue.Queue(),
        setup_kwargs=dict(
            model_name="test-model",
            base_url="http://fake/v1",
            api_key="k",
            stream=False,
            disable_thinking=True,
            compact_history=False,
            **setup_kwargs,
        ),
    )


def _sent_system_prompt(handler) -> str:
    chat = Chat(10)
    chat.add_item(make_user_message("Who won the World Series this year?"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="You are a helpful assistant.")
    rc = RuntimeConfig(chat=chat, session=session)
    list(handler.process(GenerateResponseRequest(runtime_config=rc, turn_id="t", turn_revision=0)))

    messages = handler.client.chat.completions.calls[-1]["messages"]
    return next(m["content"] for m in messages if m.get("role") == "system")


def test_system_prompt_starts_with_date_in_configured_timezone(monkeypatch):
    # 03:00 UTC on Nov 5 is still Nov 4 in New York.
    _freeze_clock(monkeypatch, datetime(2026, 11, 5, 3, 0, tzinfo=timezone.utc))
    handler = _make_handler(monkeypatch, llm_current_date=True, llm_timezone="America/New_York")

    prompt = _sent_system_prompt(handler)

    assert prompt.splitlines()[0] == "Today is Wednesday, November 4, 2026."
    assert VOICE_SYSTEM_PROMPT_LEAD.rstrip() in prompt


def test_date_is_computed_per_request_and_rolls_over_at_midnight(monkeypatch):
    handler = _make_handler(monkeypatch, llm_current_date=True, llm_timezone="America/New_York")

    # 23:59 and 00:01 New York time (UTC-5 after DST ends).
    _freeze_clock(monkeypatch, datetime(2026, 11, 5, 4, 59, tzinfo=timezone.utc))
    before = _sent_system_prompt(handler).splitlines()[0]
    _freeze_clock(monkeypatch, datetime(2026, 11, 5, 5, 1, tzinfo=timezone.utc))
    after = _sent_system_prompt(handler).splitlines()[0]

    assert before == "Today is Wednesday, November 4, 2026."
    assert after == "Today is Thursday, November 5, 2026."


def test_no_date_line_when_the_flag_is_disabled(monkeypatch):
    _freeze_clock(monkeypatch, datetime(2026, 11, 5, 3, 0, tzinfo=timezone.utc))
    handler = _make_handler(monkeypatch)

    prompt = _sent_system_prompt(handler)

    assert "Today is" not in prompt
    assert prompt.startswith(VOICE_SYSTEM_PROMPT_LEAD.rstrip())


INVALID_TIMEZONE_MESSAGE = (
    "Unknown --llm_timezone 'Not/AZone': use an IANA name like 'America/New_York' (on Windows, install 'tzdata')"
)


def test_invalid_timezone_raises_clear_value_error():
    with pytest.raises(ValueError) as exc_info:
        llm_utils.prompt_timezone("Not/AZone")

    assert str(exc_info.value) == INVALID_TIMEZONE_MESSAGE


def test_invalid_timezone_fails_at_setup(monkeypatch):
    with pytest.raises(ValueError, match=re.escape(INVALID_TIMEZONE_MESSAGE)):
        _make_handler(monkeypatch, llm_current_date=True, llm_timezone="Not/AZone")


@pytest.mark.parametrize("instructions", ["You are a helpful assistant.", None])
def test_local_handler_prepends_date_line(instructions):
    handler = object.__new__(LanguageModelHandler)
    chat = Chat(10)
    chat.add_item(make_user_message("Hi"))

    handler._apply_instructions(chat, instructions, [], None, current_date="Today is Wednesday, November 4, 2026.")

    system = chat.to_transformers_chat()[0]
    assert system["role"] == "system"
    assert system["content"].splitlines()[0] == "Today is Wednesday, November 4, 2026."


@pytest.mark.parametrize(
    ("backend", "config", "date_key", "tz_key"),
    [
        ("chat-completions", ChatCompletionsLanguageModelHandlerArguments, "llm_current_date", "llm_timezone"),
        ("transformers", LanguageModelHandlerArguments, "current_date", "timezone"),
    ],
)
def test_flags_reach_each_backend_setup(backend, config, date_key, tz_key):
    normalized = LLM_BACKENDS[backend].normalize(config(llm_current_date=True, llm_timezone="Europe/Paris"))

    assert normalized[date_key] is True
    assert normalized[tz_key] == "Europe/Paris"
