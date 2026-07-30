"""End-to-end coverage for the ``--enable_lang_prompt`` language instruction.

The instruction is gated on a language *name*, not on the language code:

    language_code, lang_name = resolve_auto_language(language_code)
    if lang_name and self.enable_lang_prompt:
        active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

so any code missing from ``WHISPER_LANGUAGE_TO_LLM_LANGUAGE`` silently produces no
instruction at all. Parakeet TDT is the default STT and reports 25 languages, so this
asserts the instruction actually reaches the outgoing request for the ones it detects --
not merely that the mapping dict has keys.

The OpenAI client is faked, so this runs with no network and no GPU.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.STT.parakeet_tdt_handler import SUPPORTED_LANGUAGES as PARAKEET_LANGUAGES


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


def _make_handler(*, enable_lang_prompt):
    orig_openai = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
    try:
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
                enable_lang_prompt=enable_lang_prompt,
            ),
        )
    finally:
        base_mod.OpenAI = orig_openai


def _sent_messages(handler, language_code, *, enable_lang_prompt=True):
    """Drive one request and return the messages the backend actually sent."""
    chat = Chat(10)
    chat.add_item(make_user_message("Hej, hur mar du?"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="You are a robot.")
    rc = RuntimeConfig(chat=chat, session=session)
    req = GenerateResponseRequest(runtime_config=rc, language_code=language_code, turn_id="t", turn_revision=0)

    list(handler.process(req))

    calls = handler.client.chat.completions.calls
    assert calls, "the backend never issued a request"
    return [m.get("content") for m in calls[-1]["messages"] if m.get("role") == "user"]


def test_swedish_gets_a_language_instruction():
    """Swedish is one of the 17 Parakeet languages that previously got nothing."""
    handler = _make_handler(enable_lang_prompt=True)

    contents = _sent_messages(handler, "sv-auto")

    assert "Please reply to my message in swedish." in contents


@pytest.mark.parametrize("code", sorted(PARAKEET_LANGUAGES))
def test_every_parakeet_language_produces_an_instruction(code):
    """No language the default STT can report may silently skip the instruction."""
    handler = _make_handler(enable_lang_prompt=True)

    contents = _sent_messages(handler, f"{code}-auto")

    instructions = [c for c in contents if isinstance(c, str) and c.startswith("Please reply to my message in ")]
    assert len(instructions) == 1, f"no language instruction emitted for {code!r}"


def test_no_instruction_when_the_flag_is_disabled():
    """The flag still gates the instruction; this is the default configuration."""
    handler = _make_handler(enable_lang_prompt=False)

    contents = _sent_messages(handler, "sv-auto")

    assert not [c for c in contents if isinstance(c, str) and c.startswith("Please reply to my message in ")]


def test_unknown_language_code_still_emits_no_instruction():
    """An unnameable code must not produce 'reply in None.' -- absent is correct here."""
    handler = _make_handler(enable_lang_prompt=True)

    contents = _sent_messages(handler, "xx-auto")

    assert not [c for c in contents if isinstance(c, str) and c.startswith("Please reply to my message in ")]
