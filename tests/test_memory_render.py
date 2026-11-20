"""The optional long-term memory: what is rendered into a request, where, and when nothing is."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

from openai.types.realtime.conversation_item import RealtimeConversationItemUserMessage
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.memory import MemoryPreparation, memory_provider, render_memory, reset_memory_provider
from speech_to_speech.memory.provider import _sha256

RESULT = {
    "schemaVersion": 1, "operation": "memory_get_context", "requestId": "item_1", "status": "ok",
    "facts": [{"id": "f1", "subject": "user", "predicate": "works_in", "object": "Zurich", "confidence": 0.9,
               "category": None, "validFrom": None, "validUntil": None, "createdAt": "2026-11-20T00:00:00+00:00",
               "metadata": {}}],
    "preferences": [{"id": "p1", "category": "style", "text": "Prefers concise answers", "context": None,
                     "confidence": 0.9, "createdAt": "2026-11-20T00:00:00+00:00", "metadata": {}}],
    "truncated": False, "timing": {"queuedMs": 0, "runMs": 2},
}


def user_item(item_id: str, text: str) -> RealtimeConversationItemUserMessage:
    return RealtimeConversationItemUserMessage(
        id=item_id, type="message", role="user", content=[UserContent(type="input_text", text=text)])


def chat_with(*items: RealtimeConversationItemUserMessage) -> Chat:
    chat = Chat(size=10)
    for item in items:
        chat.add_item(item)
    return chat


def preparation(item_id: str, text: str, result=RESULT, *, deadline_s: float = 1.0, done: bool = True) -> MemoryPreparation:
    future: Future = Future()
    if done:
        future.set_result(result)
    return MemoryPreparation(item_id, _sha256(text), future, threading.Event(), time.monotonic() + deadline_s)


def texts(chat: Chat, item_id: str) -> list[str]:
    item = next(i for i in chat.buffer if getattr(i, "id", None) == item_id)
    return [part.text for part in item.content]


def test_memory_is_off_without_configuration(monkeypatch):
    monkeypatch.delenv("S2S_MEMORY_CONFIG", raising=False)
    reset_memory_provider(None)
    assert memory_provider() is None


def test_evidence_goes_into_the_anchored_item_of_the_copy_only():
    canonical = chat_with(user_item("item_0", "Hello"), user_item("item_1", "Where do I work?"))
    copy = canonical.copy(deep=True)

    assert render_memory(copy, preparation("item_1", "Where do I work?")) == 2

    rendered = texts(copy, "item_1")
    assert len(rendered) == 2 and rendered[1] == "Where do I work?"
    assert "[memory]\n- user works in Zurich\n- Prefers concise answers\n[/memory]" in rendered[0]
    assert texts(copy, "item_0") == ["Hello"], "only the anchored item gets evidence"
    assert texts(canonical, "item_1") == ["Where do I work?"], "the stored conversation never does"


def test_a_turn_whose_text_changed_gets_no_memory():
    copy = chat_with(user_item("item_1", "Where do I work now?"))
    assert render_memory(copy, preparation("item_1", "Where do I work?")) == 0
    assert texts(copy, "item_1") == ["Where do I work now?"]


def test_a_missing_anchor_gets_no_memory_and_no_other_item_is_used():
    copy = chat_with(user_item("item_9", "Where do I work?"))
    assert render_memory(copy, preparation("item_1", "Where do I work?")) == 0
    assert texts(copy, "item_9") == ["Where do I work?"]


def test_a_late_recall_is_dropped_and_its_job_is_told_to_stop():
    late = preparation("item_1", "Where do I work?", done=False, deadline_s=0.05)
    copy = chat_with(user_item("item_1", "Where do I work?"))
    started = time.monotonic()
    assert render_memory(copy, late) == 0
    assert time.monotonic() - started < 0.5, "never waits past the deadline"
    assert late.stop.is_set()


def test_a_failed_recall_and_an_empty_one_add_nothing():
    failed = dict(RESULT, status="unavailable", facts=[], preferences=[],
                  error={"code": "store_unavailable", "message": "down"})
    empty = dict(RESULT, facts=[], preferences=[])
    for result in (failed, empty):
        copy = chat_with(user_item("item_1", "Where do I work?"))
        assert render_memory(copy, preparation("item_1", "Where do I work?", result)) == 0
        assert texts(copy, "item_1") == ["Where do I work?"]


def test_a_follow_up_of_a_late_turn_gets_no_memory_even_if_the_result_arrives_later():
    late = preparation("item_1", "Where do I work?", done=False, deadline_s=0.01)
    assert render_memory(chat_with(user_item("item_1", "Where do I work?")), late) == 0
    late.future.set_result(RESULT)
    copy = chat_with(user_item("item_1", "Where do I work?"))
    assert render_memory(copy, late) == 0
    assert texts(copy, "item_1") == ["Where do I work?"]


def test_an_undecodable_result_adds_nothing_and_does_not_raise():
    copy = chat_with(user_item("item_1", "Where do I work?"))
    assert render_memory(copy, preparation("item_1", "Where do I work?", {"no": "status"})) == 0


def test_a_late_failed_or_refused_recall_reports_trouble_and_a_good_one_does_not():
    failed = dict(RESULT, status="unavailable", facts=[], preferences=[],
                  error={"code": "store_unavailable", "message": "down"})
    cases = [
        (preparation("item_1", "Where do I work?", done=False, deadline_s=0.01), 1),
        (preparation("item_1", "Where do I work?", failed), 1),
        (preparation("item_1", "Where do I work?", {"no": "status"}), 1),
        (preparation("item_1", "Where do I work?"), 0),
    ]
    for prepared, expected in cases:
        calls = []
        prepared.on_trouble = lambda: calls.append(1)
        render_memory(chat_with(user_item("item_1", "Where do I work?")), prepared)
        assert len(calls) == expected


def test_memory_status_with_memory_off_says_only_that():
    from speech_to_speech.memory import memory_status

    reset_memory_provider(None)
    assert memory_status() == {"enabled": False}


def test_with_memory_off_the_server_never_imports_the_memory_library():
    """A fresh interpreter, as a real server: build the realtime service with S2S_MEMORY_CONFIG unset."""
    import os
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "from speech_to_speech.api.openai_realtime.service import RealtimeService\n"
        "RealtimeService()\n"
        "print('agentmemory_native' in sys.modules)\n"
    )
    environment = {k: v for k, v in os.environ.items() if k != "S2S_MEMORY_CONFIG"}
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=environment,
                               timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().splitlines()[-1] == "False"


def test_no_preparation_renders_nothing():
    copy = chat_with(user_item("item_1", "Where do I work?"))
    assert render_memory(copy, None) == 0
