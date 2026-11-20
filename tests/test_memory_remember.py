"""Explicit remember by voice: what is admitted, what the model is told about it, and that it never says more."""

from __future__ import annotations

import time
from types import SimpleNamespace

import agentmemory_native
import pytest
from openai.types.realtime.conversation_item import RealtimeConversationItemUserMessage
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.memory.provider import AgentMemoryProvider, forget_intent, remember_intent, render_memory

TEXT = "Please remember that I prefer tea."


def receipt(state: str, *, searchable: bool = False, reason: str | None = None, operation: str = "op-1") -> dict:
    return {"operationId": operation, "sourceId": "src", "duplicate": False, "state": state, "reason": reason,
            "searchable": searchable, "occurredAt": "2026-11-20T10:00:00Z", "acceptedAt": "2026-11-20T10:00:00Z",
            "startedAt": None, "finishedAt": None, "facts": [], "preferences": ["p1"] if searchable else [],
            "closed": []}


def write_result(document: dict | None, *, status: str = "ok", code: str | None = None) -> dict:
    result = {"schemaVersion": 1, "operation": "memory_remember", "requestId": "x", "status": status,
              "timing": {"queuedMs": 0, "runMs": 1}}
    if document is not None:
        result["receipt"] = document
    if code is not None:
        result["error"] = {"code": code, "message": code}
    return result


EMPTY = {"schemaVersion": 1, "operation": "memory_get_context", "requestId": "x", "status": "ok", "facts": [],
         "preferences": [], "truncated": False, "timing": {"queuedMs": 0, "runMs": 1}}
PAUSED = {"schemaVersion": 1, "operation": "memory_get_context", "requestId": "x", "status": "unavailable",
          "error": {"code": "reads_paused", "message": "paused"}, "timing": {"queuedMs": 0, "runMs": 0}}


class FakeClient:
    """The native client, scripted: remember answers in order, then operation_get answers the final receipt."""

    max_pending_jobs = 4

    def __init__(self, config):
        self.config = config
        self.requests: list[dict] = []
        self.remember_answers: list[dict] = []
        self.final: dict | None = None
        self.recall = EMPTY
        self.flags = 7

    def run_job(self, request, timeout_ms=0, stop=None):
        self.requests.append(request)
        operation = request["operation"]
        if operation == "memory_get_context":
            return self.recall
        if operation == "memory_remember":
            return self.remember_answers.pop(0)
        if operation == "memory_forget":
            return write_result(dict(receipt("applied", operation="op-forget"), kind="forget",
                                     targetOperationId=request["operationId"]))
        if operation == "memory_operation_get":
            return write_result(self.final) if self.final is not None else write_result(receipt("running"))
        raise AssertionError(operation)

    def state(self):
        return SimpleNamespace(flags=self.flags)

    def close(self, timeout_s=10.0):
        pass


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(agentmemory_native, "MemoryClient", FakeClient)
    made = AgentMemoryProvider({"profile": "explicit_write"}, deadline_ms=300, max_facts=5, max_preferences=3,
                               min_score=0.75)
    yield made
    made.close()


def user_item(item_id: str, text: str) -> RealtimeConversationItemUserMessage:
    return RealtimeConversationItemUserMessage(
        id=item_id, type="message", role="user", content=[UserContent(type="input_text", text=text)])


def rendered(item_id: str, text: str, preparation) -> list[str]:
    chat = Chat(size=10)
    chat.add_item(user_item(item_id, text))
    render_memory(chat, preparation)
    return [part.text for part in chat.buffer[0].content]


def wait_until(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize("text", [
    "Remember that I prefer tea.", "please remember my locker is 42", "Can you remember that I'm allergic to nuts?",
    "Don't forget I have a meeting at three.", "OK, remember: my sister lives in Bern.",
])
def test_a_spoken_request_to_remember_is_recognised(text):
    assert remember_intent(text)


@pytest.mark.parametrize("text", [
    "Do you remember my name?", "I remember the old house.", "remember", "What should I remember?",
    "Remember that.", "Remember the old house?", "Can you remember where I put my keys?", "", None,
])
def test_anything_else_is_not_a_request_to_remember(text):
    assert not remember_intent(text)


def test_the_read_only_profile_never_admits(monkeypatch):
    monkeypatch.setattr(agentmemory_native, "MemoryClient", FakeClient)
    reader = AgentMemoryProvider({"profile": "read_only"}, deadline_ms=300, max_facts=5, max_preferences=3,
                                 min_score=0.75)
    try:
        assert reader.admit("conn", "item_1", 1, TEXT) is None
        assert "writes" not in reader.status()
        assert all(r["operation"] != "memory_remember" for r in reader._client.requests)
    finally:
        reader.close()


def test_a_spoken_remember_is_admitted_with_its_own_words_and_a_stable_source(provider):
    provider._client.remember_answers.append(write_result(receipt("applied", searchable=True)))
    ticket = provider.admit("conn-1", "item_7", 2, "  " + TEXT + " ")
    assert ticket is not None
    assert wait_until(lambda: ticket.final is not None)
    sent = next(r for r in provider._client.requests if r["operation"] == "memory_remember")
    assert sent["sourceId"] == "s2s/conn-1/item_7/rev-2"
    assert sent["sessionId"] == "conn-1"
    assert sent["text"] == TEXT
    assert sent["occurredAt"].endswith("Z")
    assert "owner" not in sent


def test_the_model_hears_saving_until_the_receipt_supports_saved(provider):
    provider._client.remember_answers.append(write_result(receipt("running")))
    ticket = provider.admit("conn", "item_1", 1, TEXT)
    parts = rendered("item_1", TEXT, provider.prepare("conn", "item_1", TEXT, write=ticket))
    assert len(parts) == 2 and parts[1] == TEXT
    assert "not confirmed yet" in parts[0] and "never that it is saved" in parts[0]
    assert "It is saved." not in parts[0]


def test_the_model_hears_saved_only_for_an_applied_searchable_receipt(provider):
    provider._client.remember_answers.append(write_result(receipt("applied", searchable=True)))
    ticket = provider.admit("conn", "item_1", 1, TEXT)
    parts = rendered("item_1", TEXT, provider.prepare("conn", "item_1", TEXT, write=ticket))
    assert "It is saved." in parts[0]


@pytest.mark.parametrize("answer, expected", [
    (write_result(receipt("rejected", reason="model_unavailable")), "could not be saved"),
    (write_result(None, status="unavailable", code="write_queue_full"), "could not be saved"),
    (write_result(receipt("partial", reason="not_verified")), "not confirmed"),
    (write_result(receipt("applied", reason="nothing_to_keep")), "Nothing in it could be kept"),
])
def test_every_other_outcome_is_told_as_it_is(provider, answer, expected):
    provider._client.remember_answers.append(answer)
    ticket = provider.admit("conn", "item_1", 1, TEXT)
    parts = rendered("item_1", TEXT, provider.prepare("conn", "item_1", TEXT, write=ticket))
    assert expected in parts[0]
    assert "It is saved." not in parts[0]


def test_an_outcome_that_comes_later_reaches_the_next_turn_once(provider):
    provider._client.remember_answers.append(write_result(receipt("running")))
    ticket = provider.admit("conn", "item_1", 1, TEXT)
    first = rendered("item_1", TEXT, provider.prepare("conn", "item_1", TEXT, write=ticket))
    assert "not confirmed yet" in first[0]
    provider._client.final = receipt("applied", searchable=True)
    assert wait_until(lambda: ticket.final is not None)
    second = rendered("item_2", "What's the weather?", provider.prepare("conn", "item_2", "What's the weather?"))
    assert "it is now saved" in second[0] and second[1] == "What's the weather?"
    third = rendered("item_3", "Thanks.", provider.prepare("conn", "item_3", "Thanks."))
    assert third == ["Thanks."], "an outcome is told once"


def test_another_connection_never_hears_this_ones_outcomes(provider):
    provider._client.remember_answers.append(write_result(receipt("applied", searchable=True)))
    ticket = provider.admit("conn-a", "item_1", 1, TEXT)
    assert wait_until(lambda: ticket.final is not None)
    assert rendered("item_9", "Hello.", provider.prepare("conn-b", "item_9", "Hello.")) == ["Hello."]


def test_a_paused_recall_is_not_trouble_and_adds_no_evidence(provider):
    provider._client.recall = PAUSED
    preparation = provider.prepare("conn", "item_1", "Where do I work?")
    calls = []
    preparation.on_trouble = lambda: calls.append(1)
    assert rendered("item_1", "Where do I work?", preparation) == ["Where do I work?"]
    assert calls == []


def test_status_says_saving_and_needing_attention_without_any_content(provider):
    provider._client.remember_answers.append(write_result(receipt("running")))
    provider.admit("conn", "item_1", 1, TEXT)
    assert wait_until(lambda: provider.status()["writes"]["saving"] == 1)
    status = provider.status()
    assert status["profile"] == "explicit_write" and status["writes"]["voiceRemember"] is True
    provider._client.flags = 2 | 4  # quiescent, recall paused: an uncertain write waits for a decision
    provider._client.final = receipt("unknown", reason="store_failed_while_writing: X")
    assert wait_until(lambda: provider.status()["writes"] == {"voiceRemember": True, "saving": 0,
                                                               "needsAttention": True})
    assert "tea" not in str(provider.status())


@pytest.mark.parametrize("text", [
    "Forget what I just told you.", "Please forget that.", "Could you forget what I said?", "OK, forget that last one.",
])
def test_a_spoken_request_to_forget_is_recognised(text):
    assert forget_intent(text) and not remember_intent(text)


@pytest.mark.parametrize("text", [
    "Forget it.", "Forget that, let's talk about cars.", "I forgot my keys.", "Did you forget what I said?", "", None,
])
def test_anything_else_is_not_a_request_to_forget(text):
    assert not forget_intent(text)


def test_forget_names_the_last_remember_of_this_connection_and_says_it_will_no_longer_be_used(provider):
    provider._client.remember_answers.append(write_result(receipt("applied", searchable=True, operation="op-tea")))
    remembered = provider.admit("conn", "item_1", 1, TEXT)
    assert wait_until(lambda: remembered.final is not None)
    ticket = provider.admit("conn", "item_2", 1, "Forget what I just told you.")
    assert ticket is not None and ticket.kind == "forget"
    parts = rendered("item_2", "Forget what I just told you.",
                     provider.prepare("conn", "item_2", "Forget what I just told you.", write=ticket))
    sent = next(r for r in provider._client.requests if r["operation"] == "memory_forget")
    assert sent["operationId"] == "op-tea"
    assert "It will no longer be used." in parts[0]
    assert "deleted" not in parts[0], "a forget closes memories; it never claims they are deleted"


def test_forget_with_nothing_remembered_here_says_so_and_asks_nothing_of_the_library(provider):
    ticket = provider.admit("conn", "item_1", 1, "Please forget that.")
    parts = rendered("item_1", "Please forget that.", provider.prepare("conn", "item_1", "Please forget that.", write=ticket))
    assert "nothing to forget" in parts[0]
    assert all(r["operation"] != "memory_forget" for r in provider._client.requests)


def test_forget_never_reaches_another_connections_remember(provider):
    provider._client.remember_answers.append(write_result(receipt("applied", searchable=True, operation="op-a")))
    remembered = provider.admit("conn-a", "item_1", 1, TEXT)
    assert wait_until(lambda: remembered.final is not None)
    ticket = provider.admit("conn-b", "item_9", 1, "Forget what I just told you.")
    assert ticket.refused == "nothing_to_forget"
