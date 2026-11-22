"""Explicit remember by voice: what is admitted, what the model is told about it, and that it never says more."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from openai.types.realtime.conversation_item import RealtimeConversationItemUserMessage
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.memory.provider import (
    AgentMemoryProvider,
    forget_intent,
    memory_erase,
    observable,
    remember_intent,
    render_memory,
    reset_memory_provider,
)

TEXT = "Please remember that I prefer tea."


def receipt(state: str, *, searchable: bool = False, reason: str | None = None, operation: str = "op-1") -> dict:
    return {
        "operationId": operation,
        "sourceId": "src",
        "duplicate": False,
        "state": state,
        "reason": reason,
        "searchable": searchable,
        "occurredAt": "2026-11-20T10:00:00Z",
        "acceptedAt": "2026-11-20T10:00:00Z",
        "startedAt": None,
        "finishedAt": None,
        "facts": [],
        "preferences": ["p1"] if searchable else [],
        "closed": [],
    }


def write_result(document: dict | None, *, status: str = "ok", code: str | None = None) -> dict:
    result = {
        "schemaVersion": 1,
        "operation": "memory_remember",
        "requestId": "x",
        "status": status,
        "timing": {"queuedMs": 0, "runMs": 1},
    }
    if document is not None:
        result["receipt"] = document
    if code is not None:
        result["error"] = {"code": code, "message": code}
    return result


EMPTY = {
    "schemaVersion": 1,
    "operation": "memory_get_context",
    "requestId": "x",
    "status": "ok",
    "facts": [],
    "preferences": [],
    "truncated": False,
    "timing": {"queuedMs": 0, "runMs": 1},
}
PAUSED = {
    "schemaVersion": 1,
    "operation": "memory_get_context",
    "requestId": "x",
    "status": "unavailable",
    "error": {"code": "reads_paused", "message": "paused"},
    "timing": {"queuedMs": 0, "runMs": 0},
}


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
        self.erase_answer = write_result(dict(receipt("applied", operation="op-erase"), kind="erase", deleted=7))

    def run_job(self, request, timeout_ms=0, stop=None):
        self.requests.append(request)
        operation = request["operation"]
        if operation == "memory_get_context":
            return self.recall
        if operation == "memory_remember":
            return self.remember_answers.pop(0)
        if operation == "memory_forget":
            return write_result(
                dict(receipt("applied", operation="op-forget"), kind="forget", targetOperationId=request["operationId"])
            )
        if operation == "memory_erase_owner":
            return self.erase_answer
        if operation == "memory_operation_get":
            return write_result(self.final) if self.final is not None else write_result(receipt("running"))
        raise AssertionError(operation)

    def state(self):
        return SimpleNamespace(flags=self.flags)

    def close(self, timeout_s=10.0):
        pass


def use_client(monkeypatch, client_class) -> None:
    """Install a scripted native client; the tests that need it skip without the optional memory extra."""
    native = pytest.importorskip("agentmemory_native", reason='needs the optional "memory" extra (agentmemory-native)')
    monkeypatch.setattr(native, "MemoryClient", client_class)


@pytest.fixture
def provider(monkeypatch):
    use_client(monkeypatch, FakeClient)
    made = AgentMemoryProvider(
        {"profile": "explicit_write"}, deadline_ms=300, max_facts=5, max_preferences=3, min_score=0.75
    )
    yield made
    made.close()


def user_item(item_id: str, text: str) -> RealtimeConversationItemUserMessage:
    return RealtimeConversationItemUserMessage(
        id=item_id, type="message", role="user", content=[UserContent(type="input_text", text=text)]
    )


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


@pytest.mark.parametrize(
    "text",
    [
        "Remember that I prefer tea.",
        "please remember my locker is 42",
        "Can you remember that I'm allergic to nuts?",
        "Don't forget I have a meeting at three.",
        "OK, remember: my sister lives in Bern.",
    ],
)
def test_a_spoken_request_to_remember_is_recognised(text):
    assert remember_intent(text)


@pytest.mark.parametrize(
    "text",
    [
        "Do you remember my name?",
        "I remember the old house.",
        "remember",
        "What should I remember?",
        "Remember that.",
        "Remember the old house?",
        "Can you remember where I put my keys?",
        "",
        None,
    ],
)
def test_anything_else_is_not_a_request_to_remember(text):
    assert not remember_intent(text)


def test_the_read_only_profile_never_admits(monkeypatch):
    use_client(monkeypatch, FakeClient)
    reader = AgentMemoryProvider(
        {"profile": "read_only"}, deadline_ms=300, max_facts=5, max_preferences=3, min_score=0.75
    )
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


@pytest.mark.parametrize(
    "answer, expected",
    [
        (write_result(receipt("rejected", reason="model_unavailable")), "could not be saved"),
        (write_result(None, status="unavailable", code="write_queue_full"), "could not be saved"),
        (write_result(receipt("partial", reason="not_verified")), "not confirmed"),
        (write_result(receipt("applied", reason="nothing_to_keep")), "Nothing in it could be kept"),
    ],
)
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
    assert wait_until(
        lambda: (
            provider.status()["writes"]
            == {"voiceRemember": True, "observing": False, "saving": 0, "needsAttention": True}
        )
    )
    assert "tea" not in str(provider.status())


@pytest.mark.parametrize(
    "text",
    [
        "Forget what I just told you.",
        "Please forget that.",
        "Could you forget what I said?",
        "OK, forget that last one.",
    ],
)
def test_a_spoken_request_to_forget_is_recognised(text):
    assert forget_intent(text) and not remember_intent(text)


@pytest.mark.parametrize(
    "text",
    [
        "Forget it.",
        "Forget that, let's talk about cars.",
        "I forgot my keys.",
        "Did you forget what I said?",
        "",
        None,
    ],
)
def test_anything_else_is_not_a_request_to_forget(text):
    assert not forget_intent(text)


def test_forget_names_the_last_remember_of_this_connection_and_says_it_will_no_longer_be_used(provider):
    provider._client.remember_answers.append(write_result(receipt("applied", searchable=True, operation="op-tea")))
    remembered = provider.admit("conn", "item_1", 1, TEXT)
    assert wait_until(lambda: remembered.final is not None)
    ticket = provider.admit("conn", "item_2", 1, "Forget what I just told you.")
    assert ticket is not None and ticket.kind == "forget"
    parts = rendered(
        "item_2",
        "Forget what I just told you.",
        provider.prepare("conn", "item_2", "Forget what I just told you.", write=ticket),
    )
    sent = next(r for r in provider._client.requests if r["operation"] == "memory_forget")
    assert sent["operationId"] == "op-tea"
    assert "It will no longer be used." in parts[0]
    assert "deleted" not in parts[0], "a forget closes memories; it never claims they are deleted"


def test_forget_with_nothing_remembered_here_says_so_and_asks_nothing_of_the_library(provider):
    ticket = provider.admit("conn", "item_1", 1, "Please forget that.")
    parts = rendered(
        "item_1", "Please forget that.", provider.prepare("conn", "item_1", "Please forget that.", write=ticket)
    )
    assert "nothing to forget" in parts[0]
    assert all(r["operation"] != "memory_forget" for r in provider._client.requests)


def test_forget_never_reaches_another_connections_remember(provider):
    provider._client.remember_answers.append(write_result(receipt("applied", searchable=True, operation="op-a")))
    remembered = provider.admit("conn-a", "item_1", 1, TEXT)
    assert wait_until(lambda: remembered.final is not None)
    ticket = provider.admit("conn-b", "item_9", 1, "Forget what I just told you.")
    assert ticket.refused == "nothing_to_forget"


def test_owner_erase_needs_the_control_token_and_a_confirmation(provider, monkeypatch, tmp_path):
    reset_memory_provider(provider)
    try:
        monkeypatch.delenv("S2S_MEMORY_CONTROL_TOKEN_FILE", raising=False)
        assert memory_erase("Bearer x", {"confirm": True})[0] == 404, "no token file: the control is off"
        token = tmp_path / "token"
        token.write_text("s3cret-token\n")
        monkeypatch.setenv("S2S_MEMORY_CONTROL_TOKEN_FILE", str(token))
        assert memory_erase(None, {"confirm": True})[0] == 401
        assert memory_erase("Bearer wrong", {"confirm": True})[0] == 401
        assert memory_erase("Bearer s3cret-token", {"confirm": "yes"})[0] == 400
        assert all(r["operation"] != "memory_erase_owner" for r in provider._client.requests)
        status, document = memory_erase("Bearer s3cret-token", {"confirm": True})
        assert (status, document["state"], document["deleted"]) == (200, "applied", 7)
        assert "s3cret" not in str(document)
        provider._client.erase_answer = write_result(None, status="unavailable", code="writes_pending")
        assert memory_erase("Bearer s3cret-token", {"confirm": True}) == (409, {"error": "writes_pending"})
    finally:
        reset_memory_provider(None)


def test_a_read_only_server_offers_no_erase(monkeypatch, tmp_path):
    use_client(monkeypatch, FakeClient)
    reader = AgentMemoryProvider(
        {"profile": "read_only"}, deadline_ms=300, max_facts=5, max_preferences=3, min_score=0.75
    )
    token = tmp_path / "token"
    token.write_text("t")
    monkeypatch.setenv("S2S_MEMORY_CONTROL_TOKEN_FILE", str(token))
    reset_memory_provider(reader)
    try:
        assert memory_erase("Bearer t", {"confirm": True})[0] == 404
    finally:
        reset_memory_provider(None)
        reader.close()


@pytest.mark.parametrize(
    "text",
    [
        "My sister lives in Bern.",
        "I started a new job at the library last week.",
        "We adopted a puppy called Nala.",
        # Told, then asked: the question at the end must not hide what was told.
        "Hey, yesterday I had dinner with my old classmates and I met their new teacher. So today I joined her course. "
        "What do you think?",
        "His name is Sam. Do you remember now?",
    ],
)
def test_what_the_owner_tells_is_worth_learning(text):
    assert observable(text)


@pytest.mark.parametrize(
    "text",
    [
        "Yes.",
        "Thanks, robot!",
        "Good morning.",
        "What time is it?",
        "Okay sure.",
        "Sounds good",
        "",
        None,
        "You know, I told you, right?",
        "Hey. How are you?",
        "Yes. Thanks. What did I do yesterday?",
        "Okay. Bye!",
    ],
)
def test_questions_assent_and_greetings_are_not(text):
    assert not observable(text)


def learning(monkeypatch, observe: bool) -> AgentMemoryProvider:
    use_client(monkeypatch, FakeClient)
    return AgentMemoryProvider(
        {"profile": "learning"}, deadline_ms=300, max_facts=5, max_preferences=3, min_score=0.75, observe=observe
    )


def test_with_learning_on_a_told_fact_is_remembered_unasked_and_silently(monkeypatch):
    provider = learning(monkeypatch, observe=True)
    try:
        provider._client.remember_answers.append(write_result(receipt("applied", searchable=True, operation="op-nala")))
        ticket = provider.admit("conn", "item_1", 1, "We adopted a puppy called Nala.")
        assert ticket is not None and ticket.kind == "observe"
        assert wait_until(lambda: any(r["operation"] == "memory_remember" for r in provider._client.requests))
        sent = next(r for r in provider._client.requests if r["operation"] == "memory_remember")
        assert sent["origin"] == "observed"
        parts = rendered(
            "item_1",
            "We adopted a puppy called Nala.",
            provider.prepare("conn", "item_1", "We adopted a puppy called Nala.", write=ticket),
        )
        assert parts == ["We adopted a puppy called Nala."], "an unasked write is never mentioned to the model"
        assert wait_until(lambda: ticket.final is not None)
        later = rendered("item_2", "How are you?", provider.prepare("conn", "item_2", "How are you?"))
        assert later == ["How are you?"]
        provider.admit("conn", "item_3", 1, "Forget what I just told you.")
        assert wait_until(lambda: any(r["operation"] == "memory_forget" for r in provider._client.requests))
        forget = next(r for r in provider._client.requests if r["operation"] == "memory_forget")
        assert forget["operationId"] == "op-nala"
        assert provider.status()["writes"]["observing"] is True
    finally:
        provider.close()


def test_with_learning_off_only_explicit_requests_are_remembered(monkeypatch):
    provider = learning(monkeypatch, observe=False)
    try:
        assert provider.admit("conn", "item_1", 1, "We adopted a puppy called Nala.") is None
        assert provider.status()["writes"]["observing"] is False
    finally:
        provider.close()


def test_the_explicit_profile_never_learns_unasked(monkeypatch):
    use_client(monkeypatch, FakeClient)
    explicit = AgentMemoryProvider(
        {"profile": "explicit_write"}, deadline_ms=300, max_facts=5, max_preferences=3, min_score=0.75, observe=True
    )
    try:
        assert explicit.observe is False
        assert explicit.admit("conn", "item_1", 1, "We adopted a puppy called Nala.") is None
    finally:
        explicit.close()


def test_a_write_still_running_after_its_client_hung_up_is_still_saving(monkeypatch):
    provider = learning(monkeypatch, observe=True)
    try:
        provider._client.remember_answers.append(write_result(receipt("running", operation="op-tea")))
        ticket = provider.admit("conn", "item_1", 1, "My favourite tea is Earl Grey.")
        assert wait_until(lambda: ticket.future.done())
        provider._client.flags = 7  # a flag read before the write was accepted: quiescent
        provider._refresh_flags()
        provider._client.flags = 1 | 2  # the library now: a write pending, recall open (learning)
        provider.forget_connection("conn")
        assert provider.status()["writes"]["saving"] == 1
        provider._client.flags = 7
        provider._client.final = receipt("applied", searchable=True, operation="op-tea")
        assert wait_until(lambda: provider.status()["writes"]["saving"] == 0)
    finally:
        provider.close()


def test_startup_warms_the_store_with_a_series_of_read_only_recalls(monkeypatch):
    use_client(monkeypatch, FakeClient)
    provider = AgentMemoryProvider(
        {"profile": "learning"}, deadline_ms=300, max_facts=5, max_preferences=3, min_score=0.75
    )
    try:
        expected = len(AgentMemoryProvider.WARM_UP_WORDS)
        assert wait_until(
            lambda: sum(r["operation"] == "memory_get_context" for r in provider._client.requests) == expected
        )
        warm = [r for r in provider._client.requests if r["operation"] == "memory_get_context"]
        assert len({r["query"] for r in warm}) == expected, "different words each time"
        assert all(r["maxFacts"] == 5 and r["maxPreferences"] == 3 for r in warm), "the size a turn asks for"
        assert not any(r["operation"] in ("memory_remember", "memory_forget") for r in provider._client.requests)
    finally:
        provider.close()


def test_a_failed_first_warm_up_does_not_run_the_rest(monkeypatch):
    class StoreDown(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recall = {
                "schemaVersion": 1,
                "operation": "memory_get_context",
                "requestId": "x",
                "status": "unavailable",
                "error": {"code": "store_unavailable", "message": "down"},
                "timing": {"queuedMs": 0, "runMs": 1},
            }

    use_client(monkeypatch, StoreDown)
    provider = AgentMemoryProvider(
        {"profile": "learning"}, deadline_ms=300, max_facts=5, max_preferences=3, min_score=0.75
    )
    try:
        time.sleep(0.3)
        assert sum(r["operation"] == "memory_get_context" for r in provider._client.requests) == 1
    finally:
        provider.close()


RECALLED = {
    "schemaVersion": 1,
    "operation": "memory_get_context",
    "requestId": "x",
    "status": "ok",
    "facts": [
        {
            "id": "f1",
            "subject": "user",
            "predicate": "is_named",
            "object": "Alex",
            "confidence": 0.9,
            "category": None,
            "validFrom": None,
            "validUntil": None,
            "createdAt": "2026-11-20T00:00:00+00:00",
            "metadata": {},
        }
    ],
    "preferences": [
        {
            "id": "p1",
            "category": "likes",
            "text": "Likes hiking",
            "context": None,
            "confidence": 0.9,
            "createdAt": "2026-11-20T00:00:00+00:00",
            "metadata": {},
        }
    ],
    "truncated": False,
    "timing": {"queuedMs": 0, "runMs": 2},
}


def memory_lines(parts: list[str]) -> list[str]:
    block = next((p for p in parts if "[memory]" in p), "")
    return [line for line in block.split("\n") if line.startswith("- ")]


def test_a_later_turn_keeps_what_was_recalled_earlier_in_the_session(provider):
    """A turn recalls the name and the next recalls nothing: without the evidence, the model may take back what it had
    said. A memory stays true after the turn that recalled it, so later turns of the session carry it."""
    provider._client.recall = RECALLED
    first = rendered(
        "item_1",
        "I am Alex, what do you know about me?",
        provider.prepare("conn", "item_1", "I am Alex, what do you know about me?"),
    )
    assert memory_lines(first) == ["- user is named Alex", "- Likes hiking"]

    provider._client.recall = EMPTY
    later = rendered("item_2", "How am I today?", provider.prepare("conn", "item_2", "How am I today?"))
    assert memory_lines(later) == ["- user is named Alex", "- Likes hiking"]

    other = rendered("item_3", "How am I today?", provider.prepare("other-conn", "item_3", "How am I today?"))
    assert memory_lines(other) == [], "another session starts with nothing carried"


def test_a_memory_a_later_write_closes_is_no_longer_carried(provider):
    provider._client.recall = RECALLED
    rendered("item_1", "What do you know about me?", provider.prepare("conn", "item_1", "What do you know about me?"))
    replaced = dict(receipt("applied", searchable=True, operation="op-2"), closed=["p1"])
    provider._client.remember_answers.append(write_result(replaced))
    ticket = provider.admit("conn", "item_2", 1, "Please remember that I do not like hiking any more.")
    assert wait_until(lambda: ticket.final is not None)

    provider._client.recall = EMPTY
    later = rendered("item_3", "And what else?", provider.prepare("conn", "item_3", "And what else?"))
    assert memory_lines(later) == ["- user is named Alex"], "a replaced or forgotten memory is never carried on"


def test_closing_the_session_or_erasing_drops_carried_memories(provider):
    provider._client.recall = RECALLED
    rendered("item_1", "Hello", provider.prepare("conn", "item_1", "Hello"))
    provider.forget_connection("conn")
    provider._client.recall = EMPTY
    assert memory_lines(rendered("item_2", "Hi", provider.prepare("conn", "item_2", "Hi"))) == []

    provider._client.recall = RECALLED
    rendered("item_3", "Hello", provider.prepare("conn-2", "item_3", "Hello"))
    assert provider.erase_owner()["accepted"]
    provider._client.recall = EMPTY
    assert memory_lines(rendered("item_4", "Hi", provider.prepare("conn-2", "item_4", "Hi"))) == []


def test_at_most_the_cap_of_memories_goes_with_a_turn(provider):
    many = dict(RECALLED, facts=[dict(RECALLED["facts"][0], id=f"f{i}", object=f"N{i}") for i in range(50)])
    provider._client.recall = many
    rendered("item_1", "a", provider.prepare("conn", "item_1", "a"))
    more = dict(RECALLED, facts=[dict(RECALLED["facts"][0], id=f"g{i}", object=f"M{i}") for i in range(5)])
    provider._client.recall = more
    lines = memory_lines(rendered("item_2", "b", provider.prepare("conn", "item_2", "b")))
    assert len(lines) == provider.CARRY_MAX
    assert lines[:6] == [f"- user is named M{i}" for i in range(5)] + ["- Likes hiking"], "this turn's recall first"


def test_a_new_conversation_starts_with_what_is_remembered_about_the_owner(provider):
    """A new conversation's first questions can recall nothing (under the cut-off, or past the deadline while speech
    recognition loads the CPU); what is recalled at connect fills that gap."""
    provider._client.recall = RECALLED
    provider.connected("conn")
    assert wait_until(lambda: len(provider._carried.get("conn", {})) == 2)
    assert sum(r.get("requestId") == "s2s-start" for r in provider._client.requests) == len(
        provider.SESSION_START_WORDS
    )

    provider._client.recall = EMPTY
    first = rendered("item_1", "How are you? Who am I?", provider.prepare("conn", "item_1", "How are you? Who am I?"))
    assert memory_lines(first) == ["- user is named Alex", "- Likes hiking"]
    other = rendered("item_2", "Who am I?", provider.prepare("never-connected", "item_2", "Who am I?"))
    assert memory_lines(other) == []


def test_a_conversation_closed_or_erased_before_its_start_recall_ends_starts_with_nothing(provider):
    provider._client.recall = RECALLED
    with provider._lock:
        provider._open.add("closed")
        provider._open.add("erased")
        erasures = provider._erasures
    provider.forget_connection("closed")
    provider._start_conversation("closed", erasures)
    assert "closed" not in provider._carried, "a closed connection keeps nothing behind"

    assert provider.erase_owner()["accepted"]
    provider._start_conversation("erased", erasures)
    assert "erased" not in provider._carried, "memories recalled before an erase are not carried after it"


SECTIONS = {"max_messages": 5, "max_entities": 10, "max_relationships": 10, "time_windows": True}
FULL = dict(
    RECALLED,
    windowFacts=[
        {
            "id": "w1",
            "subject": "user",
            "predicate": "had_dinner_with",
            "object": "friends on 19 November 2026",
            "confidence": 0.9,
            "category": None,
            "validFrom": None,
            "validUntil": None,
            "createdAt": "2026-11-20T00:00:00+00:00",
            "metadata": {},
        }
    ],
    relationships=[{"id": "r1", "source": "Sam", "type": "WORKS_WITH", "target": "the user", "description": None}],
    entities=[{"id": "e1", "name": "Sam", "type": "PERSON", "description": "a colleague"}],
    messages=[
        {"id": "m1", "role": "user", "text": "Yesterday I had dinner with  friends.", "at": "2026-11-19T20:00:00Z"},
        {"id": "m2", "role": "user", "text": "Who is Sam?", "at": "2026-11-20T20:00:00Z"},
    ],
)


def test_every_recall_asks_for_the_same_sections(monkeypatch):
    """A turn's recall, a conversation's start and the warm-up ask the same way (one helper, no second copy)."""
    use_client(monkeypatch, FakeClient)
    made = AgentMemoryProvider(
        {"profile": "read_only"}, deadline_ms=300, max_facts=10, max_preferences=5, min_score=0.7, sections=SECTIONS
    )
    try:
        made.connected("conn")
        made.prepare("conn", "item_1", "Who is Sam?")
        assert wait_until(
            lambda: {r["requestId"] for r in made._client.requests} >= {"s2s-warm-up", "s2s-start", "item_1"}
        )
        for request in made._client.requests:
            assert (request["maxFacts"], request["maxPreferences"], request["minScore"]) == (10, 5, 0.7)
            assert (request["maxMessages"], request["maxEntities"], request["maxRelationships"]) == (5, 10, 10)
            assert request["timeWindows"] is True and "details" not in request
    finally:
        made.close()


def test_people_relations_dated_facts_and_past_words_reach_the_model(provider):
    provider._client.recall = FULL
    lines = memory_lines(rendered("item_1", "Who is Sam?", provider.prepare("conn", "item_1", "Who is Sam?")))
    assert lines == [
        "- user is named Alex",
        "- user had dinner with friends on 19 November 2026",
        "- Likes hiking",
        "- Sam works with the user",
        "- Sam (person): a colleague",
        '- earlier (19 November 2026) the user said: "Yesterday I had dinner with friends."',
    ], "the turn's own words, already saved, do not come back as a past word"
