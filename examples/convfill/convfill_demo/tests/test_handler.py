from __future__ import annotations

import queue
import threading
from queue import Queue
from types import SimpleNamespace

from convfill_demo.backend import register_convfill_backend
from convfill_demo.delivery import AUDIO_SAMPLE_RATE_HZ, DeliveryLedgerQueue
from convfill_demo.handler import ConvFillRealtimeHandler, latest_user_text
from convfill_demo.runtime.dialogue_state_manager import DialogueStateManager
from convfill_demo.runtime.run_convfill import ConvFillSystem
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AudioOutput,
    EndOfResponse,
    LLMResponseChunk,
)


class _FakeChat:
    def __init__(self, messages=None):
        self.messages = messages or [{"role": "user", "content": "What is ConvFill?"}]
        self.provisional = []
        self.trimmed = False

    def to_transformers_chat(self):
        return list(self.messages)

    def add_provisional_generation_items(self, response_key, items):
        self.provisional.append((response_key, items))
        return items

    def trim_if_needed(self):
        self.trimmed = True


class _FakeSystem:
    def __init__(self, **callbacks):
        self.callbacks = callbacks
        self.cancelled = threading.Event()
        self.reset_called = False
        self.backend_transcripts = []
        self.interrupted = None
        self.discarded = []
        self.interrupted_thoughts = ["knowledge"]
        self.dialogue_state_manager = DialogueStateManager(num_history_turns=1)

    def set_audio_in_flight_fn(self, callback):
        self.audio_in_flight_fn = callback

    def commit_interrupted_turn(self, user_text, heard_text):
        self.interrupted = (user_text, heard_text)
        return self.dialogue_state_manager.commit_turn(
            user_text,
            f"{heard_text.strip()} [interrupted by the user]",
            self.interrupted_thoughts,
        )

    def discard_pending_turn(self, user_text):
        self.discarded.append(user_text)
        return self.dialogue_state_manager.rollback_turn(user_text)

    def run_turn(self, user_text):
        self.dialogue_state_manager.update_user_state(user_text)
        self.backend_transcripts.append(self.dialogue_state_manager.get_transcript())
        self.callbacks["on_thought"]("<sil>")
        self.callbacks["on_response_fragment"]("Let me think about that.")
        self.callbacks["on_thought"]("ConvFill pairs a Talker with a Reasoner.")
        self.callbacks["on_response_fragment"]("It pairs a local Talker with a cloud Reasoner.")
        self.dialogue_state_manager.update_response_state(
            "Let me think about that. It pairs a local Talker with a cloud Reasoner."
        )
        self.dialogue_state_manager.update_thoughts_state(["<sil>", "ConvFill pairs a Talker with a Reasoner."])
        self.callbacks["on_turn_complete"]("complete", [])
        return not self.cancelled.is_set()

    def cancel_turn(self):
        self.cancelled.set()

    def reset(self):
        self.reset_called = True


def _request(chat):
    return SimpleNamespace(
        # Keep the lightweight fake chat while satisfying the concrete type
        # required by HF's Pydantic pipeline messages.
        runtime_config=RuntimeConfig.model_construct(chat=chat),
        response=None,
        language_code="en",
        turn_id="turn-one",
        turn_revision=0,
        speech_stopped_at_s=1.25,
        response_key="response-one",
        prefetch_transaction=None,
    )


def _handler(system_factory=_FakeSystem, **setup_kwargs):
    return ConvFillRealtimeHandler(
        threading.Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"system_factory": system_factory, **setup_kwargs},
    )


def test_latest_user_text_uses_the_newest_text_turn():
    chat = _FakeChat(
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                    {"type": "input_text", "text": " newest turn "},
                ],
            },
        ]
    )
    assert latest_user_text(chat) == "newest turn"


def test_handler_streams_only_talker_fragments_and_commits_hf_history():
    chat = _FakeChat()
    handler = _handler()

    outputs = list(handler.process(_request(chat)))

    chunks = [item for item in outputs if isinstance(item, LLMResponseChunk)]
    assert [item.text for item in chunks] == [
        "Let me think about that.",
        "It pairs a local Talker with a cloud Reasoner.",
    ]
    assert all("ConvFill pairs" not in item.text for item in chunks)
    assert isinstance(outputs[-1], EndOfResponse)
    assert outputs[-1].error is None
    assert chat.trimmed is True
    assert chat.provisional[0][0] == "response-one"
    assistant = chat.provisional[0][1][0]
    assert assistant.content[0].text == ("Let me think about that. It pairs a local Talker with a cloud Reasoner.")


def test_backend_transcript_retains_the_spoken_response_across_turns():
    state = DialogueStateManager(num_history_turns=1)
    state.update_user_state("My favorite color is green.")
    state.update_response_state("I will remember that your favorite color is green.")
    state.update_thoughts_state(["The user stated a color preference."])
    state.update_user_state("What color did I say?")

    assert state.get_transcript() == (
        "User: My favorite color is green.\n"
        "You: I will remember that your favorite color is green.\n"
        "User: What color did I say?"
    )
    assert "The user stated a color preference." not in state.get_transcript()


def test_handler_reuses_convfill_history_for_the_next_hf_turn():
    chat = _FakeChat([{"role": "user", "content": "first question"}])
    handler = _handler()

    first_outputs = list(handler.process(_request(chat)))
    assert first_outputs[-1].error is None

    chat.messages.append({"role": "user", "content": "follow-up question"})
    second = _request(chat)
    second.turn_id = "turn-two"
    second.response_key = "response-two"
    second_outputs = list(handler.process(second))

    assert second_outputs[-1].error is None
    assert handler.system.backend_transcripts == [
        "User: first question",
        (
            "User: first question\n"
            "You: Let me think about that. It pairs a local Talker with a cloud Reasoner.\n"
            "User: follow-up question"
        ),
    ]


def test_session_end_resets_the_reused_convfill_system():
    handler = _handler()
    handler.on_session_end()
    assert handler.system.reset_called is True


def test_failed_turn_is_removed_from_convfill_history():
    class FailingSystem(_FakeSystem):
        def run_turn(self, user_text):
            self.dialogue_state_manager.update_user_state(user_text)
            raise RuntimeError("reasoner unavailable")

    handler = _handler(FailingSystem)
    outputs = list(handler.process(_request(_FakeChat())))

    assert isinstance(outputs[-1], EndOfResponse)
    assert outputs[-1].error == "ConvFill inference failed. Please try again."
    assert handler.system.dialogue_state_manager.user_turns == []


def test_cancelled_turn_emits_no_more_audio_and_rolls_back_history():
    class SlowCommittingSystem(_FakeSystem):
        def run_turn(self, user_text):
            self.dialogue_state_manager.update_user_state(user_text)
            self.callbacks["on_response_fragment"]("partial response")
            assert self.cancelled.wait(timeout=1.0)
            self.dialogue_state_manager.update_response_state("partial response")
            self.dialogue_state_manager.update_thoughts_state(["knowledge"])
            return True

    cancel_scope = CancelScope()
    handler = _handler(SlowCommittingSystem, cancel_scope=cancel_scope)
    output = handler.process(_request(_FakeChat()))

    first = next(output)
    assert isinstance(first, LLMResponseChunk)
    cancel_scope.cancel()
    remaining = list(output)

    assert len(remaining) == 1
    assert isinstance(remaining[0], EndOfResponse)
    assert handler.system.dialogue_state_manager.user_turns == []


def test_dialogue_state_rolls_back_pending_and_completed_turns():
    state = DialogueStateManager(num_history_turns=1)
    state.update_user_state("pending")
    assert state.rollback_turn("pending") is True
    assert state.user_turns == []

    state.update_user_state("completed")
    state.update_response_state("spoken response")
    state.update_thoughts_state(["knowledge"])
    assert state.rollback_turn("completed") is True
    assert state.user_turns == []
    assert state.responses == []
    assert state.thoughts == []


def test_backend_registration_is_idempotent(monkeypatch):
    from speech_to_speech.backend_registry import LLM_BACKENDS

    monkeypatch.delitem(LLM_BACKENDS, "convfill", raising=False)
    try:
        register_convfill_backend()
        register_convfill_backend()
    finally:
        LLM_BACKENDS.pop("convfill", None)


class _InterruptibleSystem(_FakeSystem):
    """Emits one fragment, then blocks until the turn is cancelled."""

    def run_turn(self, user_text):
        self.dialogue_state_manager.update_user_state(user_text)
        self.callbacks["on_response_fragment"]("Partial response.")
        assert self.cancelled.wait(timeout=2.0)
        return False


def _deliver(ledger, text, response_key="response-one"):
    """Simulate HF's send loop dispatching one assistant text event."""
    ledger.put(AssistantOutputEvent(parts=[AssistantTextPart(text=text)], response_key=response_key))
    ledger.get_nowait()


def test_interrupted_turn_is_committed_with_what_the_user_heard():
    cancel_scope = CancelScope()
    handler = _handler(_InterruptibleSystem, cancel_scope=cancel_scope)
    ledger = DeliveryLedgerQueue()
    handler.set_delivery_ledger(ledger)

    output = handler.process(_request(_FakeChat()))
    assert isinstance(next(output), LLMResponseChunk)

    _deliver(ledger, "Partial response.")
    _deliver(ledger, "Second half.")
    cancel_scope.cancel()
    remaining = list(output)

    assert isinstance(remaining[-1], EndOfResponse)
    assert remaining[-1].error is None
    assert handler.system.interrupted == ("What is ConvFill?", "Partial response. Second half.")
    assert handler.system.discarded == []
    assert handler.system.dialogue_state_manager.get_transcript() == (
        "User: What is ConvFill?\nYou: Partial response. Second half. [interrupted by the user]"
    )
    # The ledger is released so the next response starts from a clean slate.
    assert ledger.delivered_text("response-one") == []
    assert ledger.audio_playing() is False


def test_interrupted_turn_with_nothing_delivered_is_discarded():
    cancel_scope = CancelScope()
    handler = _handler(_InterruptibleSystem, cancel_scope=cancel_scope)
    handler.set_delivery_ledger(DeliveryLedgerQueue())

    output = handler.process(_request(_FakeChat()))
    assert isinstance(next(output), LLMResponseChunk)

    cancel_scope.cancel()
    remaining = list(output)

    assert isinstance(remaining[-1], EndOfResponse)
    assert handler.system.interrupted is None
    assert handler.system.discarded == ["What is ConvFill?"]
    assert handler.system.dialogue_state_manager.user_turns == []


def test_audio_in_flight_follows_the_delivery_ledger():
    handler = _handler()
    ledger = DeliveryLedgerQueue()
    handler.set_delivery_ledger(ledger)
    assert handler.system.audio_in_flight_fn == handler._audio_in_flight

    assert handler._audio_in_flight() == 0

    ledger.put(AudioOutput(audio=b"\x00\x00" * AUDIO_SAMPLE_RATE_HZ, response_key="response-one"))
    assert handler._audio_in_flight() == 1  # queued but not yet dispatched
    ledger.get_nowait()
    assert handler._audio_in_flight() == 1  # dispatched and still playing
    ledger.reset_playback()
    assert handler._audio_in_flight() == 0

    handler._response_key = "response-one"
    handler._yielded_fragments = 1
    assert handler._audio_in_flight() == 1  # yielded text has not reached the client
    _deliver(ledger, "Partial response.")
    assert handler._audio_in_flight() == 0


def _bare_system(dialogue_state_manager):
    system = ConvFillSystem.__new__(ConvFillSystem)
    system.dialogue_state_manager = dialogue_state_manager
    system.thought_queue = queue.Queue()
    system.tts_queue = queue.Queue()
    system._interrupted_thoughts = []
    return system


def test_commit_interrupted_turn_keeps_the_heard_prefix_and_stays_consistent():
    state = DialogueStateManager(num_history_turns=1)
    system = _bare_system(state)
    system._interrupted_thoughts = ["<sil>", "ConvFill pairs a Talker with a Reasoner."]
    state.update_user_state("q")

    assert system.commit_interrupted_turn("q", "partial") is True
    assert state.get_transcript() == "User: q\nYou: partial [interrupted by the user]"
    assert len(state.user_turns) == len(state.responses) == len(state.thoughts) == 1
    assert state.thoughts[0] == ["ConvFill pairs a Talker with a Reasoner."]
    assert system._interrupted_thoughts == []

    state.update_user_state("next")
    assert state.get_transcript().endswith("User: next")
    assert state.get_history() == {
        "user_turns_history": ["q"],
        "responses_history": ["partial [interrupted by the user]"],
    }


def test_commit_interrupted_turn_replaces_a_response_that_beat_the_cancel():
    state = DialogueStateManager(num_history_turns=1)
    system = _bare_system(state)
    state.update_user_state("q")
    state.update_response_state("full response that was never finished aloud")
    state.update_thoughts_state(["knowledge"])

    assert system.commit_interrupted_turn("q", "partial") is True
    assert state.responses == ["partial [interrupted by the user]"]
    assert state.thoughts == [["knowledge"]]
    assert len(state.user_turns) == 1


def test_discard_pending_turn_removes_the_turn():
    state = DialogueStateManager(num_history_turns=1)
    system = _bare_system(state)
    system._interrupted_thoughts = ["knowledge"]
    state.update_user_state("q")

    assert system.discard_pending_turn("q") is True
    assert state.user_turns == []
    assert system._interrupted_thoughts == []


def test_tool_history_is_proposed_before_delivery_and_turn_is_not_duplicated():
    from openai.types.realtime import RealtimeSessionCreateRequest as RealtimeSession
    from openai.types.realtime.conversation_item import RealtimeConversationItemUserMessage

    from speech_to_speech.LLM.chat import Chat

    class Remote:
        def configure_request(self, **kwargs):
            self.config = kwargs

    class ToolSystem(_FakeSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.remote = Remote()
            self.convfill_backend = SimpleNamespace(model_backend=SimpleNamespace(remote=self.remote))

        def run_turn(self, text):
            self.dialogue_state_manager.update_user_state(text)
            self.remote.config["on_tool_call"]("web_search", '{"query":"weather"}', "call_test")
            return True

    chat = Chat(6)
    chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message", role="user", content=[{"type": "input_text", "text": "Find the weather."}]
        )
    )
    request = _request(chat)
    request.runtime_config.session = RealtimeSession(
        type="realtime",
        tools=[{"type": "function", "name": "web_search", "parameters": {"type": "object"}}],
        instructions="Be concise.",
    )
    handler = _handler(ToolSystem)
    output = list(handler.process(request))
    chunk, terminal = output
    assert chunk.tools[0].name == "web_search"
    assert chunk.text == ""
    assert chunk.history.items[0].call_id == "call_test"
    assert terminal.error is None and terminal.history.complete
    assert handler.system.remote.config["instructions"] == "Be concise."
    assert handler.system.dialogue_state_manager.user_turns == []


def test_latest_text_survives_camera_only_user_message():
    assert (
        latest_user_text(
            _FakeChat(
                [
                    {"role": "user", "content": "Describe my camera."},
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,abc"}]},
                ]
            )
        )
        == "Describe my camera."
    )


def test_tool_round_trip_seeds_only_the_current_spoken_prefix():
    prepared = []

    class System(_FakeSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.convfill_backend = SimpleNamespace(
                model_backend=SimpleNamespace(remote=SimpleNamespace(configure_request=lambda **kwargs: None))
            )
            self.convfill_frontend = SimpleNamespace(prepare_continuation=lambda *args: prepared.append(args))

    chat = _FakeChat(
        [
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Do not seed this old answer."},
            {"role": "tool", "tool_call_id": "old", "content": "old result"},
            {"role": "user", "content": "Current question"},
            {"role": "assistant", "content": "Let me check the current result."},
            {"role": "tool", "tool_call_id": "current", "content": "verified result"},
        ]
    )
    list(_handler(System).process(_request(chat)))
    assert prepared == [("Current question", "Let me check the current result.")]


def test_echo_only_tool_followup_keeps_prior_response_without_an_error():
    class System(_FakeSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prefix = ""
            self.convfill_backend = SimpleNamespace(
                model_backend=SimpleNamespace(remote=SimpleNamespace(configure_request=lambda **kwargs: None))
            )
            self.convfill_frontend = SimpleNamespace(prepare_continuation=self.prepare)

        def prepare(self, user, prefix):
            self.prefix = prefix

        def run_turn(self, user):
            self.dialogue_state_manager.update_user_state(user)
            self.dialogue_state_manager.commit_turn(user, self.prefix, [])
            return True

    handler = _handler(System)
    chat = _FakeChat(
        [
            {"role": "user", "content": "Current question"},
            {"role": "assistant", "content": "Here is the answer."},
            {"role": "tool", "tool_call_id": "current", "content": "verified result"},
        ]
    )
    output = list(handler.process(_request(chat)))
    assert len(output) == 1 and output[0].error is None
    assert handler.system.dialogue_state_manager.responses == ["Here is the answer."]


class _GreetingRemote:
    def __init__(self):
        self.calls = []
        self.cancelled = False
        self.error = False
        self.wait_for_cancel = False

    def configure_request(self, **kwargs):
        pass

    def begin_turn(self, event):
        self.event = event

    def greeting(self, prompt, instructions=""):
        self.calls.append((prompt, instructions))
        if self.error:
            raise RuntimeError("Greeting endpoint failed")
        yield "Hello!"
        if self.wait_for_cancel:
            assert self.event.wait(1)
            return
        yield "What would you like to talk about?"

    def cancel(self):
        self.cancelled = True
        self.event.set()


class _GreetingSystem(_FakeSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remote = _GreetingRemote()
        self.normal_calls = []
        self.convfill_backend = SimpleNamespace(model_backend=SimpleNamespace(remote=self.remote))

    def run_turn(self, text):
        self.normal_calls.append(text)
        return super().run_turn(text)

    def cancel_turn(self):
        super().cancel_turn()
        self.remote.cancel()


def _greeting_request(chat):
    from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

    request = _request(chat)
    request.response = RealtimeResponseCreateParams(metadata={"speech_to_speech.startup_greeting": "true"})
    return request


def test_marked_startup_greeting_uses_only_reasoner_and_preserves_followup_history():
    handler = _handler(_GreetingSystem)
    chat = _FakeChat([{"role": "user", "content": "Greet the user."}])
    output = list(handler.process(_greeting_request(chat)))
    assert [chunk.text for chunk in output if isinstance(chunk, LLMResponseChunk)] == [
        "Hello!",
        "What would you like to talk about?",
    ]
    assert output[-1].error is None
    assert len(handler.system.remote.calls) == 1 and handler.system.normal_calls == []
    assert handler.system.remote.calls[0][0] == "Greet the user."
    assert handler.system.dialogue_state_manager.user_turns == ["[call connected]"]
    assert handler.system.dialogue_state_manager.responses == ["Hello! What would you like to talk about?"]
    assert chat.provisional
    chat.messages.extend(
        [
            {"role": "assistant", "content": "Hello! What would you like to talk about?"},
            {"role": "user", "content": "A regular follow-up."},
        ]
    )
    assert list(handler.process(_request(chat)))[-1].error is None
    assert handler.system.normal_calls == ["A regular follow-up."]
    assert "Hello!" in handler.system.backend_transcripts[-1]
    assert "Greet the user." not in handler.system.backend_transcripts[-1]
    handler.on_session_end()
    assert handler._greeting_attempted is False


def test_unmarked_or_later_turn_does_not_bypass_talker():
    handler = _handler(_GreetingSystem)
    chat = _FakeChat([{"role": "user", "content": "Greet the user."}])
    assert list(handler.process(_request(chat)))[-1].error is None
    chat.messages.append({"role": "assistant", "content": "A previous answer."})
    chat.messages.append({"role": "user", "content": "Greet the user."})
    assert list(handler.process(_greeting_request(chat)))[-1].error is None
    assert handler.system.remote.calls == [] and len(handler.system.normal_calls) == 2


def test_direct_greeting_failure_propagates_without_talker_fallback():
    handler = _handler(_GreetingSystem)
    handler.system.remote.error = True
    output = list(handler.process(_greeting_request(_FakeChat())))
    assert output[-1].error == "ConvFill inference failed. Please try again."
    assert handler.system.normal_calls == [] and handler.system.dialogue_state_manager.user_turns == []


def test_direct_greeting_cancellation_closes_stream_and_discards_stale_text():
    scope = CancelScope()
    handler = _handler(_GreetingSystem, cancel_scope=scope)
    handler.system.remote.wait_for_cancel = True
    output = handler.process(_greeting_request(_FakeChat()))
    assert next(output).text == "Hello!"
    scope.cancel()
    remaining = list(output)
    assert len(remaining) == 1 and isinstance(remaining[0], EndOfResponse)
    assert handler.system.remote.cancelled and handler.system.normal_calls == []
    assert handler.system.dialogue_state_manager.user_turns == []


def test_cancelled_greeting_keeps_dispatched_prefix_under_neutral_marker():
    scope = CancelScope()
    handler = _handler(_GreetingSystem, cancel_scope=scope)
    ledger = DeliveryLedgerQueue()
    handler.set_delivery_ledger(ledger)
    handler.system.remote.wait_for_cancel = True
    chat = _FakeChat([{"role": "user", "content": "Greet the user."}])
    output = handler.process(_greeting_request(chat))

    assert next(output).text == "Hello!"
    _deliver(ledger, "Hello!")
    scope.cancel()
    remaining = list(output)

    assert len(remaining) == 1 and isinstance(remaining[0], EndOfResponse)
    assert handler.system.remote.cancelled and handler.system.normal_calls == []
    state = handler.system.dialogue_state_manager
    assert state.user_turns == ["[call connected]"]
    assert state.responses == ["Hello! [interrupted by the user]"]
    assert "Greet the user." not in state.get_transcript()
    assert "What would you like to talk about?" not in state.get_transcript()
    assert handler.system.remote.calls[0][0] == "Greet the user."
