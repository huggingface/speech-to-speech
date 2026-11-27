"""The conversation continues while a client tool call is still running."""

import copy
import json

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.chat import (
    LATE_TOOL_RESULT,
    LATE_TOOL_RESULT_END,
    PENDING_TOOL_OUTPUT,
    RUNNING_TOOL_STATUS,
    Chat,
    ChatItemError,
    make_assistant_message,
    make_user_message,
)
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
)
from tests.llm_history import close_llm_runners as close_llm_runners
from tests.llm_history import drive_llm
from tests.test_chat import _make_stub_compactor, _wait_thread
from tests.test_chat_completions_backend import _chunk, _FakeStream, _make_handler, _tc_delta
from tests.test_voice_prompt import _local_handler_with_chunks

REPORT_ARGS = '{"topic": "batteries"}'
REPORT_CALL = f"research_report({REPORT_ARGS})"


def _call(call_id="call_report", name="research_report", arguments=REPORT_ARGS):
    return RealtimeConversationItemFunctionCall(
        type="function_call", id=f"fc_{call_id}", call_id=call_id, name=name, arguments=arguments
    )


def _output(call_id="call_report", output="the report"):
    return RealtimeConversationItemFunctionCallOutput(
        type="function_call_output", id=f"fco_{call_id}", call_id=call_id, output=output
    )


def _late(output="the report", name="research_report", arguments=REPORT_ARGS):
    return LATE_TOOL_RESULT.format(name=name, arguments=arguments, output=output)


def _status(*calls):
    return RUNNING_TOOL_STATUS.format(calls=", ".join(calls))


def _shape(chat):
    """Compact view of what a Chat Completions or local model is shown.

    Spoken text and the call it led into share one assistant message; the text is
    listed first so the order of speech and calls stays visible.
    """
    shape = []
    for message in chat.to_transformers_chat():
        if message["role"] == "assistant" and message.get("tool_calls"):
            if message["content"]:
                shape.append(("assistant", message["content"]))
            shape.append(("call", message["tool_calls"][0]["id"]))
        elif message["role"] == "tool":
            shape.append(("tool", message["tool_call_id"], message["content"]))
        else:
            shape.append((message["role"], message["content"]))
    return shape


def _assert_every_call_has_an_output(chat):
    """Providers and chat templates reject a call that is not directly followed by its output."""
    messages = chat.to_transformers_chat()
    for index, message in enumerate(messages):
        for tool_call in message.get("tool_calls") or []:
            assert messages[index + 1]["role"] == "tool"
            assert messages[index + 1]["tool_call_id"] == tool_call["id"]
        if message["role"] == "tool":
            assert message["tool_call_id"] in {call["id"] for call in messages[index - 1]["tool_calls"]}
    items = chat.to_responses_api_chat()
    for index, item in enumerate(items):
        if item["type"] == "function_call":
            assert items[index + 1]["type"] == "function_call_output"
            assert items[index + 1]["call_id"] == item["call_id"]
        if item["type"] == "function_call_output":
            assert items[index - 1]["type"] == "function_call"


def _chat_with_running_call(*later):
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_ordered_function_call(_call())
    for item in later:
        chat.add_item(item)
    return chat


def test_running_call_stays_visible_with_a_pending_output():
    chat = _chat_with_running_call(make_user_message("what is the capital of Australia?"))

    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", PENDING_TOOL_OUTPUT),
        ("user", f"what is the capital of Australia?\n{_status(REPORT_CALL)}"),
    ]
    _assert_every_call_has_an_output(chat)
    # The stand-in and the status line exist only in what the model is shown.
    assert [item.type for item in chat.buffer] == ["message", "function_call", "message"]
    assert chat.has_pending_tool_calls()


def test_result_before_any_later_reply_sits_next_to_its_call():
    chat = _chat_with_running_call(make_user_message("still there?"), _output())

    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
        ("user", "still there?"),
    ]
    _assert_every_call_has_an_output(chat)


def test_result_after_a_later_reply_is_shown_where_it_arrived():
    chat = _chat_with_running_call(
        make_user_message("what is the capital of Australia?"),
        make_assistant_message("Canberra."),
    )
    while_running = _shape(chat)
    chat.add_item(_output())

    # The reply was given while the tool was running. Apart from the status line,
    # the request the model saw then is a prefix of the one that delivers the result.
    question = "what is the capital of Australia?"
    assert while_running[3] == ("user", f"{question}\n{_status(REPORT_CALL)}")
    assert _shape(chat) == [*while_running[:3], ("user", question), while_running[4], ("user", _late())]
    assert while_running[2] == ("tool", "call_report", PENDING_TOOL_OUTPUT)
    _assert_every_call_has_an_output(chat)
    responses = chat.to_responses_api_chat()
    assert [item["type"] for item in responses] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "message",
        "message",
    ]
    assert responses[2]["output"] == PENDING_TOOL_OUTPUT
    assert responses[-1]["role"] == "user" and responses[-1]["content"][0]["text"] == _late()
    # Stored history keeps the real output, unchanged, where it arrived.
    assert [item.type for item in chat.buffer] == [
        "message",
        "function_call",
        "message",
        "message",
        "function_call_output",
    ]
    assert chat.buffer[-1].output == "the report"
    assert not chat.has_pending_tool_calls()


def test_late_result_and_the_next_user_message_form_one_user_turn():
    chat = _chat_with_running_call(
        make_user_message("what is the capital of Australia?"),
        make_assistant_message("Canberra."),
        _output(),
        make_user_message("anything new?"),
    )

    # The two are merged for chat templates, so the result is closed before the user's words.
    merged = f"{_late()}\n{LATE_TOOL_RESULT_END}\nanything new?"
    assert _shape(chat)[-1] == ("user", merged)
    assert chat.to_responses_api_chat()[-1]["content"][0]["text"] == merged
    _assert_every_call_has_an_output(chat)


def test_parallel_results_before_any_reply_pair_with_their_calls():
    chat = Chat(10)
    chat.add_item(make_user_message("weather in two cities"))
    chat.add_ordered_function_call(_call("call_paris", "get_weather", '{"city": "Paris"}'))
    chat.add_ordered_function_call(_call("call_rome", "get_weather", '{"city": "Rome"}'))
    chat.add_item(_output("call_rome", "sunny"))

    # One result is back and the other call is still running.
    assert _shape(chat) == [
        ("user", "weather in two cities\n" + _status('get_weather({"city": "Paris"})')),
        ("call", "call_paris"),
        ("tool", "call_paris", PENDING_TOOL_OUTPUT),
        ("call", "call_rome"),
        ("tool", "call_rome", "sunny"),
    ]
    chat.add_item(_output("call_paris", "rain"))
    assert _shape(chat) == [
        ("user", "weather in two cities"),
        ("call", "call_paris"),
        ("tool", "call_paris", "rain"),
        ("call", "call_rome"),
        ("tool", "call_rome", "sunny"),
    ]
    _assert_every_call_has_an_output(chat)


def test_result_after_a_reply_to_its_sibling_is_late():
    chat = Chat(10)
    chat.add_item(make_user_message("a report and the weather"))
    chat.add_ordered_function_call(_call())
    chat.add_ordered_function_call(_call("call_paris", "get_weather", '{"city": "Paris"}'))
    chat.add_item(_output("call_paris", "rain"))
    chat.add_item(make_assistant_message("It is raining in Paris. The report is still running."))
    chat.add_item(_output())

    assert _shape(chat) == [
        ("user", "a report and the weather"),
        ("call", "call_report"),
        ("tool", "call_report", PENDING_TOOL_OUTPUT),
        ("call", "call_paris"),
        ("tool", "call_paris", "rain"),
        ("assistant", "It is raining in Paris. The report is still running."),
        ("user", _late()),
    ]
    _assert_every_call_has_an_output(chat)


def test_text_after_a_call_in_the_same_response_is_not_a_later_reply():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items(
        "response", [make_assistant_message("On it."), _call(), make_assistant_message("One moment.")]
    )
    chat.add_item(_output())

    assert _shape(chat) == [
        ("user", "make a report"),
        ("assistant", "On it."),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
        ("assistant", "One moment."),
    ]


def test_status_line_stays_on_the_user_message_when_a_fresh_result_ends_the_request():
    chat = _chat_with_running_call(make_user_message("and the weather in Paris?"))
    chat.add_ordered_function_call(_call("call_paris", "get_weather", '{"city": "Paris"}'))
    asked = _shape(chat)[3]
    chat.add_item(_output("call_paris", "rain"))

    # The result is the last thing the model reads, and the request that issued
    # the call is a prefix of the one that delivers its result.
    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", PENDING_TOOL_OUTPUT),
        ("user", f"and the weather in Paris?\n{_status(REPORT_CALL)}"),
        ("call", "call_paris"),
        ("tool", "call_paris", "rain"),
    ]
    both_running = _status(REPORT_CALL, 'get_weather({"city": "Paris"})')
    assert asked == ("user", f"and the weather in Paris?\n{both_running}")
    assert chat.to_responses_api_chat()[-1]["type"] == "function_call_output"


def test_newest_user_message_lists_every_call_that_is_still_running():
    chat = Chat(10)
    chat.add_item(make_user_message("three reports and the weather"))
    for topic in ("batteries", "fusion", "chips"):
        chat.add_ordered_function_call(_call(f"call_{topic}", arguments=f'{{"topic": "{topic}"}}'))
    chat.add_ordered_function_call(_call("call_paris", "get_weather", '{"city": "Paris"}'))
    chat.add_item(_output("call_paris", "rain"))
    chat.add_item(make_assistant_message("It is raining in Paris."))
    chat.add_item(_output("call_fusion", "the fusion report"))

    # A late result is shown in the user role, but the status line stays with what the user said:
    # the request ends with the result. The line lists only what is left, oldest first.
    left = _status('research_report({"topic": "batteries"})', 'research_report({"topic": "chips"})')
    late = _late("the fusion report", arguments='{"topic": "fusion"}')
    assert _shape(chat)[0] == ("user", f"three reports and the weather\n{left}")
    assert _shape(chat)[-1] == ("user", late)
    assert chat.to_responses_api_chat()[-1]["content"][0]["text"] == late
    _assert_every_call_has_an_output(chat)

    # The line is rebuilt for each request and never stored.
    chat.add_item(make_assistant_message("The fusion report is in."))
    chat.add_item(make_user_message("thanks"))
    assert _shape(chat)[-3:] == [
        ("user", late),
        ("assistant", "The fusion report is in."),
        ("user", f"thanks\n{left}"),
    ]
    assert all(item.type != "message" or "Tool status" not in str(item.content) for item in chat.buffer)

    chat.add_item(_output("call_batteries", "the batteries report"))
    chat.add_item(_output("call_chips", "the chips report"))
    assert "Tool status" not in str(chat.to_transformers_chat())
    assert "Tool status" not in str(chat.to_responses_api_chat())


@pytest.mark.parametrize("copy_chat", [lambda chat: chat.copy(), lambda chat: chat.copy(deep=True)])
def test_copies_and_repeated_serialization_show_the_same_history(copy_chat):
    chat = _chat_with_running_call(
        make_user_message("what is the capital of Australia?"),
        make_assistant_message("Canberra."),
        _output(),
    )
    chat.add_ordered_function_call(_call("call_second"))

    assert chat.to_transformers_chat() == chat.to_transformers_chat() == copy_chat(chat).to_transformers_chat()
    assert chat.to_responses_api_chat() == chat.to_responses_api_chat() == copy_chat(chat).to_responses_api_chat()


def test_incomplete_tool_batch_holds_back_only_a_follow_up_the_user_has_not_spoken_after():
    chat = Chat(10)
    chat.add_item(make_user_message("weather in two cities"))
    chat.add_ordered_function_call(_call("call_paris", "get_weather", '{"city": "Paris"}'))
    chat.add_ordered_function_call(_call("call_rome", "get_weather", '{"city": "Rome"}'))
    # Nothing has come back yet: a response here is not a partial follow-up.
    assert not chat.has_incomplete_tool_batch()

    chat.add_item(_output("call_rome", "sunny"))
    assert chat.has_incomplete_tool_batch()

    chat.add_item(_output("call_paris", "rain"))
    assert not chat.has_incomplete_tool_batch()


def test_running_call_the_user_has_spoken_after_does_not_hold_back_other_follow_ups():
    chat = _chat_with_running_call(make_user_message("and the weather in Paris?"))
    chat.add_ordered_function_call(_call("call_paris", "get_weather", '{"city": "Paris"}'))
    chat.add_item(_output("call_paris", "rain"))

    assert chat.has_pending_tool_calls()
    assert not chat.has_incomplete_tool_batch()


def test_compaction_keeps_a_running_call_and_its_stand_in():
    chat = Chat(size=2)
    summarized: list = []
    chat.add_item(make_user_message("make a report"))
    chat.add_ordered_function_call(_call())
    for index in range(2):
        chat.add_item(make_user_message(f"question {index}"))
        chat.add_item(make_assistant_message(f"answer {index}"))
    chat.add_item(make_user_message("latest"))

    chat.trim_if_needed(_make_stub_compactor(captured=summarized))
    _wait_thread(chat)

    # The summarizer is shown a complete pair, and the call outlives the summary.
    types = [item["type"] for item in summarized[0]]
    assert types[types.index("function_call") + 1] == "function_call_output"
    assert "Tool status" not in str(summarized[0])
    assert chat.has_pending_tool_calls()
    assert ("tool", "call_report", PENDING_TOOL_OUTPUT) in _shape(chat)
    _assert_every_call_has_an_output(chat)

    chat.add_item(_output())
    assert ("tool", "call_report", "the report") in _shape(chat)
    _assert_every_call_has_an_output(chat)


def _tools():
    return [{"type": "function", "name": "research_report", "parameters": {"type": "object"}}]


def test_chat_completions_backend_answers_while_a_tool_runs_then_delivers_its_result():
    handler = _make_handler(stream=True)
    requests = []
    replies = iter(
        [
            [_chunk(tool_calls=[_tc_delta(0, id="srv_1", name="research_report", arguments=REPORT_ARGS)])],
            [_chunk(content="Canberra.")],
            [_chunk(content="Your report is ready.")],
        ]
    )

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return _FakeStream(next(replies))

    handler.client.chat.completions.create = create
    chat = Chat(10)
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Be brief.")
    session.tools = _tools()
    config = RuntimeConfig(chat=chat, session=session)
    service = RealtimeService()
    conn_id = service.register()

    def respond(user_text=None):
        if user_text:
            chat.add_item(make_user_message(user_text))
        request = GenerateResponseRequest(runtime_config=config)
        outputs = list(drive_llm(handler, request, service=service, conn_id=conn_id))
        assert next(output for output in outputs if isinstance(output, EndOfResponse)).error is None
        chat.finalize_provisional_generation(request.response_key)
        return outputs

    first = respond("Make a report on batteries.")
    call = next(output.tools[0] for output in first if isinstance(output, LLMResponseChunk) and output.tools)

    # The tool has not returned and the user keeps talking.
    question = "What is the capital of Australia?"
    respond(question)
    assert [message["role"] for message in requests[1]] == ["system", "user", "assistant", "tool", "user"]
    assert requests[1][4] == {"role": "user", "content": f"{question}\n{_status(REPORT_CALL)}"}
    assert requests[1][2]["tool_calls"][0]["function"] == {"name": "research_report", "arguments": REPORT_ARGS}
    assert requests[1][3] == {
        "role": "tool",
        "tool_call_id": requests[1][2]["tool_calls"][0]["id"],
        "content": PENDING_TOOL_OUTPUT,
    }
    assert chat.has_pending_tool_calls()

    # The result arrives after that reply; the follow-up ends with it.
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id=call.call_id, output="done")
    )
    respond()
    assert requests[2][:4] == requests[1][:4]
    assert requests[2][4:] == [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "Canberra."},
        {"role": "user", "content": _late("done")},
    ]
    assert [item.type for item in chat.buffer] == [
        "message",
        "function_call",
        "message",
        "message",
        "function_call_output",
        "message",
    ]


def test_local_backend_answers_while_a_tool_runs():
    handler = _local_handler_with_chunks([LLMResponseChunk(parts=[AssistantTextPart(text="Canberra.")])])
    chat = _chat_with_running_call(make_user_message("what is the capital of Australia?"))

    outputs = list(drive_llm(handler, GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat))))

    assert next(output for output in outputs if isinstance(output, EndOfResponse)).error is None
    assert [item.type for item in chat.buffer] == ["message", "function_call", "message", "message"]
    assert chat.buffer[-1].content[0].text == "Canberra."
    assert chat.has_pending_tool_calls()


def test_late_result_text_names_the_call_it_answers():
    text = _late(output=json.dumps({"status": "done"}))

    assert "research_report" in text and REPORT_ARGS in text and '{"status": "done"}' in text
    assert "not said by the user" in text


@pytest.mark.parametrize("injected", [False, True])
def test_pending_report_stays_visible_after_trimming(injected):
    chat = Chat(1)
    chat.add_item(make_user_message("Make a report."))
    if injected:
        chat.add_item(_call("call_report"))
    else:
        chat.add_ordered_function_call(_call("call_report"))
    chat.add_item(make_user_message("What is the capital of Australia?"))
    chat.add_item(make_assistant_message("Canberra."))
    if not injected:
        chat.trim_if_needed()
    assert chat.has_pending_tool_calls()

    for view in (chat.to_transformers_chat(), chat.to_responses_api_chat()):
        assert "research_report" in json.dumps(view), view


def test_result_for_a_trimmed_call_is_shown_with_its_call_when_it_arrives():
    chat = Chat(1)
    chat.add_item(make_user_message("Make a report."))
    chat.add_ordered_function_call(_call("call_report"))
    chat.add_item(make_user_message("What is the capital of Australia?"))
    chat.add_item(make_assistant_message("Canberra."))
    chat.trim_if_needed()
    running = _status(f"research_report({REPORT_ARGS})")
    assert _shape(chat) == [("user", f"What is the capital of Australia?\n{running}"), ("assistant", "Canberra.")]

    chat.add_item(_output("call_report"))

    assert _shape(chat)[-2:] == [("call", "call_report"), ("tool", "call_report", "the report")]
    assert "Tool status" not in json.dumps(chat.to_transformers_chat())
    _assert_every_call_has_an_output(chat)


def test_running_calls_outside_the_history_are_left_out_of_a_summary():
    chat = Chat(size=2)
    summarized: list = []
    chat.add_item(_call("call_report"))  # created by the client: staged, not part of the history
    for index in range(2):
        chat.add_item(make_user_message(f"question {index}"))
        chat.add_item(make_assistant_message(f"answer {index}"))
    chat.add_item(make_user_message("latest"))

    chat.trim_if_needed(_make_stub_compactor(captured=summarized))
    _wait_thread(chat)

    assert summarized and "Tool status" not in str(summarized[0])
    assert _status(f"research_report({REPORT_ARGS})") in chat.to_transformers_chat()[-1]["content"]


def test_reply_from_a_later_response_makes_the_result_late_but_text_of_the_same_response_does_not():
    same = Chat(8)
    same.add_item(make_user_message("Make a report."))
    same.add_provisional_generation_items("resp_1", [_call("call_report"), make_assistant_message("On it.")])
    same.finalize_provisional_generation("resp_1")
    same.add_item(_output("call_report"))
    assert _shape(same) == [
        ("user", "Make a report."),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
        ("assistant", "On it."),
    ]

    later = Chat(8)
    later.add_item(make_user_message("Make a report."))
    later.add_provisional_generation_items("resp_1", [_call("call_report")])
    later.finalize_provisional_generation("resp_1")
    later.add_provisional_generation_items("resp_2", [make_assistant_message("The report is still running.")])
    later.finalize_provisional_generation("resp_2")
    later.add_item(_output("call_report"))
    assert _shape(later) == [
        ("user", "Make a report."),
        ("call", "call_report"),
        ("tool", "call_report", PENDING_TOOL_OUTPUT),
        ("assistant", "The report is still running."),
        ("user", _late()),
    ]
    assert _shape(later.copy(deep=True)) == _shape(later)
    _assert_every_call_has_an_output(later)


def test_second_result_for_a_call_is_rejected():
    chat = _chat_with_running_call(_output(output="first"))
    before = _shape(chat)

    with pytest.raises(ChatItemError, match="already has an output"):
        chat.add_item(
            RealtimeConversationItemFunctionCallOutput(
                type="function_call_output", id="fco_again", call_id="call_report", output="second"
            )
        )

    assert _shape(chat) == before
    assert [item.type for item in chat.buffer] == ["message", "function_call", "function_call_output"]


def test_reply_that_is_rolled_back_does_not_make_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call(), make_assistant_message("One moment.")])
    chat.finalize_provisional_generation("first")
    chat.add_provisional_generation_items("progress", [make_assistant_message("Still running.")])
    chat.rollback_provisional_generation("progress")
    chat.add_item(_output())

    # The cancelled reply is gone from history, so nothing was said after the call.
    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
        ("assistant", "One moment."),
    ]

    # A reply that was delivered still makes a later result late.
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    chat.add_provisional_generation_items("progress", [make_assistant_message("Still running.")])
    chat.add_provisional_generation_items("cancelled", [make_assistant_message("As I said")])
    chat.rollback_provisional_generation("cancelled")
    chat.add_item(_output())
    assert _shape(chat)[-1] == ("user", _late())


def test_heard_part_of_a_cancelled_reply_still_makes_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    chat.add_provisional_generation_items("progress", [make_assistant_message("The report is still running.")])
    # The user interrupts: the reply is rolled back and the part that was heard is kept.
    chat.rollback_provisional_generation("progress")
    chat.add_heard_assistant_message("item_progress", "The report is still")
    chat.add_item(_output())

    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", PENDING_TOOL_OUTPUT),
        ("assistant", "The report is still"),
        ("user", _late()),
    ]


def test_reply_reported_as_heard_after_it_was_removed_still_makes_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    reply = chat.add_provisional_generation_items("progress", [make_assistant_message("The report is still running.")])[
        0
    ]
    chat.finalize_provisional_generation("progress")
    assert chat.truncate_assistant_message(reply.id, "")
    assert chat.truncate_assistant_message(reply.id, "The report is still")
    chat.add_item(_output())

    assert _shape(chat)[-2:] == [("assistant", "The report is still"), ("user", _late())]


def test_delivered_reply_keeps_the_result_late_when_an_overlapping_reply_is_rolled_back():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    chat.add_provisional_generation_items("cancelled", [make_assistant_message("Still")])
    chat.add_provisional_generation_items("delivered", [make_assistant_message("As I said, still running.")])
    chat.rollback_provisional_generation("cancelled")
    chat.finalize_provisional_generation("delivered")
    chat.add_item(_output())

    assert _shape(chat)[-2:] == [("assistant", "As I said, still running."), ("user", _late())]


def test_restoring_text_of_the_response_that_made_the_call_does_not_make_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    text = chat.add_provisional_generation_items("first", [_call(), make_assistant_message("I am making it.")])[1]
    chat.finalize_provisional_generation("first")
    # Playback first reports nothing heard, then the real position.
    assert chat.truncate_assistant_message(text.id, "")
    assert chat.truncate_assistant_message(text.id, "I am making")
    chat.add_item(_output())

    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
        ("assistant", "I am making"),
    ]


def test_reply_removed_as_unheard_no_longer_makes_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call(), make_assistant_message("On it.")])
    chat.finalize_provisional_generation("first")
    chat.add_provisional_generation_items("progress", [make_assistant_message("The report is still running.")])
    chat.rollback_provisional_generation("progress")
    chat.add_heard_assistant_message("item_progress", "The report is still")
    # Playback then reports that none of it was heard after all.
    assert chat.truncate_assistant_message("item_progress", "")
    chat.add_item(_output())

    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
        ("assistant", "On it."),
    ]


def test_shortened_reply_still_makes_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    reply = chat.add_provisional_generation_items("progress", [make_assistant_message("The report is still running.")])[
        0
    ]
    chat.finalize_provisional_generation("progress")
    assert chat.truncate_assistant_message(reply.id, "The report")
    chat.add_item(_output())

    assert _shape(chat)[-2:] == [("assistant", "The report"), ("user", _late())]


def test_reply_written_before_a_client_created_call_is_placed_does_not_make_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_item(_call())  # Created by the client: placed in history when its result arrives.
    chat.add_provisional_generation_items("reply", [make_assistant_message("Working on it.")])
    chat.finalize_provisional_generation("reply")
    chat.add_item(_output())

    assert _shape(chat) == [
        ("user", "make a report"),
        ("assistant", "Working on it."),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
    ]


def test_result_of_a_call_trimmed_from_history_is_not_late_after_earlier_replies():
    chat = Chat(1)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    for turn in range(3):
        chat.add_item(make_user_message(f"question {turn}"))
        chat.add_provisional_generation_items(f"reply {turn}", [make_assistant_message(f"answer {turn}")])
        chat.finalize_provisional_generation(f"reply {turn}")
        chat.trim_if_needed()
    assert not any(item.type == "function_call" for item in chat.buffer)
    chat.add_item(_output())

    assert _shape(chat)[-2:] == [("call", "call_report"), ("tool", "call_report", "the report")]


def test_result_that_arrives_during_an_interrupted_reply_is_late_after_its_heard_part():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    anchor = chat.history_anchor_id()
    chat.add_provisional_generation_items(
        "progress", [make_assistant_message("The report is still running.")], after_item_id=anchor
    )
    chat.add_item(_output())  # The tool returns while the reply is being spoken.
    chat.rollback_provisional_generation("progress")
    chat.add_heard_assistant_message("item_progress", "The report is still", after_item_id=anchor)

    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", PENDING_TOOL_OUTPUT),
        ("assistant", "The report is still"),
        ("user", _late()),
    ]


def test_deleted_reply_leaves_no_record_for_an_item_that_reuses_its_id():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call(), make_assistant_message("On it.")])
    chat.finalize_provisional_generation("first")
    progress = make_assistant_message("The report is still running.")
    progress.id = "msg_progress"
    chat.add_provisional_generation_items("progress", [progress])
    chat.finalize_provisional_generation("progress")
    chat.delete_item("msg_progress")
    assert chat._replies_past == {}
    recreated = make_assistant_message("Recreated by the client.")
    recreated.id = "msg_progress"
    chat.add_item(recreated)
    chat.add_item(_output())

    assert ("tool", "call_report", "the report") in _shape(chat)


def test_rolled_back_reply_leaves_no_record():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    chat.add_provisional_generation_items("progress", [make_assistant_message("The report is still running.")])
    assert chat._replies_past
    chat.rollback_provisional_generation("progress")

    assert chat._replies_past == {}


def test_text_written_after_its_own_call_in_a_second_batch_does_not_make_the_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.add_provisional_generation_items("first", [make_assistant_message("I am making it.")])
    chat.finalize_provisional_generation("first")
    chat.add_item(_output())

    assert _shape(chat) == [
        ("user", "make a report"),
        ("call", "call_report"),
        ("tool", "call_report", "the report"),
        ("assistant", "I am making it."),
    ]


def test_call_from_a_later_response_makes_the_earlier_result_late():
    chat = Chat(10)
    chat.add_item(make_user_message("make a report"))
    chat.add_provisional_generation_items("first", [_call()])
    chat.finalize_provisional_generation("first")
    chat.add_provisional_generation_items("second", [_call("call_weather", "get_weather", "{}")])
    chat.finalize_provisional_generation("second")
    chat.add_item(_output())

    role, content = _shape(chat)[-1]
    assert role == "user" and content.startswith(_late())


def test_image_sent_after_a_result_does_not_release_the_rest_of_the_batch():
    chat = Chat(10)
    chat.add_item(make_user_message("weather in two cities"))
    chat.add_ordered_function_call(_call("call_paris", "get_weather", '{"city": "Paris"}'))
    chat.add_ordered_function_call(_call("call_rome", "get_weather", '{"city": "Rome"}'))
    chat.add_item(_output("call_rome", "sunny"))
    assert chat.has_incomplete_tool_batch()

    chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
        )
    )
    assert chat.has_incomplete_tool_batch()

    chat.add_item(make_user_message("never mind Paris"))
    assert not chat.has_incomplete_tool_batch()


def test_image_sent_after_a_late_result_does_not_take_the_status_line():
    chat = Chat(10)
    chat.add_item(make_user_message("two reports"))
    chat.add_ordered_function_call(_call())
    chat.add_ordered_function_call(_call("call_chips", arguments='{"topic": "chips"}'))
    chat.add_item(make_user_message("how long will they take?"))
    chat.add_item(make_assistant_message("A few minutes."))
    chat.add_item(_output())
    chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
        )
    )

    status = _status('research_report({"topic": "chips"})')
    texts = [row[1] for row in _shape(chat) if row[0] == "user"]
    assert texts[1] == f"how long will they take?\n{status}"
    assert "Tool status" not in str(texts[-1]) and LATE_TOOL_RESULT_END not in str(texts[-1])
