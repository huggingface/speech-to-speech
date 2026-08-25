import json
from threading import Event
from types import MethodType, SimpleNamespace

from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex, build_tool_system_prompt
from speech_to_speech.LLM.voice_prompt import VOICE_SYSTEM_PROMPT, build_voice_system_prompt
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    GenerateResponseRequest,
    LLMResponseChunk,
)


def _tool(name="dance"):
    return FunctionTool(
        type="function",
        name=name,
        description=f"Use {name}.",
        parameters={"type": "object", "properties": {}},
    )


def _stream_context(*tool_names, sentence_batch=None):
    return StreamContext(
        function_tools=[_tool(name) for name in tool_names],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
        sentence_batch=sentence_batch or [],
    )


def _tool_call(name):
    return ResponseFunctionToolCall(
        type="function_call",
        id=f"fc_{name}",
        call_id=f"call_{name}",
        name=name,
        arguments="{}",
    )


def test_voice_prompt_is_short_and_keeps_persona_in_session_prompt():
    prompt = build_voice_system_prompt("Be concise.")

    assert len(VOICE_SYSTEM_PROMPT.split()) < 230
    assert len(prompt.split()) < 240
    assert "The session prompt defines persona" in prompt
    assert "Match the user's intent" not in prompt


def test_voice_prompt_makes_speech_the_default_and_handles_noisy_stt():
    prompt = build_voice_system_prompt("Be concise.")

    assert "Speech is the default." in prompt
    assert "Use tools when they help" in prompt
    assert "Use at most one tool" not in prompt
    assert "Treat transcripts as noisy." in prompt
    assert "Correct likely mishearings only if asked or meaning depends on it" in prompt
    assert "Reachy/Richie/Richy" not in prompt
    assert "If unsure whether a tool is needed, just speak." in prompt


def test_voice_prompt_requests_one_lead_in_and_silent_follow_up_tools():
    prompt = build_voice_system_prompt("Be concise.")

    assert "act immediately rather than merely offering" in prompt
    assert "one brief acknowledgement before the first call" in prompt
    assert "After tool results" in prompt
    assert "make further calls without speaking" in prompt
    assert "give one final answer; do not narrate individual calls" in prompt
    assert prompt.count("Never mention tools or their function names in spoken output.") == 1
    assert "For expression/background tools, speak first." in prompt
    assert "Sure, here's my best <emotion>." in prompt
    assert "Sure, here's my best sadness." not in prompt
    assert "do not add a second spoken comment" in prompt
    assert "Use motion, dance, emotion, and similar tools sparingly" in prompt


def test_local_tool_prompt_preserves_multiple_tool_call_order():
    prompt = build_tool_system_prompt(
        [
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    assert "each named-argument function call inside its own" in prompt
    assert "preserve the intended text/tool order" in prompt
    assert "Only one tool call may appear in a response." not in prompt


def test_local_tool_prompt_leaves_voice_choreography_to_voice_prompt():
    prompt = build_tool_system_prompt(
        [
            FunctionTool(
                type="function",
                name="camera",
                description="Look through the camera.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    assert "brief acknowledgement" not in prompt
    assert "speak first" not in prompt
    assert "<emotion>" not in prompt
    assert "do not claim tool results before a tool result is available" in prompt
    assert "Omit optional args instead of placeholder values" in prompt


def test_local_tool_parser_flushes_lead_in_before_tool_even_with_large_sentence_batch():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"Here we go. {ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Here we go.", ""]
    assert chunks[0].tools == []
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def test_local_tool_parser_flushes_pending_batch_before_tool_with_empty_before_text():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
        sentence_batch=["Queued lead-in."],
    )
    text = f"{ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Queued lead-in.", ""]
    assert chunks[0].tools == []
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def test_local_tool_parser_preserves_repeated_tool_blocks(monkeypatch):
    monkeypatch.setattr(
        "speech_to_speech.LLM.language_model.sent_tokenize",
        lambda value: [value.strip()] if value.strip() else [],
    )
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"Watch this. {ENTER_CODE}dance(){END_CODE} Watch this. {ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Watch this.", "", "Watch this.", ""]
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in chunks[3].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance", "dance"]
    assert remaining == ""


def test_local_tool_parser_preserves_interleaved_text_and_tool_calls(monkeypatch):
    monkeypatch.setattr(
        "speech_to_speech.LLM.language_model.sent_tokenize",
        lambda value: [value.strip()] if value.strip() else [],
    )
    handler = object.__new__(LanguageModelHandler)
    ctx = _stream_context("dance", "camera")
    text = f"First. {ENTER_CODE}dance(){END_CODE} Middle. {ENTER_CODE}camera(){END_CODE} Last."

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [(chunk.text, [tool.name for tool in chunk.tools]) for chunk in chunks] == [
        ("First.", []),
        ("", ["dance"]),
        ("Middle.", []),
        ("", ["camera"]),
    ]
    assert [tool.name for tool in tools] == ["dance", "camera"]
    assert remaining.strip() == "Last."


def test_local_markdown_cleanup_does_not_modify_tool_block(monkeypatch):
    monkeypatch.setattr(
        "speech_to_speech.LLM.language_model.sent_tokenize",
        lambda value: [value.strip()] if value.strip() else [],
    )
    handler = object.__new__(LanguageModelHandler)
    argument = "**bold** _italic_ x*y #topic"
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="search_docs",
                description="Search for text.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"**Checking.** {ENTER_CODE}search_docs(query={argument!r}){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Checking.", ""]
    assert [tool.name for tool in chunks[1].tools] == ["search_docs"]
    assert json.loads(tools[0].arguments) == {"query": argument}
    assert remaining == ""


def test_local_text_only_with_tools_preserves_markdown_verbatim():
    handler = object.__new__(LanguageModelHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler.stream_batch_sentences = 3
    ctx = _stream_context("lookup")
    raw = "  **First.**\n\n_Second._  "

    chunks = list(
        handler._stream_tokens(
            iter([raw]),
            None,
            None,
            ctx,
            response=RealtimeResponseCreateParams(output_modalities=["text"]),
        )
    )

    assert "".join(chunk.text for chunk in chunks) == raw
    assert ctx.printable_text == ""


def test_local_audio_strips_markdown_after_reassembling_streamed_deltas():
    handler = object.__new__(LanguageModelHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler.stream_batch_sentences = 1
    ctx = StreamContext()

    chunks = list(
        handler._stream_tokens(
            iter(["This is *ita", "lic* text. ", "Second sentence."]),
            None,
            None,
            ctx,
        )
    )

    assert [chunk.text for chunk in chunks] == ["This is italic text."]
    assert ctx.printable_text.strip() == "Second sentence."


def test_local_audio_streaming_preserves_whitespace_across_chunks():
    handler = object.__new__(LanguageModelHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.stop_event = Event()
    handler.stream_batch_sentences = 1
    ctx = StreamContext()

    chunks = list(
        handler._stream_tokens(
            iter(["It is sunny. What is ", "the weather like?"]),
            None,
            None,
            ctx,
        )
    )

    text = "".join(chunk.text for chunk in chunks) + ctx.printable_text
    assert [chunk.text for chunk in chunks] == ["It is sunny."]
    assert "isthe" not in text
    assert ctx.printable_text == "What is the weather like?"


def test_local_text_only_tool_marker_can_span_tokens_without_losing_whitespace():
    handler = object.__new__(LanguageModelHandler)
    ctx = _stream_context("dance")
    response = RealtimeResponseCreateParams(output_modalities=["text"])

    first, tools, remaining = handler._process_printable_text("  before <co", None, [], ctx, response=response)
    second, tools, remaining = handler._process_printable_text(
        f"{remaining}de>dance(){END_CODE}\n after  ",
        None,
        tools,
        ctx,
        response=response,
    )

    chunks = [*first, *second]
    assert [(chunk.text, [tool.name for tool in chunk.tools]) for chunk in chunks] == [
        ("  before ", []),
        ("", ["dance"]),
        ("\n after  ", []),
    ]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def _local_handler_with_chunks(chunks, *, cancelled=False):
    handler = object.__new__(LanguageModelHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.user_role = "user"
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])

    def generate(_self, _chat, _language_code, _generation, ctx, _runtime_config, _response):
        ctx.raw_generated_text = "raw model output with <code>tool blocks</code>"
        ctx.generated_text = ctx.raw_generated_text
        yield from chunks
        ctx.cancelled = cancelled

    handler._generate = MethodType(generate, handler)
    return handler


def test_local_history_commits_text_and_tools_in_emitted_order():
    first_tool = _tool_call("first")
    second_tool = _tool_call("second")
    chunks = [
        LLMResponseChunk(
            parts=[
                AssistantTextPart(text="before"),
                AssistantToolCallPart(tool=first_tool),
                AssistantTextPart(text="between"),
                AssistantToolCallPart(tool=second_tool),
                AssistantTextPart(text="after"),
            ]
        )
    ]
    handler = _local_handler_with_chunks(chunks)
    chat = Chat(5)
    chat.add_item(make_user_message("go"))

    list(handler.process(GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat))))

    output = chat.buffer[1:]
    assert [item.type for item in output] == ["message", "function_call", "message", "function_call", "message"]
    assert [item.content[0].text for item in output if isinstance(item, RealtimeConversationItemAssistantMessage)] == [
        "before",
        "between",
        "after",
    ]
    assert [item.name for item in output if isinstance(item, RealtimeConversationItemFunctionCall)] == [
        "first",
        "second",
    ]
    assert "<code>" not in str(output)

    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id="call_first",
            output="first result",
        )
    )

    assert [item["type"] for item in chat.to_responses_api_chat()[1:]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert [item["role"] for item in chat.to_transformers_chat()[1:]] == [
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]

    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id="call_second",
            output="second result",
        )
    )

    assert [item.type for item in chat.buffer[1:]] == [
        "message",
        "function_call",
        "message",
        "function_call",
        "message",
        "function_call_output",
        "function_call_output",
    ]
    assert [item["type"] for item in chat.to_responses_api_chat()[1:]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]


def test_cancelled_local_generation_does_not_write_partial_history():
    handler = _local_handler_with_chunks([LLMResponseChunk(text="partial")], cancelled=True)
    chat = Chat(5)
    user = chat.add_item(make_user_message("go"))

    list(handler.process(GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat))))

    assert chat.buffer == [user]


def test_local_tool_call_is_recorded_before_chunk_is_emitted():
    tool = _tool_call("first")
    handler = _local_handler_with_chunks(
        [
            LLMResponseChunk(
                parts=[
                    AssistantTextPart(text="before"),
                    AssistantToolCallPart(tool=tool),
                ]
            ),
            LLMResponseChunk(text="after"),
        ]
    )
    chat = Chat(5)
    chat.add_item(make_user_message("go"))
    request = GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat))
    generation = handler.process(request)

    first = next(generation)

    assert first.tools == [tool]
    assert chat.has_pending_tool_calls()
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id="call_first",
            output="result",
        )
    )
    list(generation)

    assert [item.type for item in chat.buffer] == [
        "message",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert request.response_key in chat._provisional_generations
    chat.finalize_provisional_generation(request.response_key)
    assert chat._provisional_generations == {}


def test_cancelled_local_tool_turn_rolls_back_fast_output():
    tool = _tool_call("first")
    handler = _local_handler_with_chunks(
        [LLMResponseChunk(parts=[AssistantToolCallPart(tool=tool)])],
        cancelled=True,
    )
    chat = Chat(5)
    user = chat.add_item(make_user_message("go"))
    generation = handler.process(GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat)))

    next(generation)
    assert chat._provisional_generations
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id="call_first",
            output="result",
        )
    )
    list(generation)

    assert chat.buffer == [user]
    assert not chat.has_pending_tool_calls()
    assert chat._provisional_generations == {}
