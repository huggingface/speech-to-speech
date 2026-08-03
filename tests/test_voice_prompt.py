from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex, build_tool_system_prompt
from speech_to_speech.LLM.voice_prompt import VOICE_SYSTEM_PROMPT, build_voice_system_prompt


def test_voice_prompt_is_short_and_keeps_persona_in_session_prompt():
    prompt = build_voice_system_prompt("Be concise.")

    assert len(VOICE_SYSTEM_PROMPT.split()) < 230
    assert len(prompt.split()) < 240
    assert "The session prompt defines persona" in prompt
    assert "Match the user's intent" not in prompt


def test_voice_prompt_makes_speech_the_default_and_handles_noisy_stt():
    prompt = build_voice_system_prompt("Be concise.")

    assert "Speech is the default." in prompt
    assert "Use at most one tool" in prompt
    assert "Treat transcripts as noisy." in prompt
    assert "Correct likely mishearings only if asked or meaning depends on it" in prompt
    assert "Reachy/Richie/Richy" not in prompt
    assert "If unsure whether a tool is needed, just speak." in prompt


def test_voice_prompt_requests_spoken_lead_in_and_sparing_expression_tools():
    prompt = build_voice_system_prompt("Be concise.")

    assert "Before a tool call, use a brief natural utterance" in prompt
    assert "briefly say that you will check" in prompt
    assert "For expression/background tools, speak first." in prompt
    assert "Sure, here's my best <emotion>." in prompt
    assert "Sure, here's my best sadness." not in prompt
    assert "Never mention tools." in prompt
    assert "do not add a second spoken comment" in prompt
    assert "Use motion, dance, emotion, and similar tools sparingly" in prompt


def test_local_tool_prompt_forbids_multiple_tool_calls():
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

    assert "Only one tool call may appear in a response." in prompt
    assert "Multiple tool calls can live" not in prompt


def test_local_tool_prompt_allows_spoken_lead_in_before_code_block():
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

    assert "one brief natural sentence before the tool call" in prompt
    assert "always speak first" in prompt
    assert "Sure, here's my best <emotion>." in prompt
    assert "Sure, here's my best sadness." not in prompt
    assert "fitting empathetic sentence" in prompt
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


def test_local_tool_parser_skips_duplicate_tool_blocks_and_preserves_trailing_text():
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

    assert [chunk.text for chunk in chunks] == ["Watch this.", ""]
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining.strip() == "Watch this."
