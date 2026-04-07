"""Tests for incremental Qwen XML tool-call buffering (OpenAPI stream path)."""

import json

from LLM.tool_call.function_tool import FunctionTool
from LLM.tool_call.qwen3coder_tool_parser import (
    Qwen3CoderToolParser,
    process_printable_text_qwen_xml,
    strip_qwen_tool_markup_for_chat,
)


def _fn_tool() -> FunctionTool:
    return FunctionTool(
        type="function",
        name="fn",
        parameters={
            "type": "object",
            "properties": {"p": {"type": "string"}},
        },
    )


def _tool_block(param: str) -> str:
    return (
        "<tool_call><function=fn>"
        f"<parameter=p>{param}</parameter>"
        "</function></tool_call>"
    )


def _feed(
    deltas: list[str],
    parser: Qwen3CoderToolParser,
    tools: list | None = None,
):
    buf = ""
    tools = tools if tools is not None else []
    all_chunks: list = []
    for d in deltas:
        buf += d
        chunks, tools, buf = process_printable_text_qwen_xml(buf, tools, parser)
        all_chunks.extend(chunks)
    return all_chunks, tools, buf


class TestProcessPrintableTextQwenXml:
    def test_speech_only_streams_complete_sentences(self):
        parser = Qwen3CoderToolParser(tools=[_fn_tool()])
        chunks, tools, buf = _feed(["Hello. ", "World."], parser)
        assert tools == []
        assert any("Hello" in c for c in chunks)
        assert "World" in buf or any("World" in c for c in chunks)

    def test_eager_flush_prefix_when_tool_tag_arrives(self):
        parser = Qwen3CoderToolParser(tools=[_fn_tool()])
        chunks, tools, buf = _feed(["Hello.", " ", "<tool_call>"], parser)
        assert any("Hello" in c for c in chunks)
        assert "<tool_call>" in buf
        assert "</tool_call>" not in buf
        assert tools == []

    def test_parse_only_on_outer_close_two_blocks_with_speech(self):
        parser = Qwen3CoderToolParser(tools=[_fn_tool()])
        text = (
            "First. "
            + _tool_block("a")
            + " Second. "
            + _tool_block("b")
            + " Third."
        )
        chunks, tools, buf = _feed([text], parser)
        assert len(tools) == 2
        assert json.loads(tools[0]["arguments"])["p"] == "a"
        assert json.loads(tools[1]["arguments"])["p"] == "b"
        joined = " ".join(chunks) + " " + buf
        assert "tool_call" not in joined
        assert "function=" not in joined

    def test_open_tool_call_no_parse_strip_tail(self):
        parser = Qwen3CoderToolParser(tools=[_fn_tool()])
        chunks, tools, buf = _feed(["Speech. ", "<tool_call><function=fn><parameter=p>x"], parser)
        assert tools == []
        assert "tool_call" in buf or "<tool_call>" in buf
        tail = strip_qwen_tool_markup_for_chat(buf)
        assert "tool_call" not in tail
        assert "Speech" in tail or "Speech" in "".join(chunks)

    def test_orphan_close_tag_dropped(self):
        parser = Qwen3CoderToolParser(tools=[_fn_tool()])
        text = "</tool_call>Hello. " + _tool_block("z") + " Done."
        _, tools, buf = _feed([text], parser)
        assert len(tools) == 1
        assert "Hello" in buf or "Done" in buf

    def test_orphan_close_tag_after_block_dropped(self):
        """Stray ``</tool_call>`` after a removed complete block must not reach TTS."""
        parser = Qwen3CoderToolParser(tools=[_fn_tool()])
        text = _tool_block("a") + "</tool_call> Hello."
        chunks, tools, buf = _feed([text], parser)
        assert len(tools) == 1
        joined = " ".join(chunks) + " " + buf
        assert "</tool_call>" not in joined


class TestStripHelpers:
    def test_strip_qwen_tool_markup_for_chat(self):
        raw = "Hi. " + _tool_block("x") + " Bye."
        assert "<tool_call>" not in strip_qwen_tool_markup_for_chat(raw)
        assert "Hi" in strip_qwen_tool_markup_for_chat(raw)
        assert "Bye" in strip_qwen_tool_markup_for_chat(raw)

    def test_strip_qwen_joins_speech_between_and_after_blocks(self):
        raw = (
            "First. "
            + _tool_block("a")
            + "\n\nSecond. "
            + _tool_block("b")
            + " Third."
        )
        out = strip_qwen_tool_markup_for_chat(raw)
        assert out == "First. Second. Third."

    def test_strip_qwen_drops_unclosed_tool_call_tail(self):
        s = "Hello <tool_call>partial"
        assert strip_qwen_tool_markup_for_chat(s) == "Hello"
