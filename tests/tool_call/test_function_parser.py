import json

import pytest

from speech_to_speech.LLM.tool_call import function_call as function_call_module
from speech_to_speech.LLM.tool_call.function_call import (
    FunctionToolCall,
    extract_function_calls_from_text,
    parse_function_call,
)
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool

# ---------------------------------------------------------------------------
# parse_function_call – single calls
# ---------------------------------------------------------------------------


class TestParseFunctionCall:
    @pytest.mark.parametrize(
        "call_str, expected_name, expected_params",
        [
            ("mobile.home()", "mobile.home", {}),
            ("mobile.back()", "mobile.back", {}),
            ("mobile.open_app(app_name='drupe')", "mobile.open_app", {"app_name": "drupe"}),
            ("mobile.long_press(x=0.799, y=0.911)", "mobile.long_press", {"x": 0.799, "y": 0.911}),
            ("mobile.terminate(status='success')", "mobile.terminate", {"status": "success"}),
            ("answer('text')", "answer", {"__arg_0__": "text"}),
            ("pyautogui.hscroll(page=-0.1)", "pyautogui.hscroll", {"page": -0.1}),
            ("pyautogui.scroll(page=-0.1)", "pyautogui.scroll", {"page": -0.1}),
            ("pyautogui.scroll(0.13)", "pyautogui.scroll", {"__arg_0__": 0.13}),
            ("pyautogui.click(x=0.8102, y=0.9463)", "pyautogui.click", {"x": 0.8102, "y": 0.9463}),
            ("pyautogui.hotkey(keys=['ctrl', 'c'])", "pyautogui.hotkey", {"keys": ["ctrl", "c"]}),
            ("pyautogui.press(keys='enter')", "pyautogui.press", {"keys": "enter"}),
            ("pyautogui.press(keys=['enter'])", "pyautogui.press", {"keys": ["enter"]}),
            ("pyautogui.moveTo(x=0.04, y=0.405)", "pyautogui.moveTo", {"x": 0.04, "y": 0.405}),
            ("pyautogui.write(message='bread buns')", "pyautogui.write", {"message": "bread buns"}),
            ("pyautogui.dragTo(x=0.8102, y=0.9463)", "pyautogui.dragTo", {"x": 0.8102, "y": 0.9463}),
        ],
    )
    def test_single_call(self, call_str, expected_name, expected_params):
        results = parse_function_call(call_str)
        assert len(results) == 1
        assert results[0].function_name == expected_name
        assert results[0].parameters == expected_params

    def test_swipe_with_list_params(self):
        results = parse_function_call("mobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])")
        assert len(results) == 1
        assert results[0].function_name == "mobile.swipe"
        assert results[0].parameters["from_coord"] == [0.581, 0.898]
        assert results[0].parameters["to_coord"] == [0.601, 0.518]


# ---------------------------------------------------------------------------
# parse_function_call – multiple positional arguments
# ---------------------------------------------------------------------------


class TestPositionalArguments:
    def test_bare_identifiers(self):
        results = parse_function_call("function(arg1, arg2, arg3)")
        assert len(results) == 1
        assert results[0].function_name == "function"

    def test_mixed_positional_and_named(self):
        results = parse_function_call("function('hello', 123, x=0.5)")
        r = results[0]
        assert r.parameters["__arg_0__"] == "hello"
        assert r.parameters["__arg_1__"] == 123
        assert r.parameters["x"] == 0.5

    def test_positional_with_named_trailing(self):
        results = parse_function_call("function(arg1, arg2, named_param='value')")
        assert results[0].parameters["named_param"] == "value"

    def test_many_positional(self):
        results = parse_function_call("function(1, 2, 3, 4, 5)")
        r = results[0]
        for i in range(5):
            assert r.parameters[f"__arg_{i}__"] == i + 1

    def test_strings_with_kwargs(self):
        results = parse_function_call("function('a', 'b', 'c', x=1, y=2)")
        r = results[0]
        assert r.parameters["__arg_0__"] == "a"
        assert r.parameters["__arg_1__"] == "b"
        assert r.parameters["__arg_2__"] == "c"
        assert r.parameters["x"] == 1
        assert r.parameters["y"] == 2


# ---------------------------------------------------------------------------
# parse_function_call – nested parens / special characters (Bug 1 fixes)
# ---------------------------------------------------------------------------


class TestNestedParens:
    def test_closing_paren_inside_string(self):
        results = parse_function_call("tool(msg='hello ) world')")
        assert len(results) == 1
        assert results[0].function_name == "tool"
        assert results[0].parameters == {"msg": "hello ) world"}

    def test_tuple_argument(self):
        results = parse_function_call("tool(x=(1, 2))")
        assert len(results) == 1
        assert results[0].parameters == {"x": [1, 2]}

    def test_dict_with_paren_in_value(self):
        results = parse_function_call("tool(a={'nested': ')'})")
        assert len(results) == 1
        assert results[0].parameters == {"a": {"nested": ")"}}

    def test_mixed_nested_structures(self):
        results = parse_function_call("tool(items=[1, (2, 3), 4])")
        assert len(results) == 1
        assert results[0].parameters == {"items": [1, [2, 3], 4]}


# ---------------------------------------------------------------------------
# parse_function_call – multi-line (multiple calls)
# ---------------------------------------------------------------------------


class TestMultiLineParsing:
    def test_two_calls_on_separate_lines(self):
        text = "mobile.wait(seconds=3)\nmobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])"
        results = parse_function_call(text)
        assert len(results) == 2
        assert results[0].function_name == "mobile.wait"
        assert results[1].function_name == "mobile.swipe"


# ---------------------------------------------------------------------------
# extract_function_calls_from_text
# ---------------------------------------------------------------------------


class TestExtractFromText:
    CODE_BLOCK_REGEX = r"<code>.*?</code>"

    def test_no_code_block_returns_original_text_no_calls(self):
        text = "Hello world, no code blocks here"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert outside == text
        assert calls == []

    def test_extracts_calls_inside_code_block(self):
        text = "Sure, I'll do that.\n<code>mobile.click(x=0.5)</code>\nDone."
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert len(calls) == 1
        assert calls[0].function_name == "mobile.click"
        assert "mobile.click" not in outside

    def test_ignores_calls_outside_code_block(self):
        text = "mobile.click(x=0.5)\n<code>real.call(a=1)</code>\nmobile.home()"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        names = [c.function_name for c in calls]
        assert names == ["real.call"]

    def test_multiline_code_block(self):
        text = "Here:\n<code>\ndo.a()\ndo.b()\n</code>\nDone."
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        names = [c.function_name for c in calls]
        assert "do.a" in names
        assert "do.b" in names
        assert len(calls) == 2

    def test_multiple_code_blocks(self):
        text = "Step 1\n<code>a.first()</code>\nStep 2\n<code>b.second()</code>\nDone"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        names = [c.function_name for c in calls]
        assert names == ["a.first", "b.second"]

    def test_outside_text_excludes_code_blocks(self):
        text = "Hello\n<code>hidden()</code>\nWorld"
        outside, _ = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert "<code>" not in outside
        assert "hidden" not in outside
        assert "Hello" in outside
        assert "World" in outside

    def test_no_calls_when_code_block_has_no_functions(self):
        text = "<code>just plain text</code>"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert calls == []

    def test_nested_parens_inside_code_block(self):
        text = "<code>tool(msg='hello ) world')</code>"
        _, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert len(calls) == 1
        assert calls[0].parameters == {"msg": "hello ) world"}

    def test_recovers_simple_sibling_call_from_malformed_code_block(self, monkeypatch):
        fallback_used = False
        original_fallback = function_call_module._split_simple_calls_with_regex

        def spy_fallback(source: str) -> list[str]:
            nonlocal fallback_used
            fallback_used = True
            return original_fallback(source)

        monkeypatch.setattr(function_call_module, "_split_simple_calls_with_regex", spy_fallback)
        text = "Let me check.\n<code>camera(question='What is in front of me?') dance(</code>"
        outside, calls = extract_function_calls_from_text(text, block_regex=self.CODE_BLOCK_REGEX)
        assert "Let me check." in outside
        assert fallback_used
        assert len(calls) == 1
        assert calls[0].function_name == "camera"
        assert calls[0].parameters == {"question": "What is in front of me?"}


# ---------------------------------------------------------------------------
# to_realtime_function_tool_call – arg stripping & validation (Bug 2 fixes)
# ---------------------------------------------------------------------------


def _make_tool(name: str, properties: dict, required: list[str] | None = None) -> FunctionTool:
    schema = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    return FunctionTool(type="function", name=name, parameters=schema)


class TestToRealtimeToolCall:
    def test_positional_and_named_for_same_parameter_raises(self):
        fc = FunctionToolCall(
            function_name="greet",
            parameters={"__arg_0__": 1, "msg": "hi"},
            original_string="greet(1, msg='hi')",
        )
        tool = _make_tool("greet", {"msg": {"type": "string"}}, required=["msg"])
        with pytest.raises(ValueError, match="both a positional and a named value"):
            fc.to_realtime_function_tool_call([tool])

    def test_undeclared_args_stripped_when_required_present(self):
        fc = FunctionToolCall(
            function_name="greet",
            parameters={"msg": "hi", "bogus": 42},
            original_string="greet(msg='hi', bogus=42)",
        )
        tool = _make_tool("greet", {"msg": {"type": "string"}}, required=["msg"])
        result = fc.to_realtime_function_tool_call([tool])
        args = json.loads(result.arguments)
        assert "bogus" not in args
        assert args == {"msg": "hi"}

    def test_positional_binds_and_undeclared_named_still_stripped(self):
        fc = FunctionToolCall(
            function_name="greet",
            parameters={"__arg_0__": 1, "bogus": 2},
            original_string="greet(1, bogus=2)",
        )
        tool = _make_tool("greet", {"msg": {"type": "string"}}, required=["msg"])
        result = fc.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"msg": 1}

    def test_raises_when_required_missing_and_nothing_to_bind(self):
        fc = FunctionToolCall(
            function_name="greet",
            parameters={"bogus": 2},
            original_string="greet(bogus=2)",
        )
        tool = _make_tool("greet", {"msg": {"type": "string"}}, required=["msg"])
        with pytest.raises(ValueError, match="Missing required"):
            fc.to_realtime_function_tool_call([tool])

    def test_optional_only_tool_keeps_the_positional_value(self):
        fc = FunctionToolCall(
            function_name="noop",
            parameters={"__arg_0__": 1, "yy": 2},
            original_string="noop(1, yy=2)",
        )
        tool = _make_tool("noop", {"x": {"type": "integer"}})
        result = fc.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"x": 1}

    def test_no_collision_with_real_arg_prefix(self):
        """A real parameter named 'arg_0' should NOT be stripped."""
        fc = FunctionToolCall(
            function_name="calc",
            parameters={"arg_0": 10, "x": 5},
            original_string="calc(arg_0=10, x=5)",
        )
        tool = _make_tool(
            "calc",
            {"arg_0": {"type": "integer"}, "x": {"type": "integer"}},
            required=["arg_0"],
        )
        result = fc.to_realtime_function_tool_call([tool])
        args = json.loads(result.arguments)
        assert args == {"arg_0": 10, "x": 5}


class TestPositionalArgumentBinding:
    """Positional calls bind against the signature the tool prompt renders."""

    def test_single_positional_binds_to_required_parameter(self):
        tool = _make_tool("list_slots", {"month": {"type": "string"}}, required=["month"])
        (call,) = parse_function_call("list_slots('June')")
        result = call.to_realtime_function_tool_call([tool])
        assert result.name == "list_slots"
        assert json.loads(result.arguments) == {"month": "June"}

    def test_multiple_positionals_bind_in_order(self):
        tool = _make_tool(
            "hold_slot",
            {"date": {"type": "string"}, "time": {"type": "string"}},
            required=["date", "time"],
        )
        (call,) = parse_function_call("hold_slot('2026-06-01', '14:30')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"date": "2026-06-01", "time": "14:30"}

    def test_mixed_positional_and_named(self):
        tool = _make_tool(
            "hold_slot",
            {"date": {"type": "string"}, "time": {"type": "string"}},
            required=["date"],
        )
        (call,) = parse_function_call("hold_slot('2026-06-01', time='14:30')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"date": "2026-06-01", "time": "14:30"}

    def test_named_only_call_is_unchanged(self):
        tool = _make_tool("list_slots", {"month": {"type": "string"}}, required=["month"])
        (call,) = parse_function_call("list_slots(month='June')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"month": "June"}

    def test_binds_against_signature_order_not_schema_order(self):
        # signature_from_schema moves required-without-default parameters first,
        # so a positional value binds to 'month', not to the optional 'limit'.
        tool = _make_tool(
            "list_slots",
            {"limit": {"type": "integer", "default": 10}, "month": {"type": "string"}},
            required=["month"],
        )
        (call,) = parse_function_call("list_slots('June')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"month": "June"}

    def test_positional_can_fill_an_optional_parameter(self):
        tool = _make_tool(
            "list_slots",
            {"month": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
            required=["month"],
        )
        (call,) = parse_function_call("list_slots('June', 5)")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"month": "June", "limit": 5}

    def test_too_many_positionals_raise(self):
        tool = _make_tool("list_slots", {"month": {"type": "string"}}, required=["month"])
        (call,) = parse_function_call("list_slots('June', 5)")
        with pytest.raises(ValueError, match="Too many positional arguments"):
            call.to_realtime_function_tool_call([tool])

    def test_duplicate_positional_and_named_raises(self):
        tool = _make_tool("list_slots", {"month": {"type": "string"}}, required=["month"])
        (call,) = parse_function_call("list_slots('June', month='July')")
        with pytest.raises(ValueError, match="both a positional and a named value"):
            call.to_realtime_function_tool_call([tool])

    def test_without_tools_positional_arguments_are_still_dropped(self):
        (call,) = parse_function_call("list_slots('June')")
        result = call.to_realtime_function_tool_call()
        assert json.loads(result.arguments) == {}

    def test_positional_call_on_a_no_parameter_tool_raises(self):
        tool = _make_tool("ping", {})
        (call,) = parse_function_call("ping('now')")
        with pytest.raises(ValueError, match="Too many positional arguments"):
            call.to_realtime_function_tool_call([tool])
