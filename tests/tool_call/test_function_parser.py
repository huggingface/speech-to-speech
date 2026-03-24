import pytest

from LLM.tool_call.function_call import (
    extract_function_calls_from_text,
    parse_function_call,
)


# ---------------------------------------------------------------------------
# parse_function_call – single calls
# ---------------------------------------------------------------------------

class TestParseFunctionCall:

    @pytest.mark.parametrize("call_str, expected_name, expected_params", [
        ("mobile.home()", "mobile.home", {}),
        ("mobile.back()", "mobile.back", {}),
        ("mobile.open_app(app_name='drupe')", "mobile.open_app", {"app_name": "drupe"}),
        ("mobile.long_press(x=0.799, y=0.911)", "mobile.long_press", {"x": 0.799, "y": 0.911}),
        ("mobile.terminate(status='success')", "mobile.terminate", {"status": "success"}),
        ("answer('text')", "answer", {"arg_0": "text"}),
        ("pyautogui.hscroll(page=-0.1)", "pyautogui.hscroll", {"page": -0.1}),
        ("pyautogui.scroll(page=-0.1)", "pyautogui.scroll", {"page": -0.1}),
        ("pyautogui.scroll(0.13)", "pyautogui.scroll", {"arg_0": 0.13}),
        ("pyautogui.click(x=0.8102, y=0.9463)", "pyautogui.click", {"x": 0.8102, "y": 0.9463}),
        ("pyautogui.hotkey(keys=['ctrl', 'c'])", "pyautogui.hotkey", {"keys": ["ctrl", "c"]}),
        ("pyautogui.press(keys='enter')", "pyautogui.press", {"keys": "enter"}),
        ("pyautogui.press(keys=['enter'])", "pyautogui.press", {"keys": ["enter"]}),
        ("pyautogui.moveTo(x=0.04, y=0.405)", "pyautogui.moveTo", {"x": 0.04, "y": 0.405}),
        ("pyautogui.write(message='bread buns')", "pyautogui.write", {"message": "bread buns"}),
        ("pyautogui.dragTo(x=0.8102, y=0.9463)", "pyautogui.dragTo", {"x": 0.8102, "y": 0.9463}),
    ])
    def test_single_call(self, call_str, expected_name, expected_params):
        results = parse_function_call(call_str)
        assert len(results) == 1
        assert results[0].function_name == expected_name
        assert results[0].parameters == expected_params

    def test_swipe_with_list_params(self):
        results = parse_function_call(
            "mobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])"
        )
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
        assert r.parameters["arg_0"] == "hello"
        assert r.parameters["arg_1"] == 123
        assert r.parameters["x"] == 0.5

    def test_positional_with_named_trailing(self):
        results = parse_function_call("function(arg1, arg2, named_param='value')")
        assert results[0].parameters["named_param"] == "value"

    def test_many_positional(self):
        results = parse_function_call("function(1, 2, 3, 4, 5)")
        r = results[0]
        for i in range(5):
            assert r.parameters[f"arg_{i}"] == i + 1

    def test_strings_with_kwargs(self):
        results = parse_function_call("function('a', 'b', 'c', x=1, y=2)")
        r = results[0]
        assert r.parameters["arg_0"] == "a"
        assert r.parameters["arg_1"] == "b"
        assert r.parameters["arg_2"] == "c"
        assert r.parameters["x"] == 1
        assert r.parameters["y"] == 2


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


