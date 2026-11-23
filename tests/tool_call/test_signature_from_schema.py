import json
from typing import Any, Literal, Optional, Union

import pytest

from speech_to_speech.LLM.tool_call.function_call import parse_function_call, parse_json_tool_call_block
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.signature_from_schema import _annotation_from_spec, signature_from_schema

# --- _annotation_from_spec tests ---


class TestAnnotationFromSpec:
    def test_basic_string(self):
        assert _annotation_from_spec({"type": "string"}) is str

    def test_basic_number(self):
        assert _annotation_from_spec({"type": "number"}) is float

    def test_basic_boolean(self):
        assert _annotation_from_spec({"type": "boolean"}) is bool

    def test_basic_integer(self):
        assert _annotation_from_spec({"type": "integer"}) is int

    def test_basic_object(self):
        assert _annotation_from_spec({"type": "object"}) is dict

    def test_basic_array(self):
        assert _annotation_from_spec({"type": "array"}) is list

    def test_null(self):
        assert _annotation_from_spec({"type": "null"}) is type(None)

    def test_enum(self):
        result = _annotation_from_spec({"type": "string", "enum": ["a", "b", "c"]})
        assert result == Literal["a", "b", "c"]

    def test_enum_empty(self):
        assert _annotation_from_spec({"enum": []}) is Any

    def test_const(self):
        assert _annotation_from_spec({"const": "turbo"}) == Literal["turbo"]

    def test_nullable_type_list(self):
        result = _annotation_from_spec({"type": ["string", "null"]})
        assert result == Optional[str]

    def test_any_of(self):
        spec = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert _annotation_from_spec(spec) == Union[str, int]

    def test_one_of(self):
        spec = {"oneOf": [{"type": "boolean"}, {"type": "number"}]}
        assert _annotation_from_spec(spec) == Union[bool, float]

    def test_one_of_single(self):
        spec = {"oneOf": [{"type": "string"}]}
        assert _annotation_from_spec(spec) is str

    def test_all_of_merge(self):
        spec = {"allOf": [{"type": "string"}, {"enum": ["x", "y"]}]}
        assert _annotation_from_spec(spec) == Literal["x", "y"]

    def test_array_with_items(self):
        spec = {"type": "array", "items": {"type": "integer"}}
        assert _annotation_from_spec(spec) == list[int]

    def test_array_with_nested_enum_items(self):
        spec = {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
        assert _annotation_from_spec(spec) == list[Literal["a", "b"]]

    def test_unknown_type(self):
        assert _annotation_from_spec({"type": "foobar"}) is Any

    def test_missing_type(self):
        assert _annotation_from_spec({}) is Any

    def test_none_spec(self):
        assert _annotation_from_spec(None) is Any


# --- signature_from_schema tests ---


class TestSignatureFromSchema:
    def test_empty_schema(self):
        sig = signature_from_schema({})
        assert str(sig) == "()"

    def test_none_schema(self):
        sig = signature_from_schema(None)
        assert str(sig) == "()"

    def test_no_properties(self):
        sig = signature_from_schema({"type": "object"})
        assert str(sig) == "()"

    def test_required_param(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(name: str)"

    def test_optional_param_defaults_to_none(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(name: str = None)"

    def test_schema_default_on_required(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer", "default": 5}},
            "required": ["count"],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(count: int = 5)"

    def test_schema_default_on_optional(self):
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(limit: int = 10)"

    def test_enum_required(self):
        schema = {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["left", "right"]},
            },
            "required": ["direction"],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(direction: Literal['left', 'right'])"

    def test_mixed_required_optional_no_star(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "verbose": {"type": "boolean"},
            },
            "required": ["query"],
        }
        sig = signature_from_schema(schema)
        assert "*" not in str(sig)
        assert str(sig) == "(query: str, limit: int = 10, verbose: bool = None)"

    def test_required_property_after_optional_property(self):
        """JSON Schema property order need not be valid Python parameter order."""
        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

        sig = signature_from_schema(schema)

        assert str(sig) == "(query: str, limit: int = None)"

    def test_all_required(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(a: str, b: int)"

    def test_all_optional(self):
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(x: float = None, y: float = None)"

    def test_invalid_identifier_hyphen(self):
        """Property name with hyphen should be sanitized to underscore."""
        schema = {
            "type": "object",
            "properties": {"user-name": {"type": "string"}},
            "required": ["user-name"],
        }
        sig = signature_from_schema(schema)
        # Hyphen replaced with underscore
        assert str(sig) == "(user_name: str)"

    def test_invalid_identifier_space(self):
        """Property name with space should be sanitized to underscore."""
        schema = {
            "type": "object",
            "properties": {"first name": {"type": "string"}},
            "required": ["first name"],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(first_name: str)"

    def test_python_keyword(self):
        """Python keyword as property name should get trailing underscore."""
        schema = {
            "type": "object",
            "properties": {"class": {"type": "string"}},
            "required": ["class"],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(class_: str)"

    def test_starts_with_digit(self):
        """Property name starting with digit should get leading underscore."""
        schema = {
            "type": "object",
            "properties": {"123abc": {"type": "string"}},
            "required": ["123abc"],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(_123abc: str)"

    def test_collision_handling(self):
        """Multiple properties sanitizing to same name should be disambiguated."""
        schema = {
            "type": "object",
            "properties": {
                "user-name": {"type": "string"},
                "user.name": {"type": "integer"},
            },
            "required": ["user-name", "user.name"],
        }
        sig = signature_from_schema(schema)
        # Both sanitize to user_name, second gets _2 suffix
        params = list(sig.parameters.keys())
        assert params == ["user_name", "user_name_2"]
        assert sig.parameters["user_name"].annotation is str
        assert sig.parameters["user_name_2"].annotation is int

    def test_empty_property_name(self):
        """Empty property name should become underscore."""
        schema = {
            "type": "object",
            "properties": {"": {"type": "string"}},
            "required": [""],
        }
        sig = signature_from_schema(schema)
        assert str(sig) == "(_: str)"

    def test_valid_unicode_identifier_is_unchanged(self):
        schema = {
            "type": "object",
            "properties": {"città": {"type": "string"}},
            "required": ["città"],
        }
        assert str(signature_from_schema(schema)) == "(città: str)"

    def test_alias_skips_a_literal_property_with_the_suffixed_name(self):
        schema = {
            "type": "object",
            "properties": {
                "user-name": {"type": "string"},
                "user.name": {"type": "string"},
                "user_name_2": {"type": "string"},
            },
        }
        params = list(signature_from_schema(schema).parameters)
        assert params == ["user_name", "user_name_3", "user_name_2"]


# --- Calls against sanitized parameter names ---


def _tool(name: str, parameters: dict) -> FunctionTool:
    return FunctionTool(type="function", name=name, parameters=parameters)


# Optional "user-name" comes first in properties, required "user.name" second, so
# the signature puts the required parameter first.
COLLIDING_SCHEMA = {
    "type": "object",
    "properties": {
        "user-name": {"type": "string"},
        "user.name": {"type": "integer"},
    },
    "required": ["user.name"],
}


class TestSanitizedNameCalls:
    def test_colliding_names_prompt(self):
        tool = _tool("lookup", COLLIDING_SCHEMA)
        assert tool.to_code_prompt(include_args_doc=False).startswith(
            "def lookup(user_name_2: int, user_name: str = None)"
        )

    def test_colliding_names_named_call(self):
        tool = _tool("lookup", COLLIDING_SCHEMA)
        (call,) = parse_function_call("lookup(user_name_2=5, user_name='bob')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"user.name": 5, "user-name": "bob"}

    def test_colliding_names_positional_call(self):
        tool = _tool("lookup", COLLIDING_SCHEMA)
        (call,) = parse_function_call("lookup(5, 'bob')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"user.name": 5, "user-name": "bob"}

    def test_missing_required_aliased_parameter_raises(self):
        tool = _tool("lookup", COLLIDING_SCHEMA)
        (call,) = parse_function_call("lookup(user_name='bob')")
        with pytest.raises(ValueError, match="Missing required"):
            call.to_realtime_function_tool_call([tool])

    def test_colliding_names_json_call_keeps_schema_keys(self):
        tool = _tool("lookup", COLLIDING_SCHEMA)
        (call,) = parse_json_tool_call_block(
            '<tool_call>{"name": "lookup", "arguments": {"user.name": 5, "user-name": "bob"}}</tool_call>'
        )
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"user.name": 5, "user-name": "bob"}

    def test_json_call_does_not_remap_a_key_equal_to_an_alias(self):
        # "user-name" renders as user_name_2 because "user_name" is a property too.
        tool = _tool(
            "lookup",
            {
                "type": "object",
                "properties": {"user-name": {"type": "string"}, "user_name": {"type": "string"}},
            },
        )
        (call,) = parse_json_tool_call_block(
            '<tool_call>{"name": "lookup", "arguments": {"user_name": "a", "user_name_2": "b"}}</tool_call>'
        )
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"user_name": "a"}

    def test_literal_property_equal_to_a_suffix_alias(self):
        tool = _tool(
            "lookup",
            {
                "type": "object",
                "properties": {
                    "user-name": {"type": "string"},
                    "user.name": {"type": "string"},
                    "user_name_2": {"type": "string"},
                },
            },
        )
        (call,) = parse_function_call("lookup(user_name='a', user_name_3='b', user_name_2='c')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"user-name": "a", "user.name": "b", "user_name_2": "c"}

    def test_required_property_with_default_is_still_required(self):
        tool = _tool(
            "lookup",
            {
                "type": "object",
                "properties": {"page-size": {"type": "integer", "default": 10}},
                "required": ["page-size"],
            },
        )
        (call,) = parse_function_call("lookup()")
        with pytest.raises(ValueError, match="Missing required"):
            call.to_realtime_function_tool_call([tool])

    def test_empty_property_name_maps_back_and_stays_required(self):
        tool = _tool(
            "lookup",
            {
                "type": "object",
                "properties": {"_": {"type": "string"}, "": {"type": "string"}},
                "required": [""],
            },
        )
        assert str(signature_from_schema(tool.parameters)) == "(__2: str, _: str = None)"
        (call,) = parse_function_call("lookup(__2='x')")
        assert json.loads(call.to_realtime_function_tool_call([tool]).arguments) == {"": "x"}
        (call,) = parse_function_call("lookup(_='y')")
        with pytest.raises(ValueError, match="Missing required"):
            call.to_realtime_function_tool_call([tool])

    def test_unicode_identifier_call(self):
        tool = _tool(
            "lookup",
            {"type": "object", "properties": {"città": {"type": "string"}}, "required": ["città"]},
        )
        (call,) = parse_function_call("lookup(città='Roma')")
        result = call.to_realtime_function_tool_call([tool])
        assert json.loads(result.arguments) == {"città": "Roma"}


# --- Tool.to_code_prompt tests ---


class TestToolToCodePrompt:
    def _make_tool(self, name, description, parameters):
        tool = FunctionTool()
        tool.name = name
        tool.description = description
        tool.type = "function"
        tool.parameters = parameters
        return tool

    def test_basic_code_prompt(self):
        tool = self._make_tool(
            "greet",
            "Greet the user.",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User name."},
                },
                "required": ["name"],
            },
        )
        result = tool.to_code_prompt(include_args_doc=True)
        assert "def greet(name: str):" in result
        assert "Greet the user." in result
        assert "name: User name." in result

    def test_no_params(self):
        tool = self._make_tool(
            "ping",
            "Ping the server.",
            {
                "type": "object",
                "properties": {},
            },
        )
        result = tool.to_code_prompt()
        assert "def ping():" in result

    def test_enum_and_optional(self):
        tool = self._make_tool(
            "move",
            "Move robot.",
            {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Direction.",
                    },
                    "speed": {"type": "number", "description": "Speed."},
                },
                "required": ["direction"],
            },
        )
        result = tool.to_code_prompt()
        assert "Literal['left', 'right']" in result
        assert "speed: float = None" in result
        assert "*" not in result.split("\n")[0]
