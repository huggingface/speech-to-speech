from typing import Any, Literal, Optional, Union

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

    def test_empty_dict(self):
        assert _annotation_from_spec({}) is Any


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
