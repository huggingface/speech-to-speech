"""Tests for MiniMax language model handler."""

import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LLM.chat import Chat
from LLM.minimax_language_model import MiniMaxModelHandler, THINK_TAG_PATTERN


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestThinkTagStripping:
    """Test that MiniMax thinking tags are properly stripped."""

    def test_strips_think_tags(self):
        text = "<think>Let me think about this...</think>Hello!"
        result = THINK_TAG_PATTERN.sub("", text).strip()
        assert result == "Hello!"

    def test_strips_multiline_think_tags(self):
        text = "<think>\nStep 1: analyze\nStep 2: respond\n</think>\nThe answer is 42."
        result = THINK_TAG_PATTERN.sub("", text).strip()
        assert result == "The answer is 42."

    def test_no_think_tags(self):
        text = "Just a normal response."
        result = THINK_TAG_PATTERN.sub("", text).strip()
        assert result == "Just a normal response."

    def test_empty_think_tags(self):
        text = "<think></think>Response."
        result = THINK_TAG_PATTERN.sub("", text).strip()
        assert result == "Response."


class TestTemperatureClamping:
    """Test that temperature is clamped to MiniMax's [0.0, 1.0] range."""

    def _make_handler(self, gen_kwargs, api_key="test-key"):
        handler = object.__new__(MiniMaxModelHandler)
        # Patch os.environ for auto-detection
        os.environ["MINIMAX_API_KEY"] = api_key
        try:
            handler.setup(gen_kwargs=gen_kwargs)
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        return handler

    def test_temperature_above_max_clamped(self, monkeypatch):
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        os.environ["MINIMAX_API_KEY"] = "test-key"
        try:
            handler.setup(gen_kwargs={"temperature": 1.5})
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        assert handler._captured_kwargs["gen_kwargs"]["temperature"] == 1.0

    def test_temperature_below_min_clamped(self, monkeypatch):
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        os.environ["MINIMAX_API_KEY"] = "test-key"
        try:
            handler.setup(gen_kwargs={"temperature": -0.5})
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        assert handler._captured_kwargs["gen_kwargs"]["temperature"] == 0.0

    def test_temperature_in_range_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        os.environ["MINIMAX_API_KEY"] = "test-key"
        try:
            handler.setup(gen_kwargs={"temperature": 0.7})
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        assert handler._captured_kwargs["gen_kwargs"]["temperature"] == 0.7

    def test_no_temperature_no_error(self, monkeypatch):
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        os.environ["MINIMAX_API_KEY"] = "test-key"
        try:
            handler.setup(gen_kwargs={})
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        assert "temperature" not in handler._captured_kwargs["gen_kwargs"]


def _capture_setup(self, **kwargs):
    """Helper to capture kwargs passed to parent setup."""
    self._captured_kwargs = kwargs


class TestApiKeyDetection:
    """Test MiniMax API key auto-detection from environment."""

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        handler = object.__new__(MiniMaxModelHandler)
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            handler.setup(api_key=None)

    def test_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key-123")
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        handler.setup()
        assert handler._captured_kwargs["api_key"] == "env-key-123"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        handler.setup(api_key="explicit-key")
        assert handler._captured_kwargs["api_key"] == "explicit-key"


class TestDefaultValues:
    """Test MiniMax handler default configuration."""

    def test_default_base_url(self, monkeypatch):
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        os.environ["MINIMAX_API_KEY"] = "test-key"
        try:
            handler.setup()
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        assert handler._captured_kwargs["base_url"] == "https://api.minimax.io/v1"

    def test_default_model(self, monkeypatch):
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        os.environ["MINIMAX_API_KEY"] = "test-key"
        try:
            handler.setup()
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        assert handler._captured_kwargs["model_name"] == "MiniMax-M2.7"

    def test_disable_thinking_off(self, monkeypatch):
        monkeypatch.setattr(
            "LLM.minimax_language_model.OpenApiModelHandler.setup",
            lambda self, **kw: _capture_setup(self, **kw),
        )
        handler = object.__new__(MiniMaxModelHandler)
        os.environ["MINIMAX_API_KEY"] = "test-key"
        try:
            handler.setup()
        finally:
            os.environ.pop("MINIMAX_API_KEY", None)
        assert handler._captured_kwargs["disable_thinking"] is False


class TestProcessThinkTagStripping:
    """Test that process() strips think tags from generated output."""

    def test_strips_think_tags_in_process(self, monkeypatch):
        from LLM import openai_api_language_model

        monkeypatch.setattr(
            openai_api_language_model,
            "sent_tokenize",
            lambda text: [text] if text else [],
        )

        handler = object.__new__(MiniMaxModelHandler)
        handler.model_name = "MiniMax-M2.7"
        handler.stream = False
        handler.gen_kwargs = {}
        handler.disable_thinking = False
        handler.user_role = "user"
        handler.chat = Chat(1)

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<think>reasoning here</think>The answer is 42."
                    )
                )
            ]
        )
        handler.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: response,
                )
            )
        )

        outputs = list(handler.process("What is the meaning of life?"))
        assert len(outputs) == 1
        text, lang, tools = outputs[0]
        assert text == "The answer is 42."
        assert "<think>" not in text

    def test_no_think_tags_passthrough(self, monkeypatch):
        from LLM import openai_api_language_model

        monkeypatch.setattr(
            openai_api_language_model,
            "sent_tokenize",
            lambda text: [text] if text else [],
        )

        handler = object.__new__(MiniMaxModelHandler)
        handler.model_name = "MiniMax-M2.7"
        handler.stream = False
        handler.gen_kwargs = {}
        handler.disable_thinking = False
        handler.user_role = "user"
        handler.chat = Chat(1)

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Normal response.")
                )
            ]
        )
        handler.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: response,
                )
            )
        )

        outputs = list(handler.process("Hello"))
        assert outputs[0][0] == "Normal response."


class TestStreamingWithThinkTags:
    """Test streaming mode strips think tags correctly."""

    def test_streaming_strips_think_tags(self, monkeypatch):
        from LLM import openai_api_language_model

        monkeypatch.setattr(
            openai_api_language_model,
            "sent_tokenize",
            lambda text: [s for s in text.split(". ") if s],
        )

        def _chunk(content):
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
            )

        chunks = [
            _chunk("<think>thinking</think>Hello. "),
            _chunk("World."),
        ]

        handler = object.__new__(MiniMaxModelHandler)
        handler.model_name = "MiniMax-M2.7"
        handler.stream = True
        handler.gen_kwargs = {}
        handler.disable_thinking = False
        handler.user_role = "user"
        handler.chat = Chat(1)
        handler.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: iter(chunks),
                )
            )
        )

        outputs = list(handler.process("Hi"))
        # All outputs should have think tags stripped
        for text, _, _ in outputs:
            assert "<think>" not in text


class TestArgumentsDataclass:
    """Test MiniMax arguments dataclass."""

    def test_default_values(self):
        from arguments_classes.minimax_language_model_arguments import (
            MiniMaxLanguageModelHandlerArguments,
        )

        args = MiniMaxLanguageModelHandlerArguments()
        assert args.minimax_model_name == "MiniMax-M2.7"
        assert args.minimax_base_url == "https://api.minimax.io/v1"
        assert args.minimax_api_key is None
        assert args.minimax_stream is False
        assert args.minimax_user_role == "user"
        assert args.minimax_init_chat_role == "system"
        assert args.minimax_chat_size == 5


class TestPipelineIntegration:
    """Test that MiniMax is properly registered in the pipeline."""

    def test_minimax_handler_is_subclass_of_openapi(self):
        """MiniMaxModelHandler extends OpenApiModelHandler."""
        from LLM.openai_api_language_model import OpenApiModelHandler

        assert issubclass(MiniMaxModelHandler, OpenApiModelHandler)

    def test_minimax_arguments_rename(self):
        """Test that argument renaming works for minimax prefix."""
        from arguments_classes.minimax_language_model_arguments import (
            MiniMaxLanguageModelHandlerArguments,
        )
        from copy import copy

        args = MiniMaxLanguageModelHandlerArguments()

        # Simulate rename_args logic
        gen_kwargs = {}
        for key in copy(args.__dict__):
            if key.startswith("minimax"):
                value = args.__dict__.pop(key)
                new_key = key[len("minimax") + 1:]
                if new_key.startswith("gen_"):
                    gen_kwargs[new_key[4:]] = value
                else:
                    args.__dict__[new_key] = value
        args.__dict__["gen_kwargs"] = gen_kwargs

        assert args.model_name == "MiniMax-M2.7"
        assert args.base_url == "https://api.minimax.io/v1"
        assert args.api_key is None
        assert args.stream is False

    def test_minimax_handler_inherits_process(self):
        """MiniMaxModelHandler has its own process method that wraps parent."""
        assert hasattr(MiniMaxModelHandler, "process")
        # It should override process (not just inherit it)
        from LLM.openai_api_language_model import OpenApiModelHandler
        assert MiniMaxModelHandler.process is not OpenApiModelHandler.process


# ---------------------------------------------------------------------------
# Integration tests (require MINIMAX_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_non_streaming_response(self):
        handler = object.__new__(MiniMaxModelHandler)
        handler.setup(
            model_name="MiniMax-M2.7",
            api_key=os.environ["MINIMAX_API_KEY"],
            stream=False,
        )

        outputs = list(handler.process("Say hello in one word."))
        assert len(outputs) >= 1
        text, _, _ = outputs[0]
        assert isinstance(text, str)
        assert len(text) > 0

    def test_streaming_response(self):
        handler = object.__new__(MiniMaxModelHandler)
        handler.setup(
            model_name="MiniMax-M2.7",
            api_key=os.environ["MINIMAX_API_KEY"],
            stream=True,
        )

        outputs = list(handler.process("Say hello in one word."))
        assert len(outputs) >= 1
        full_text = "".join(t for t, _, _ in outputs)
        assert len(full_text) > 0

    def test_session_reset(self):
        handler = object.__new__(MiniMaxModelHandler)
        handler.setup(
            model_name="MiniMax-M2.7",
            api_key=os.environ["MINIMAX_API_KEY"],
            stream=False,
        )

        list(handler.process("Hello"))
        assert len(handler.chat.buffer) > 0

        handler.on_session_end()
        assert len(handler.chat.buffer) == 0
