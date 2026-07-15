from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Callable

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError

from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler


def _status_error(error_type: type[APIStatusError], status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_type("provider error", response=response, body=None)


def _responses_handler(create: Callable[..., object]) -> ResponsesApiModelHandler:
    handler = object.__new__(ResponsesApiModelHandler)
    handler.model_name = "test-model"
    handler.request_timeout = 20.0
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return handler


def _chat_completions_handler(create: Callable[..., object]) -> ChatCompletionsApiModelHandler:
    handler = object.__new__(ChatCompletionsApiModelHandler)
    handler.model_name = "test-model"
    handler.request_timeout = 20.0
    handler._extra_body = None
    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return handler


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_transient_rate_limit_does_not_fail_startup(make_handler, caplog):
    error = _status_error(RateLimitError, 429)

    def raise_rate_limit(**_kwargs):
        raise error

    handler = make_handler(raise_rate_limit)

    with caplog.at_level(logging.WARNING):
        handler.warmup()

    assert "continuing startup without warmup" in caplog.text


def test_retryable_warmup_errors_cover_connection_timeout_and_server_failures():
    request = httpx.Request("POST", "https://provider.example/v1/responses")

    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(
        APIConnectionError(message="connection failed", request=request)
    )
    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(APITimeoutError(request))
    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(_status_error(APIStatusError, 503))


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_non_retryable_warmup_error_still_fails_startup(make_handler):
    error = _status_error(AuthenticationError, 401)

    def raise_authentication_error(**_kwargs):
        raise error

    handler = make_handler(raise_authentication_error)

    with pytest.raises(AuthenticationError):
        handler.warmup()
