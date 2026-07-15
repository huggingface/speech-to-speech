from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
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
def test_transient_rate_limit_retries_until_warmup_succeeds(make_handler, monkeypatch):
    error = _status_error(RateLimitError, 429)
    attempts = 0
    sleeps = []

    def raise_rate_limit(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 6:
            raise error

    handler = make_handler(raise_rate_limit)
    monkeypatch.setattr(base_mod.time, "sleep", sleeps.append)

    handler.warmup()

    assert attempts == 6
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_transient_warmup_failure_still_fails_startup_after_retries(make_handler, monkeypatch):
    error = _status_error(RateLimitError, 429)
    attempts = 0
    sleeps = []

    def raise_rate_limit(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise error

    handler = make_handler(raise_rate_limit)
    monkeypatch.setattr(base_mod.time, "sleep", sleeps.append)

    with pytest.raises(RateLimitError):
        handler.warmup()

    assert attempts == 6
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_retryable_warmup_errors_cover_connection_timeout_and_server_failures():
    request = httpx.Request("POST", "https://provider.example/v1/responses")

    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(
        APIConnectionError(message="connection failed", request=request)
    )
    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(APITimeoutError(request))
    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(_status_error(APIStatusError, 503))


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_non_retryable_warmup_error_still_fails_startup_without_retry(make_handler, monkeypatch):
    error = _status_error(AuthenticationError, 401)
    attempts = 0

    def raise_authentication_error(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise error

    handler = make_handler(raise_authentication_error)
    monkeypatch.setattr(base_mod.time, "sleep", lambda _seconds: pytest.fail("non-retryable error was retried"))

    with pytest.raises(AuthenticationError):
        handler.warmup()

    assert attempts == 1
