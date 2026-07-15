from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _status_error(
    error_type: type[APIStatusError],
    status_code: int,
    headers: dict[str, str] | None = None,
) -> APIStatusError:
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    response = httpx.Response(status_code, headers=headers, request=request)
    return error_type("provider error", response=response, body=None)


def _responses_handler(create: Callable[..., object], warmup_timeout_s: float = 30.0) -> ResponsesApiModelHandler:
    handler = object.__new__(ResponsesApiModelHandler)
    handler.model_name = "test-model"
    handler.request_timeout_s = 20.0
    handler.warmup_timeout_s = warmup_timeout_s
    handler._warmup_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return handler


def _chat_completions_handler(
    create: Callable[..., object], warmup_timeout_s: float = 30.0
) -> ChatCompletionsApiModelHandler:
    handler = object.__new__(ChatCompletionsApiModelHandler)
    handler.model_name = "test-model"
    handler.request_timeout_s = 20.0
    handler.warmup_timeout_s = warmup_timeout_s
    handler._extra_body = None
    handler._warmup_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return handler


def _patch_retry_timing(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(base_mod.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(base_mod.time, "sleep", clock.sleep)
    monkeypatch.setattr(base_mod.random, "random", lambda: 0.5)
    monkeypatch.setattr(base_mod.random, "uniform", lambda low, high: (low + high) / 2)
    return clock


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_warmup_succeeds_without_sleeping_on_first_attempt(make_handler, monkeypatch):
    attempts = []

    def succeed(**kwargs):
        attempts.append(kwargs)

    handler = make_handler(succeed)
    clock = _patch_retry_timing(monkeypatch)

    handler.warmup()

    assert len(attempts) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_transient_rate_limit_retries_with_jitter_until_warmup_succeeds(make_handler, monkeypatch):
    error = _status_error(RateLimitError, 429)
    attempts = 0

    def raise_rate_limit(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 6:
            raise error

    handler = make_handler(raise_rate_limit)
    clock = _patch_retry_timing(monkeypatch)

    handler.warmup()

    assert attempts == 6
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_transient_warmup_failure_still_fails_at_deadline(make_handler, monkeypatch):
    error = _status_error(RateLimitError, 429)
    attempts = 0

    def raise_rate_limit(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise error

    handler = make_handler(raise_rate_limit)
    clock = _patch_retry_timing(monkeypatch)

    with pytest.raises(RateLimitError):
        handler.warmup()

    assert attempts == 6
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_retry_after_is_respected_with_positive_jitter(monkeypatch):
    error = _status_error(RateLimitError, 429, {"retry-after": "20"})
    attempts = 0

    def recover(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error

    handler = _responses_handler(recover)
    clock = _patch_retry_timing(monkeypatch)

    handler.warmup()

    assert attempts == 2
    assert clock.sleeps == [20.5]


def test_exponential_backoff_includes_jitter(monkeypatch):
    error = _status_error(RateLimitError, 429)

    monkeypatch.setattr(base_mod.random, "random", lambda: 0.0)
    lower_delay = BaseOpenAICompatibleHandler._warmup_retry_delay(1, error)
    monkeypatch.setattr(base_mod.random, "random", lambda: 1.0)
    upper_delay = BaseOpenAICompatibleHandler._warmup_retry_delay(1, error)

    assert lower_delay == 0.5
    assert upper_delay == 1.5


def test_retryable_statuses_and_provider_override():
    for status_code in (408, 409, 429, 503):
        assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(_status_error(APIStatusError, status_code))
    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(
        _status_error(APIStatusError, 400, {"x-should-retry": "true"})
    )
    assert not BaseOpenAICompatibleHandler._is_retryable_warmup_error(
        _status_error(RateLimitError, 429, {"x-should-retry": "false"})
    )


def test_connection_errors_are_retryable():
    request = httpx.Request("POST", "https://provider.example/v1/responses")
    assert BaseOpenAICompatibleHandler._is_retryable_warmup_error(
        APIConnectionError(message="connection failed", request=request)
    )


@pytest.mark.parametrize("make_handler", [_responses_handler, _chat_completions_handler])
def test_non_retryable_warmup_error_still_fails_without_retry(make_handler, monkeypatch):
    error = _status_error(AuthenticationError, 401)
    attempts = 0

    def raise_authentication_error(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise error

    handler = make_handler(raise_authentication_error)
    clock = _patch_retry_timing(monkeypatch)

    with pytest.raises(AuthenticationError):
        handler.warmup()

    assert attempts == 1
    assert clock.sleeps == []


def test_attempt_timeout_is_capped_by_remaining_warmup_budget(monkeypatch):
    timeouts = []

    def succeed(**kwargs):
        timeouts.append(kwargs["timeout"])

    handler = _responses_handler(succeed, warmup_timeout_s=5.0)
    _patch_retry_timing(monkeypatch)

    handler.warmup()

    assert len(timeouts) == 1
    assert timeouts[0].read == 5.0
    assert timeouts[0].connect == 5.0
