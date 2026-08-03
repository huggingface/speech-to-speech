"""LLM proxy: OpenAI-compatible passthrough for remote LLM backends.

Mounts ``POST /v1/chat/completions`` (backend ``chat-completions``) or
``POST /v1/responses`` (backend ``responses-api``) as a passthrough to the
configured upstream provider. The server performs no authentication and no
throttling of its own: it is designed to run on a trusted network or behind
a gateway that owns access control (the s2s-endpoint compute replica gates
these paths with the session's HF token). Requests never touch pipeline
units' queues or cancel scopes, so proxied generations run fully concurrent
with the speech pipeline and are never interrupted by new speech.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM_BASE_URL = "https://api.openai.com/v1"
_PATHS = {
    "chat-completions": "/v1/chat/completions",
    "responses-api": "/v1/responses",
}


class LLMProxyConfig(BaseModel):
    enabled: bool = False
    llm_backend: str | None = None
    upstream_base_url: str | None = None
    upstream_api_key: str | None = None
    model_name: str | None = None
    connect_timeout_s: float = 10.0


class LLMProxyUsage(BaseModel):
    """Replica-local proxy counters, reset with the process.

    Surfaced as the additive ``llm_proxy`` section of the usage endpoint.
    429 is its own bucket (not double-counted under 4xx) so a melting
    client is visible at a glance.
    """

    requests: int = 0
    responses_2xx: int = 0
    responses_4xx: int = 0
    responses_429: int = 0
    responses_5xx: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record_status(self, status: int) -> None:
        self.requests += 1
        if status == 429:
            self.responses_429 += 1
        elif 200 <= status < 300:
            self.responses_2xx += 1
        elif 400 <= status < 500:
            self.responses_4xx += 1
        elif status >= 500:
            self.responses_5xx += 1

    def record_token_payload(self, payload: Any) -> None:
        """Accumulate tokens from any upstream JSON payload that carries usage.

        Handles the three shapes the proxy sees: chat completions bodies and
        stream chunks (``usage`` at the top level, prompt/completion keys),
        Responses bodies (``usage`` at the top level, input/output keys), and
        Responses streaming events (``usage`` under ``response``).
        """
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            response = payload.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if isinstance(input_tokens, int):
            self.input_tokens += input_tokens
        if isinstance(output_tokens, int):
            self.output_tokens += output_tokens

    def record_sse_event(self, event: bytes) -> None:
        for line in event.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                continue
            self.record_token_payload(payload)


class _ErrorDetail(BaseModel):
    message: str
    type: str


class _ErrorEnvelope(BaseModel):
    error: _ErrorDetail


def _error_response(status_code: int, message: str, error_type: str) -> Response:
    envelope = _ErrorEnvelope(error=_ErrorDetail(message=message, type=error_type))
    return Response(
        content=envelope.model_dump_json(),
        status_code=status_code,
        media_type="application/json",
    )


def _verbatim_response(upstream: httpx.Response, content: bytes) -> Response:
    return Response(
        content=content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


def _upstream_unreachable(error: httpx.HTTPError) -> Response:
    logger.warning(f"LLM proxy: upstream request failed: {type(error).__name__}: {error}")
    return _error_response(502, f"Upstream request failed: {type(error).__name__}", "upstream_unreachable")


def mount_llm_proxy(app: FastAPI, config: LLMProxyConfig | None) -> LLMProxyUsage:
    """Mount the proxy paths on *app* according to *config*.

    Both known paths always answer: the path matching an enabled remote
    backend proxies, every other combination answers 501 naming the reason,
    so a misconfigured deployment is diagnosed in one request.

    Returns the usage counters the passthrough records into, for the usage
    endpoint to surface (all zeros when the proxy is unavailable).
    """
    config = config or LLMProxyConfig()
    usage = LLMProxyUsage()

    if not config.enabled:
        reason = "The LLM proxy is disabled. Start the server with --enable_llm_proxy to enable it."
    elif config.llm_backend not in _PATHS:
        reason = (
            f"The LLM proxy requires a remote LLM backend; this server runs '{config.llm_backend}'. "
            "It works with --llm_backend chat-completions or --llm_backend responses-api."
        )
    else:
        reason = None

    if reason is not None:
        for path in _PATHS.values():
            _mount_unavailable(app, path, reason)
        return usage

    assert config.llm_backend is not None
    serving_path = _PATHS[config.llm_backend]
    for path in _PATHS.values():
        if path == serving_path:
            _mount_passthrough(app, path, config, usage)
        else:
            _mount_unavailable(
                app,
                path,
                f"This server runs the '{config.llm_backend}' backend; use {serving_path} instead.",
            )
    return usage


def _mount_unavailable(app: FastAPI, path: str, reason: str) -> None:
    @app.post(path)
    async def unavailable_endpoint() -> Response:
        return _error_response(501, reason, "not_implemented")


def _mount_passthrough(
    app: FastAPI,
    path: str,
    config: LLMProxyConfig,
    usage: LLMProxyUsage,
) -> None:
    base_url = (config.upstream_base_url or DEFAULT_UPSTREAM_BASE_URL).rstrip("/")
    upstream_url = base_url + path.removeprefix("/v1")

    @app.post(path)
    async def passthrough_endpoint(request: Request) -> Response:
        response = await _proxy(request)
        usage.record_status(response.status_code)
        return response

    async def _proxy(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return _error_response(400, "Request body must be valid JSON.", "invalid_request_error")
        if not isinstance(body, dict):
            # Valid JSON is not necessarily an object; anything else cannot
            # carry the mutations below and would be rejected upstream anyway.
            return _error_response(400, "Request body must be a JSON object.", "invalid_request_error")
        body["model"] = config.model_name
        if path == _PATHS["responses-api"]:
            # Session holders are anonymous; forcing store off keeps them from
            # creating persistent state on the operator's provider account.
            body["store"] = False
        elif body.get("stream"):
            # Streamed chat completions only report usage when asked; inject
            # the ask so the proxy can account tokens. The client receives
            # every upstream frame verbatim, including the final usage chunk
            # it may not have requested (valid protocol). Responses streams
            # always carry usage on response.completed, so no mutation there.
            # A non-dict stream_options is left alone for the upstream to
            # reject with its own error.
            stream_options = body.get("stream_options")
            if stream_options is None or isinstance(stream_options, dict):
                body["stream_options"] = {**(stream_options or {}), "include_usage": True}

        headers = {"Authorization": f"Bearer {config.upstream_api_key}"}
        # Generation can legitimately take minutes, so only the connect
        # phase gets a timeout; reads (streamed or not) have none.
        timeout = httpx.Timeout(None, connect=config.connect_timeout_s)

        if not body.get("stream"):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    upstream = await client.post(upstream_url, json=body, headers=headers)
            except httpx.HTTPError as e:
                return _upstream_unreachable(e)
            if upstream.status_code < 400:
                try:
                    usage.record_token_payload(upstream.json())
                except ValueError:
                    pass
            return _verbatim_response(upstream, upstream.content)

        # Streaming: forward every upstream byte as it arrives. The client
        # and upstream response stay open for the response's lifetime and are
        # closed by the background task after the last byte is sent.
        client = httpx.AsyncClient(timeout=timeout)
        try:
            upstream_request = client.build_request("POST", upstream_url, json=body, headers=headers)
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as e:
            await client.aclose()
            return _upstream_unreachable(e)
        except Exception:
            await client.aclose()
            raise

        async def _cleanup() -> None:
            await upstream.aclose()
            await client.aclose()

        if upstream.status_code >= 400:
            # Error before any frame: pass status and body through verbatim,
            # buffered, exactly as a non-streaming answer would be.
            content = await upstream.aread()
            await _cleanup()
            return _verbatim_response(upstream, content)

        async def _stream_and_cleanup() -> AsyncIterator[bytes]:
            # The generator owns the cleanup: Starlette only runs background
            # tasks after a successful send, so an upstream failure or a
            # client disconnect mid-stream would otherwise leak the httpx
            # client and its connection. The finally runs on normal
            # exhaustion, on upstream errors, and on cancellation alike; the
            # BackgroundTask below is a harmless second aclose for the
            # successful path.
            try:
                async for chunk in _forward_and_account(upstream, usage):
                    yield chunk
            finally:
                await _cleanup()

        return StreamingResponse(
            _stream_and_cleanup(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            background=BackgroundTask(_cleanup),
        )


# SSE events end at a blank line; the spec allows LF, CRLF, or CR line endings.
_SSE_EVENT_END = re.compile(rb"\r\n\r\n|\n\n|\r\r")


async def _forward_and_account(upstream: httpx.Response, usage: LLMProxyUsage) -> AsyncIterator[bytes]:
    """Yield upstream bytes while parsing SSE events for token usage.

    ``aiter_bytes`` (not ``aiter_raw``): the upstream may compress the
    stream, and the forwarded response carries no Content-Encoding header,
    so the bytes must be decoded here — for the client and for the SSE
    accounting alike. An uncompressed stream passes through byte-identical.
    Accounting happens on a copy of the stream; nothing it does can change
    what the client receives.
    """
    buffer = b""
    async for chunk in upstream.aiter_bytes():
        yield chunk
        buffer += chunk
        while True:
            end = _SSE_EVENT_END.search(buffer)
            if end is None:
                break
            event, buffer = buffer[: end.start()], buffer[end.end() :]
            usage.record_sse_event(event)
