"""Session-gated LLM proxy: OpenAI-compatible passthrough for remote LLM backends.

Mounts ``POST /v1/chat/completions`` (backend ``chat-completions``) or
``POST /v1/responses`` (backend ``responses-api``) as a passthrough to the
configured upstream provider. The ticket to use it is an open realtime
session: the client presents its realtime session id as the bearer token,
which is validated against the pipeline pool at request start and used for
nothing else. Requests never touch pipeline units' queues or cancel scopes,
so proxied generations run fully concurrent with the speech pipeline and are
never interrupted by new speech.
"""

import logging

import httpx
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit

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


def _bearer_session_id(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _session_is_active(pool: list[PipelineUnit], session_id: str) -> bool:
    for unit in pool:
        session = unit.session
        if session is not None and session.session_id == session_id and session.released_at is None:
            return True
    return False


def mount_llm_proxy(app: FastAPI, pool: list[PipelineUnit], config: LLMProxyConfig | None) -> None:
    """Mount the proxy paths on *app* according to *config*.

    Both known paths always answer: the path matching an enabled remote
    backend proxies, every other combination answers 501 naming the reason,
    so a misconfigured deployment is diagnosed in one request.
    """
    config = config or LLMProxyConfig()

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
        return

    assert config.llm_backend is not None
    serving_path = _PATHS[config.llm_backend]
    for path in _PATHS.values():
        if path == serving_path:
            _mount_passthrough(app, path, pool, config)
        else:
            _mount_unavailable(
                app,
                path,
                f"This server runs the '{config.llm_backend}' backend; use {serving_path} instead.",
            )


def _mount_unavailable(app: FastAPI, path: str, reason: str) -> None:
    @app.post(path)
    async def unavailable_endpoint() -> Response:
        return _error_response(501, reason, "not_implemented")


def _mount_passthrough(app: FastAPI, path: str, pool: list[PipelineUnit], config: LLMProxyConfig) -> None:
    base_url = (config.upstream_base_url or DEFAULT_UPSTREAM_BASE_URL).rstrip("/")
    upstream_url = base_url + path.removeprefix("/v1")

    @app.post(path)
    async def passthrough_endpoint(request: Request) -> Response:
        session_id = _bearer_session_id(request)
        if session_id is None or not _session_is_active(pool, session_id):
            return _error_response(
                401,
                "Invalid bearer token: pass the realtime session id of an open session "
                "(from the session.created event) as the API key.",
                "invalid_session",
            )

        try:
            body = await request.json()
        except Exception:
            return _error_response(400, "Request body must be valid JSON.", "invalid_request_error")
        body["model"] = config.model_name
        if path == _PATHS["responses-api"]:
            # Session holders are anonymous; forcing store off keeps them from
            # creating persistent state on the operator's provider account.
            body["store"] = False

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

        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            background=BackgroundTask(_cleanup),
        )
