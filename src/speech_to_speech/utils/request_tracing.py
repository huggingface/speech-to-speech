"""Bounded technical correlation for remote speech requests, without content."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from uuid import uuid4

logger = logging.getLogger(__name__)
REQUEST_ID_HEADER = "X-Speech-Request-Id"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}", re.ASCII)


def safe_identifier(value: object) -> str | None:
    """Omit invalid identifiers rather than logging or silently rewriting them."""
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None else None


def _revision(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**31 - 1 else None


def _record(event: str, **fields: object) -> None:
    logger.info("%s", json.dumps({"event": event, **fields}, ensure_ascii=True, separators=(",", ":")))


@dataclass(frozen=True)
class HttpRequestContext:
    """A scalar snapshot; no shared client defaults or thread-local context."""

    stage: Literal["stt", "llm", "tts"]
    session_id: str | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    response_key: str | None = None
    mode: Literal["final", "progressive"] | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("stt", "llm", "tts") or self.mode not in (None, "final", "progressive"):
            raise ValueError("invalid HTTP trace stage or mode")
        for name in ("session_id", "turn_id", "response_key"):
            object.__setattr__(self, name, safe_identifier(getattr(self, name)))
        object.__setattr__(self, "turn_revision", _revision(self.turn_revision))

    def accepted_stt_result(self, request_ids: Iterable[str]) -> None:
        """Associate only accepted final results, including reused window IDs."""
        if self.stage != "stt" or self.mode == "progressive":
            return
        # The source is already-retained window data, not a new request history.
        for request_id in dict.fromkeys(request_ids):
            if safe_identifier(request_id) is not None:
                _record("speech_http_stt_result", request_id=request_id, **asdict(self))


@dataclass(frozen=True)
class HttpRequestTrace:
    """One logical operation; SDK/proxy retries may reuse its diagnostic ID."""

    context: HttpRequestContext
    request_id: str = field(default_factory=lambda: uuid4().hex, init=False)

    def headers(self) -> dict[str, str]:
        return {REQUEST_ID_HEADER: self.request_id}

    def dispatch(self) -> None:
        # Dispatch/selection never claims that a worker received or finished work.
        _record("speech_http_dispatch", request_id=self.request_id, **asdict(self.context))

    def request_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Copy per-call options while preserving existing authentication headers."""
        copied = dict(options)
        supplied = copied.get("extra_headers")
        if supplied is not None and not isinstance(supplied, Mapping):
            raise TypeError("extra_headers must be a mapping")
        headers = {key: value for key, value in (supplied or {}).items() if key.lower() != REQUEST_ID_HEADER.lower()}
        headers.update(self.headers())
        copied["extra_headers"] = headers
        return copied


def log_response_binding(
    *,
    session_id: str,
    response_key: str,
    turn_id: str | None,
    turn_revision: int | None,
    source: Literal["speech", "client", "tool"],
) -> None:
    _record(
        "speech_response_binding",
        session_id=safe_identifier(session_id),
        response_key=safe_identifier(response_key),
        turn_id=safe_identifier(turn_id),
        turn_revision=_revision(turn_revision),
        source=source,
    )
