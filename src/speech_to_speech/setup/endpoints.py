from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ProbeResponse:
    status_code: int
    text: str


@dataclass(frozen=True)
class EndpointCapabilities:
    chat_completions: bool = False
    responses: bool = False
    transcriptions: bool = False
    speech: bool = False


@dataclass(frozen=True)
class EndpointCandidate:
    base_url: str
    models: tuple[str, ...] = ()
    capabilities: EndpointCapabilities = EndpointCapabilities()
    requires_auth: bool = False


Request = Callable[[str, str, dict[str, str], float], ProbeResponse]


def parse_listening_ports(output: str) -> list[int]:
    ports: set[int] = set()
    for line in output.splitlines():
        if not line.startswith("n"):
            continue
        address = line[1:]
        if not (address.startswith("127.") or address.startswith("[::1]:") or address.startswith("localhost:")):
            continue
        try:
            ports.add(int(address.rsplit(":", 1)[1].split(" ", 1)[0]))
        except (IndexError, ValueError):
            continue
    return sorted(ports)


def listening_loopback_ports() -> list[int]:
    result = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpn"],
        capture_output=True,
        text=True,
        check=False,
    )
    return parse_listening_ports(result.stdout)


def _http_request(method: str, url: str, headers: dict[str, str], timeout: float) -> ProbeResponse:
    kwargs: dict[str, Any] = {"headers": headers, "timeout": timeout, "follow_redirects": False}
    response = httpx.request(method, url, **kwargs)
    return ProbeResponse(response.status_code, response.text)


def _minimal_payload(url: str, model: str = "") -> dict[str, Any]:
    if url.endswith("/chat/completions"):
        return {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 1}
    if url.endswith("/responses"):
        return {"model": model, "input": "Reply with OK.", "max_output_tokens": 1}
    if url.endswith("/audio/speech"):
        return {"model": model, "input": "test", "voice": "default"}
    return {"model": model}


def _route_available(status_code: int) -> bool:
    return status_code not in {404, 405} and status_code < 500


def _probe_port(port: int, request: Request, timeout: float) -> EndpointCandidate | None:
    root = f"http://127.0.0.1:{port}/v1"
    try:
        models_response = request("GET", f"{root}/models", {}, timeout)
    except Exception:
        return None
    if models_response.status_code in {401, 403}:
        return EndpointCandidate(root, requires_auth=True)
    if models_response.status_code != 200:
        return None
    try:
        payload = json.loads(models_response.text)
        models = tuple(item["id"] for item in payload["data"] if isinstance(item.get("id"), str))
    except (KeyError, TypeError, json.JSONDecodeError):
        return None

    routes = {
        "chat_completions": "chat/completions",
        "responses": "responses",
        "transcriptions": "audio/transcriptions",
        "speech": "audio/speech",
    }
    available: dict[str, bool] = {}
    for capability, route in routes.items():
        try:
            response = request("OPTIONS", f"{root}/{route}", {}, timeout)
            available[capability] = _route_available(response.status_code)
        except Exception:
            available[capability] = False
    return EndpointCandidate(root, models, EndpointCapabilities(**available))


def discover_endpoints(
    *,
    ports: Iterable[int] | None = None,
    request: Request = _http_request,
    timeout: float = 0.35,
    max_workers: int = 16,
) -> list[EndpointCandidate]:
    targets = sorted(set(listening_loopback_ports() if ports is None else ports))
    if not targets:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as executor:
        candidates = executor.map(lambda port: _probe_port(port, request, timeout), targets)
    return [candidate for candidate in candidates if candidate is not None]


def validate_selected_endpoint(
    base_url: str,
    *,
    stage: str,
    model: str,
    api_key: str | None = None,
    request: Request = _http_request,
    timeout: float = 10.0,
) -> bool:
    routes = {"llm": "chat/completions", "stt": "audio/transcriptions", "tts": "audio/speech"}
    if stage not in routes:
        raise ValueError(f"Unsupported endpoint stage: {stage}")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}", **headers}
    url = f"{base_url.rstrip('/')}/{routes[stage]}"
    if request is _http_request:
        http_response = httpx.post(
            url,
            headers=headers,
            json=_minimal_payload(url, model),
            timeout=timeout,
            follow_redirects=False,
        )
        response = ProbeResponse(http_response.status_code, http_response.text)
    else:
        response = request("POST", url, headers, timeout)
    return 200 <= response.status_code < 300
