import json

from speech_to_speech.setup.endpoints import (
    ProbeResponse,
    discover_endpoints,
    parse_listening_ports,
    validate_selected_endpoint,
)


def test_lsof_parser_finds_arbitrary_loopback_ports_only():
    output = "\n".join(["p10", "n127.0.0.1:8080", "p11", "n[::1]:9090", "p12", "n*:3000", "p13", "n10.0.0.2:4000"])

    assert parse_listening_ports(output) == [3000, 8080, 9090]


def test_discovery_is_engine_independent_read_only_and_classifies_routes():
    calls = []

    def request(method, url, headers, timeout):
        calls.append((method, url))
        if url.endswith("/v1/models"):
            return ProbeResponse(200, json.dumps({"data": [{"id": "local-model"}]}))
        if url.endswith("/v1/chat/completions") or url.endswith("/v1/audio/transcriptions"):
            return ProbeResponse(204, "")
        return ProbeResponse(404, "")

    candidates = discover_endpoints(ports=[49152], request=request)

    assert len(candidates) == 1
    assert candidates[0].base_url == "http://127.0.0.1:49152/v1"
    assert candidates[0].models == ("local-model",)
    assert candidates[0].capabilities.chat_completions is True
    assert candidates[0].capabilities.transcriptions is True
    assert all(method in {"GET", "HEAD", "OPTIONS"} for method, _ in calls)


def test_discovery_keeps_auth_endpoint_without_leaking_or_posting():
    calls = []

    def request(method, url, headers, timeout):
        calls.append((method, headers))
        return ProbeResponse(401, '{"error":"key required"}')

    candidates = discover_endpoints(ports=[1234], request=request)

    assert len(candidates) == 1
    assert candidates[0].requires_auth is True
    assert all(method != "POST" for method, _ in calls)
    assert all("Authorization" not in headers for _, headers in calls)


def test_discovery_detects_route_auth_when_models_are_public():
    def request(method, url, headers, timeout):
        if url.endswith("/models"):
            return ProbeResponse(200, '{"data": [{"id": "local-model"}]}')
        return ProbeResponse(403, "")

    candidate = discover_endpoints(ports=[1234], request=request)[0]

    assert candidate.requires_auth is True


def test_discovery_ignores_malformed_and_unreachable_services():
    def request(method, url, headers, timeout):
        if ":1111/" in url:
            return ProbeResponse(200, "not json")
        raise TimeoutError("timed out")

    assert discover_endpoints(ports=[1111, 2222], request=request) == []


def test_selected_endpoint_validation_is_the_only_place_that_posts():
    calls = []

    def request(method, url, headers, timeout):
        calls.append((method, url, headers))
        return ProbeResponse(200, "{}")

    result = validate_selected_endpoint(
        "http://127.0.0.1:8080/v1", stage="llm", model="gemma", api_key="top-secret", request=request
    )

    assert result is True
    assert calls == [
        (
            "POST",
            "http://127.0.0.1:8080/v1/chat/completions",
            {"Authorization": "Bearer top-secret", "Content-Type": "application/json"},
        )
    ]
