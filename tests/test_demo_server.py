import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from http.cookies import SimpleCookie
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))
demo_auth = importlib.import_module("auth")
demo_server = importlib.import_module("server")


@pytest.mark.parametrize("filename", list(demo_server.LOCAL_VAD_ASSETS))
@pytest.mark.parametrize("bundled", [False, True])
async def test_local_vad_assets_are_served_with_browser_content_types(monkeypatch, tmp_path, filename, bundled):
    monkeypatch.setattr(demo_server, "HERE", str(tmp_path))
    package, media_type = demo_server.LOCAL_VAD_ASSETS[filename]
    installed = tmp_path / "node_modules" / package / "dist" / filename
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_bytes(b"installed asset")
    if bundled:
        destination = tmp_path / "vendor" / "local-vad" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bundled asset")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=demo_server.app), base_url="http://test") as client:
        response = await client.get(f"/vendor/local-vad/{filename}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.content == (b"bundled asset" if bundled else b"installed asset")


async def test_local_vad_missing_assets_fail_without_exposing_other_package_files(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_server, "HERE", str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=demo_server.app), base_url="http://test") as client:
        assert (await client.get("/vendor/local-vad/not-an-asset.js")).status_code == 404
        response = await client.get("/vendor/local-vad/silero_vad_v6.onnx")
    assert response.status_code == 503
    assert response.json()["detail"] == "Run npm ci in demo/"


def _mock_whoami(monkeypatch, payload):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(demo_auth, "_whoami_cache", {})
    return calls


def test_resolve_tier_prefers_oauth_pro_without_hub_lookup(monkeypatch):
    def unexpected_get(*args, **kwargs):
        pytest.fail("OAuth PRO users must not require a whoami-v2 lookup")

    monkeypatch.setattr(httpx, "get", unexpected_get)

    assert demo_auth.resolve_tier({"is_pro": True}, "hf_user_token") == "pro"


@pytest.mark.parametrize("oauth_is_pro", [None, False])
def test_user_view_uses_token_backed_pro_status(monkeypatch, oauth_is_pro):
    calls = _mock_whoami(
        monkeypatch,
        {"name": "alice", "isPro": True, "orgs": [], "private": "server-only"},
    )
    monkeypatch.setattr(
        demo_auth,
        "current_oauth",
        lambda request: {
            "access_token": "hf_user_token",
            "user_info": {
                "sub": "123",
                "preferred_username": "alice",
                "is_pro": oauth_is_pro,
            },
        },
    )

    assert demo_auth.user_view(object()) == {
        "loggedIn": True,
        "username": "alice",
        "avatar": None,
        "tier": "pro",
    }
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("whoami", "expected"),
    [
        ({"isPro": False, "orgs": []}, "free"),
        ({"isPro": False, "orgs": [{"name": "smolagents"}]}, "org"),
    ],
)
def test_token_backed_non_pro_tiers_remain_unchanged(monkeypatch, whoami, expected):
    _mock_whoami(monkeypatch, whoami)

    assert demo_auth.resolve_tier({"is_pro": False}, "hf_user_token") == expected


def test_failed_whoami_falls_back_without_blocking_retry(monkeypatch):
    monkeypatch.setattr(demo_auth, "_whoami_cache", {})
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise httpx.ConnectError("Hub unavailable")
        return httpx.Response(
            200,
            json={"isPro": True, "orgs": []},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)

    assert demo_auth.resolve_tier({"is_pro": False}, "hf_user_token") == "free"
    assert demo_auth._whoami_cache == {}
    assert demo_auth.resolve_tier({"is_pro": False}, "hf_user_token") == "pro"
    assert demo_auth.resolve_tier({"is_pro": False}, "hf_user_token") == "pro"
    assert len(calls) == 2


async def test_me_reuses_whoami_resolution_and_retries_next_request(monkeypatch):
    monkeypatch.setattr(demo_server, "LIMITER_ENABLED", True)
    monkeypatch.setattr(demo_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(demo_auth, "_whoami_cache", {})
    monkeypatch.setattr(demo_server.limiter, "remaining", lambda keys, tier: 600)
    monkeypatch.setattr(
        demo_auth,
        "current_oauth",
        lambda request: {
            "access_token": "hf_user_token",
            "user_info": {"sub": "123", "preferred_username": "alice"},
        },
    )
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise httpx.ConnectError("Hub unavailable")
        return httpx.Response(
            200,
            json={"isPro": True, "orgs": []},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)

    first = json.loads((await demo_server.me(object())).body)

    assert first["tier"] == "free"
    assert first["remainingSec"] == 600
    assert first["limitSec"] == 600
    assert demo_auth._whoami_cache == {}
    assert len(calls) == 1

    second = json.loads((await demo_server.me(object())).body)

    assert second["tier"] == "pro"
    assert second["remainingSec"] is None
    assert second["limitSec"] is None
    assert len(calls) == 2


def test_tier_and_org_resolution_share_one_whoami_request(monkeypatch):
    calls = _mock_whoami(
        monkeypatch,
        {"isPro": False, "orgs": [{"name": "smolagents"}]},
    )
    user = {"is_pro": False}

    assert demo_auth.resolve_tier(user, "hf_user_token") == "org"
    assert demo_auth._org_names(user, "hf_user_token") == {"smolagents"}
    assert len(calls) == 1


def test_current_access_token_normalizes_oauth_token(monkeypatch):
    monkeypatch.setattr(
        demo_auth,
        "current_oauth",
        lambda request: {"access_token": "  hf_user_token  "},
    )

    assert demo_auth.current_access_token(object()) == "hf_user_token"


def test_current_access_token_rejects_missing_or_empty_token(monkeypatch):
    monkeypatch.setattr(demo_auth, "current_oauth", lambda request: None)
    assert demo_auth.current_access_token(object()) is None

    monkeypatch.setattr(demo_auth, "current_oauth", lambda request: {"access_token": "  "})
    assert demo_auth.current_access_token(object()) is None


def test_expiring_oauth_token_requires_fresh_login(monkeypatch):
    monkeypatch.setattr(
        demo_auth,
        "current_oauth",
        lambda request: {
            "access_token": "hf_user_token",
            "access_token_expires_at": datetime.now() + timedelta(seconds=15),
            "user_info": {"sub": "123", "preferred_username": "alice"},
        },
    )

    assert demo_auth.current_access_token(object()) is None
    assert demo_auth.oauth_login_required_reason(object()) == "token_expired"
    assert demo_auth.user_view(object()) == {
        "loggedIn": False,
        "tier": "anon",
        "reason": "token_expired",
    }


def test_load_balancer_headers_use_user_token_for_both_hops_without_service_token(monkeypatch):
    monkeypatch.setattr(demo_server, "LB_HF_TOKEN", "")
    monkeypatch.setattr(
        demo_server.auth,
        "current_access_token",
        lambda request: "hf_user_token",
    )

    assert demo_server._load_balancer_headers(object()) == {
        "Authorization": "Bearer hf_user_token",
        "Content-Type": "application/json",
        "User-Agent": "speech-to-speech-demo",
        "X-Reachy-Mini-Authorization": "Bearer hf_user_token",
    }


def test_load_balancer_headers_keep_service_and_user_tokens_separate(monkeypatch):
    monkeypatch.setattr(demo_server, "LB_HF_TOKEN", "hf_service_token")
    monkeypatch.setattr(
        demo_server.auth,
        "current_access_token",
        lambda request: "hf_user_token",
    )

    assert demo_server._load_balancer_headers(object()) == {
        "Authorization": "Bearer hf_service_token",
        "Content-Type": "application/json",
        "User-Agent": "speech-to-speech-demo",
        "X-Reachy-Mini-Authorization": "Bearer hf_user_token",
    }


def test_load_balancer_headers_keep_anonymous_requests_credential_free(monkeypatch):
    monkeypatch.setattr(demo_server, "LB_HF_TOKEN", "")
    monkeypatch.setattr(demo_server.auth, "current_access_token", lambda request: None)

    headers = demo_server._load_balancer_headers(object())

    assert headers == {
        "Content-Type": "application/json",
        "User-Agent": "speech-to-speech-demo",
    }


def test_load_balancer_headers_authenticate_anonymous_request_with_service_token(monkeypatch):
    monkeypatch.setattr(demo_server, "LB_HF_TOKEN", "hf_service_token")
    monkeypatch.setattr(demo_server.auth, "current_access_token", lambda request: None)

    assert demo_server._load_balancer_headers(object()) == {
        "Authorization": "Bearer hf_service_token",
        "Content-Type": "application/json",
        "User-Agent": "speech-to-speech-demo",
    }


def test_load_balancer_failure_logs_safe_diagnostic_headers(caplog):
    response = httpx.Response(
        429,
        headers={
            "Authorization": "Bearer secret",
            "Content-Type": "text/html",
            "RateLimit": '"api";r=0;t=42',
            "Retry-After": "42",
            "X-Request-Id": "request-id",
        },
        text="<html>\n  rate limited\n</html>",
    )

    with caplog.at_level("WARNING", logger="s2s.search"):
        demo_server._log_load_balancer_failure("session handshake", response)

    assert "request-id" in caplog.text
    assert "retry-after" in caplog.text
    assert "<html> rate limited </html>" in caplog.text
    assert "Bearer secret" not in caplog.text


async def test_session_preserves_load_balancer_authentication_failure(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return httpx.Response(401, json={"reason": "token_invalid"})

    monkeypatch.setattr(demo_server, "LOAD_BALANCER_URL", "https://load-balancer.example")
    monkeypatch.setattr(demo_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(demo_server, "LIMITER_ENABLED", False)
    monkeypatch.setattr(demo_server.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(demo_server.auth, "oauth_login_required_reason", lambda request: None)
    monkeypatch.setattr(demo_server.auth, "resolve_identity", lambda request: ("free", ["key"], None))
    monkeypatch.setattr(demo_server.auth, "current_access_token", lambda request: "hf_user_token")

    request = SimpleNamespace(scope={"session": {"oauth_info": {"access_token": "hf_user_token"}}})
    response = await demo_server.session(request)

    assert response.status_code == 401
    assert json.loads(response.body) == {
        "reason": "token_invalid",
        "loginUrl": demo_auth.OAUTH_LOGIN_PATH,
    }
    assert "oauth_info" not in request.scope["session"]


async def test_session_rejects_locally_expired_oauth_before_proxying(monkeypatch):
    monkeypatch.setattr(demo_server, "LOAD_BALANCER_URL", "https://load-balancer.example")
    monkeypatch.setattr(demo_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(
        demo_server.auth,
        "oauth_login_required_reason",
        lambda request: "token_expired",
    )

    response = await demo_server.session(object())

    assert response.status_code == 401
    assert json.loads(response.body) == {
        "reason": "token_expired",
        "loginUrl": demo_auth.OAUTH_LOGIN_PATH,
    }


async def test_queue_rejects_locally_expired_oauth_before_polling(monkeypatch):
    monkeypatch.setattr(demo_server, "LOAD_BALANCER_URL", "https://load-balancer.example")
    monkeypatch.setattr(demo_server, "AUTH_ENABLED", True)
    monkeypatch.setattr(
        demo_server.auth,
        "oauth_login_required_reason",
        lambda request: "token_expired",
    )
    monkeypatch.setattr(
        demo_server.auth,
        "resolve_identity",
        lambda request: pytest.fail("expired OAuth must not resolve as anonymous"),
    )

    response = await demo_server.queue_status("ticket", object())

    assert response.status_code == 401
    assert json.loads(response.body) == {
        "reason": "token_expired",
        "loginUrl": demo_auth.OAUTH_LOGIN_PATH,
    }


def test_websocket_client_classifies_session_401_as_login_required():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = """
globalThis.localStorage = { getItem() { return null; } };
globalThis.fetch = async () => ({
  status: 401,
  ok: false,
  async json() {
    return { reason: "token_invalid", loginUrl: "/oauth/huggingface/login" };
  },
});
const { S2sRealtimeClient } = await import("./demo/s2s-realtime-client.js");
const client = new S2sRealtimeClient({
  transport: "websocket",
  voice: "Aiden",
  instructions: "Be helpful.",
  sessionUrl: "api/session",
});
try {
  await client._postSession();
  throw new Error("expected the session request to fail");
} catch (error) {
  if (error.code !== "login-required") throw error;
  if (error.loginUrl !== "/oauth/huggingface/login") {
    throw new Error(`unexpected login URL: ${error.loginUrl}`);
  }
}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_login_required_state_wins_over_inflight_account_refresh():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for demo client tests")

    script = """
globalThis.localStorage = { getItem() { return null; } };
const elements = new Map();
function element() {
  return {
    hidden: false,
    innerHTML: "",
    textContent: "",
    href: "",
    open: false,
    addEventListener() {},
    contains() { return false; },
    setAttribute() {},
    showModal() { this.open = true; },
    close() { this.open = false; },
  };
}
globalThis.document = {
  querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, element());
    return elements.get(selector);
  },
  addEventListener() {},
  getElementById(id) { return this.querySelector(`#${id}`); },
};

let resolveFetch;
globalThis.fetch = () => new Promise((resolve) => { resolveFetch = resolve; });

const { Account } = await import("./demo/ui/account.js");
const account = new Account();
const refresh = account.refresh();
account.showLoginRequired("/oauth/huggingface/login");
resolveFetch({
  ok: true,
  async json() {
    return {
      enabled: true,
      auth: true,
      loggedIn: true,
      username: "alice",
      tier: "free",
      loginUrl: "/oauth/huggingface/login",
    };
  },
});
await refresh;

if (account.tier !== "anon") throw new Error(`unexpected tier: ${account.tier}`);
const root = elements.get("#account");
if (!root.innerHTML.includes("signin-pill")) {
  throw new Error(`stale account refresh won: ${root.innerHTML}`);
}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("metered,service_token", [(False, "hf_service_token"), (True, "")])
def test_guest_service_auth_requires_metering_and_service_credential(monkeypatch, metered, service_token):
    monkeypatch.setattr(demo_server, "ALLOW_GUEST_SESSIONS", True)
    monkeypatch.setattr(demo_server, "LIMITER_ENABLED", metered)
    monkeypatch.setattr(demo_server, "LB_HF_TOKEN", service_token)
    monkeypatch.setattr(demo_server.auth, "current_access_token", lambda request: None)

    assert "X-Reachy-Mini-Authorization" not in demo_server._load_balancer_headers(object())


def test_guest_opt_in_preserves_signed_in_visitor_attribution(monkeypatch):
    monkeypatch.setattr(demo_server, "ALLOW_GUEST_SESSIONS", True)
    monkeypatch.setattr(demo_server, "LIMITER_ENABLED", True)
    monkeypatch.setattr(demo_server, "LB_HF_TOKEN", "hf_service_token")
    monkeypatch.setattr(demo_server.auth, "current_access_token", lambda request: "hf_visitor_token")

    headers = demo_server._load_balancer_headers(object())

    assert headers["Authorization"] == "Bearer hf_service_token"
    assert headers["X-Reachy-Mini-Authorization"] == "Bearer hf_visitor_token"


def _guest_request(cookie=None, payload=None):
    headers = [(b"cookie", f"s2s_anon={cookie}".encode())] if cookie else []

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload or {}).encode(), "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/api/session",
            "headers": headers,
            "client": ("203.0.113.9", 1234),
            "server": ("demo.example", 443),
            "query_string": b"",
        },
        receive,
    )


@pytest.fixture
def guest_demo(monkeypatch, tmp_path):
    limiter = demo_server.limiter
    monkeypatch.setattr(demo_server, "ALLOW_GUEST_SESSIONS", True)
    monkeypatch.setattr(demo_server, "LIMITER_ENABLED", True)
    monkeypatch.setattr(demo_server, "LB_HF_TOKEN", "hf_guest_service_secret")
    monkeypatch.setattr(demo_server, "LOAD_BALANCER_URL", "https://allocator.example")
    monkeypatch.setattr(demo_server.auth, "current_oauth", lambda request: None)
    monkeypatch.setattr(limiter, "_resolve_db_path", lambda: tmp_path / "usage.sqlite3")
    monkeypatch.setattr(limiter, "_db_path", None)
    monkeypatch.setattr(limiter, "_HASH_SECRET", b"test-cookie-signing-secret")
    monkeypatch.setattr(limiter, "ANON_SEC", 300)
    monkeypatch.setattr(limiter, "CHUNK_SEC", 10)
    clock = {"now": 1000.0}
    monkeypatch.setattr(limiter.time, "time", lambda: clock["now"])
    limiter.init()
    calls = []

    def allocator(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer hf_guest_service_secret"
        assert request.headers["X-Reachy-Mini-Authorization"] == "Bearer hf_guest_service_secret"
        return httpx.Response(
            200,
            json={
                "state": "granted",
                "session_id": f"guest-{len(calls)}",
                "connect_url": "wss://worker.example/v1/realtime?token=temporary",
            },
        )

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        demo_server.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(allocator), **kwargs),
    )
    return clock, calls


async def test_guest_gets_five_minutes_then_heartbeat_expires_and_reconnect_is_denied(guest_demo):
    clock, calls = guest_demo
    first = await demo_server.session(_guest_request())
    data = json.loads(first.body)
    assert first.status_code == 200
    assert data["tier"] == "anon" and data["limited"] is True
    assert data["remainingSec"] == 290
    assert demo_server.limiter.budget_for(data["tier"]) == 300
    assert b"hf_guest_service_secret" not in first.body
    cookie = SimpleCookie(first.headers["set-cookie"])["s2s_anon"].value

    for elapsed in range(5, 300, 5):
        clock["now"] = 1000.0 + elapsed
        assert await demo_server.session_heartbeat(_guest_request(cookie, {"sessionId": data["session_id"]})) == {
            "expired": False
        }
    clock["now"] = 1300.0
    assert await demo_server.session_heartbeat(_guest_request(cookie, {"sessionId": data["session_id"]})) == {
        "expired": True
    }
    assert (await demo_server.session(_guest_request(cookie))).status_code == 402
    assert (await demo_server.session(_guest_request())).status_code == 402  # Cookie deletion keeps the IP budget.
    assert len(calls) == 1


async def test_guest_queue_claim_uses_same_service_auth_and_reserves_guest_budget(guest_demo):
    _, calls = guest_demo
    response = await demo_server.queue_status("ticket", _guest_request())

    assert response.status_code == 200
    assert json.loads(response.body)["limited"] is True
    assert json.loads(response.body)["tier"] == "anon"
    assert json.loads(response.body)["remainingSec"] == 290
    assert calls[0].url.path == "/queue/ticket"
    assert b"hf_guest_service_secret" not in response.body


def test_guest_reservation_never_exceeds_a_partially_spent_budget(guest_demo):
    clock, _ = guest_demo
    limiter = demo_server.limiter
    keys = ["ip-budget", "cookie-budget"]
    with limiter._lock, limiter._connect() as con:
        limiter._add(con, keys, limiter._today(), 287)
    assert limiter.begin("last-thirteen-seconds", keys, "anon") == 10
    clock["now"] += 12
    assert limiter.heartbeat("last-thirteen-seconds") is True
    clock["now"] += 1
    assert limiter.heartbeat("last-thirteen-seconds") is False
    assert limiter.remaining(keys, "anon") == 0
    with limiter._lock, limiter._connect() as con:
        assert limiter._spent_max(con, keys, limiter._today()) == 300
