import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))
demo_auth = importlib.import_module("auth")
demo_server = importlib.import_module("server")


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


def test_load_balancer_headers_forward_signed_in_user_token(monkeypatch):
    monkeypatch.setattr(
        demo_server.auth,
        "current_access_token",
        lambda request: "hf_user_token",
    )

    assert demo_server._load_balancer_headers(object()) == {
        "Content-Type": "application/json",
        "User-Agent": "speech-to-speech-demo",
        "X-Reachy-Mini-Authorization": "Bearer hf_user_token",
    }


def test_load_balancer_headers_keep_anonymous_requests_credential_free(monkeypatch):
    monkeypatch.setattr(demo_server.auth, "current_access_token", lambda request: None)

    headers = demo_server._load_balancer_headers(object())

    assert headers == {
        "Content-Type": "application/json",
        "User-Agent": "speech-to-speech-demo",
    }


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
