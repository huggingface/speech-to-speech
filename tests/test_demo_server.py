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
