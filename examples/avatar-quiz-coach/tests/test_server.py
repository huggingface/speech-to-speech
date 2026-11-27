"""Sign-in on a Space: who may open a voice session, and with whose token."""

import importlib
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOAD_BALANCER_URL", "https://lb.invalid")
    monkeypatch.delenv("SPEECH_TO_SPEECH_URL", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    mod = importlib.reload(importlib.import_module("server"))
    monkeypatch.setattr(mod, "OAUTH", True)
    return mod


def visitor(expires_in_s=3600):
    return SimpleNamespace(
        user_info=SimpleNamespace(preferred_username="ada", picture="https://example.invalid/ada.png"),
        access_token="hf_oauth_ada",
        access_token_expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in_s),
    )


def test_anonymous_visitor_must_sign_in_before_the_load_balancer_is_asked(server, monkeypatch):
    def no_upstream(*a, **k):
        raise AssertionError("the load balancer must not be called")

    monkeypatch.setattr(server.httpx, "AsyncClient", no_upstream)
    client = TestClient(server.app)
    resp = client.post("/api/session")
    assert resp.status_code == 401
    assert resp.json() == {"reason": "login_required", "loginUrl": "oauth/huggingface/login"}
    cfg = client.get("/api/config").json()
    assert cfg["loginRequired"] and cfg["user"] is None and cfg["loginUrl"]


def test_signed_in_visitor_opens_the_session_with_their_own_token(server, monkeypatch):
    monkeypatch.setattr(server, "_oauth", lambda request: visitor())
    req = SimpleNamespace(headers={})
    headers = server._upstream_headers(req, {})
    assert headers["Authorization"] == "Bearer hf_oauth_ada"
    assert headers["X-Reachy-Mini-Authorization"] == "Bearer hf_oauth_ada"
    cfg = TestClient(server.app).get("/api/config").json()
    assert cfg["user"] == "ada" and cfg["avatar"].endswith("ada.png") and cfg["logoutUrl"]


def test_expired_sign_in_counts_as_signed_out(server, monkeypatch):
    monkeypatch.setattr(server, "_oauth", lambda request: visitor(expires_in_s=10))
    resp = TestClient(server.app).post("/api/session")
    assert resp.status_code == 401
    assert server._username(SimpleNamespace()) is None


def test_lb_user_token_is_ignored_on_a_space(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LB_USER_TOKEN", "hf_developer")
    monkeypatch.setenv("SPACE_ID", "someone/viva")
    mod = importlib.reload(importlib.import_module("server"))
    assert mod.LB_USER_TOKEN == ""
    monkeypatch.delenv("SPACE_ID")
    assert importlib.reload(mod).LB_USER_TOKEN == "hf_developer"
