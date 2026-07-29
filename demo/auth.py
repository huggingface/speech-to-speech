"""
HF OAuth + per-request identity for the duration limiter.

Login uses Hugging Face's native Spaces OAuth via `huggingface_hub`
(`attach_huggingface_oauth` / `parse_huggingface_oauth`). The OAuth env
(`OAUTH_CLIENT_ID`, ...) is injected by the platform when the Space README sets
`hf_oauth: true`, so this only activates on a deployed Space — locally and in
direct mode there's no OAuth and the limiter treats everyone as anonymous.

Identity:
  - signed in   -> tier 'pro' | 'free', keyed by hashed HF `sub`
  - anonymous   -> tier 'anon', keyed by BOTH hashed client IP and a hashed
                   signed-cookie id (OR-matched in the limiter)
"""

import logging
import os
import secrets

import limiter

logger = logging.getLogger("s2s.auth")

# huggingface_hub adds these routes when OAuth is attached. Centralised so a
# version change is a one-line fix; the paths are handed to the client via
# /api/me rather than hardcoded there.
OAUTH_LOGIN_PATH = "/oauth/huggingface/login"
OAUTH_LOGOUT_PATH = "/oauth/huggingface/logout"

ANON_COOKIE = "s2s_anon"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


# Members of these orgs get unlimited usage (like PRO) out of the box. The
# UNLIMITED_ORGS env adds to this set; it doesn't replace it.
_DEFAULT_UNLIMITED_ORGS = {"cerebras", "huggingfacem4", "smolagents", "pollen-robotics"}


def _unlimited_orgs() -> "set[str]":
    """Org usernames whose members get unlimited usage (like PRO).

    Defaults to {cerebras, HuggingFaceM4, smolagents}; the UNLIMITED_ORGS env
    (comma/space-separated, e.g. `UNLIMITED_ORGS=my-team`) adds more. Matched
    case-insensitively against the signed-in user's organisations."""
    raw = os.environ.get("UNLIMITED_ORGS", "")
    extra = {o.strip().lower() for o in raw.replace(",", " ").split() if o.strip()}
    return _DEFAULT_UNLIMITED_ORGS | extra

try:
    from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth
    _OAUTH_IMPORTABLE = True
except Exception as exc:  # pragma: no cover - import guard
    logger.info("huggingface_hub OAuth unavailable (%s); sign-in disabled.", exc)
    _OAUTH_IMPORTABLE = False

# Set by attach(): True once OAuth is actually wired (importable + env present).
oauth_enabled = False


def attach(app) -> bool:
    """Wire HF OAuth onto the app if it's importable and configured. Returns
    whether sign-in is available."""
    global oauth_enabled
    if not _OAUTH_IMPORTABLE or not os.environ.get("OAUTH_CLIENT_ID"):
        return False
    try:
        attach_huggingface_oauth(app)
        oauth_enabled = True
        logger.info("HF OAuth attached (sign-in enabled).")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to attach HF OAuth: %r", exc)
        oauth_enabled = False
    return oauth_enabled


def _field(obj, name, default=None):
    """Read a field whether the user-info is an object or a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# Surface what we detect (orgs, tier) in logs and on /api/me when set. Handy for
# verifying org gating on the live Space without guessing.
AUTH_DEBUG = bool(os.environ.get("AUTH_DEBUG"))

# whoami-v2 org lookups are cached for the process lifetime, keyed by token, so
# /api/me + /api/session don't each hit the Hub.
_orgs_cache: "dict[str, set[str]]" = {}


def current_oauth(request):
    """The parsed HF OAuth info (user_info + access_token), or None."""
    if not oauth_enabled:
        return None
    try:
        return parse_huggingface_oauth(request)
    except Exception:
        return None


def current_user(request):
    """The signed-in HF user-info, or None."""
    return _field(current_oauth(request), "user_info")


def _user_org_names(user) -> "set[str]":
    """The user's organisations from the OAuth userinfo, by username/name/id."""
    names = set()
    for org in _field(user, "orgs", []) or []:
        for key in ("preferred_username", "name", "sub"):
            val = _field(org, key)
            if val:
                names.add(str(val).lower())
    return names


def _orgs_via_token(token: str) -> "set[str]":
    """Fallback org lookup via the Hub `whoami-v2` API, using the user's OAuth
    access token. Covers the case where the userinfo claim omits `orgs`."""
    if not token:
        return set()
    if token in _orgs_cache:
        return _orgs_cache[token]
    names: "set[str]" = set()
    try:
        import httpx

        resp = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        for org in resp.json().get("orgs", []) or []:
            for key in ("name", "fullname"):
                val = org.get(key)
                if val:
                    names.add(str(val).lower())
    except Exception as exc:  # pragma: no cover - network/permission dependent
        logger.info("whoami-v2 org lookup failed: %r", exc)
    _orgs_cache[token] = names
    return names


def _org_names(user, token=None, allow=None) -> "set[str]":
    """The user's org usernames from the OAuth userinfo claim. If that doesn't
    already satisfy `allow`, fall back to the Hub `whoami-v2` API (the claim is
    often empty or partial), so membership is resolved either way."""
    names = _user_org_names(user)
    if token and (allow is None or not (allow & names)):
        names = names | _orgs_via_token(token)
    return names


def resolve_tier(user, token=None) -> str:
    """Tier for a signed-in user: 'pro' (paying), 'org' (allow-listed org
    member, unlimited), or 'free'. PRO wins over org if both apply."""
    if bool(_field(user, "is_pro", False)):
        return "pro"
    allow = _unlimited_orgs()
    names = _org_names(user, token, allow)
    tier = "org" if (allow & names) else "free"
    if AUTH_DEBUG:
        logger.info("tier=%s orgs=%s allow=%s", tier, sorted(names), sorted(allow))
    return tier


def user_view(request) -> dict:
    """Public profile for /api/me."""
    info = current_oauth(request)
    user = _field(info, "user_info")
    if not user:
        return {"loggedIn": False, "tier": "anon"}
    token = _field(info, "access_token")
    out = {
        "loggedIn": True,
        "username": _field(user, "preferred_username") or _field(user, "name") or "you",
        "avatar": _field(user, "picture"),
        "tier": resolve_tier(user, token),
    }
    if AUTH_DEBUG:
        out["orgs"] = sorted(_org_names(user, token))
    return out


def _client_ip(request) -> str:
    """Real client IP. On HF the app sits behind a proxy, so the user's address
    is the first hop in X-Forwarded-For, not request.client.host."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def resolve_identity(request):
    """Resolve (tier, keys, set_cookie) for this request.

    `keys` are the limiter usage_daily keys to debit (one for signed-in, two for
    anonymous). `set_cookie` is a signed value to Set-Cookie when we minted a new
    anonymous id, else None.
    """
    info = current_oauth(request)
    user = _field(info, "user_info")
    if user:
        sub = _field(user, "sub") or _field(user, "preferred_username")
        token = _field(info, "access_token")
        return resolve_tier(user, token), [limiter.hash_key(f"sub:{sub}")], None

    # Anonymous: key by IP and a signed cookie id, minting the cookie if absent.
    ip = _client_ip(request)
    cookie_id = limiter.verify_cookie(request.cookies.get(ANON_COOKIE, ""))
    set_cookie = None
    if not cookie_id:
        cookie_id = secrets.token_urlsafe(18)
        set_cookie = limiter.sign_cookie(cookie_id)
    keys = [limiter.hash_key(f"ip:{ip}"), limiter.hash_key(f"cookie:{cookie_id}")]
    return "anon", keys, set_cookie


def set_anon_cookie(response, signed: str) -> None:
    # The Space runs inside an iframe on huggingface.co, so the cookie lives in a
    # cross-site context — it must be SameSite=None; Secure or the browser drops it.
    response.set_cookie(
        ANON_COOKIE, signed,
        max_age=_COOKIE_MAX_AGE, httponly=True, samesite="none", secure=True,
    )
