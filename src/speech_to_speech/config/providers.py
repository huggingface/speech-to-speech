"""Centralized provider registry and configuration for LLM backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from speech_to_speech.utils.utils import load_dotenv_if_present


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    default_base_url: str
    api_key_env_vars: tuple[str, ...]
    base_url_env_vars: tuple[str, ...]
    model_keywords: tuple[str, ...]
    url_keywords: tuple[str, ...]


# Centralized registry of supported AI model providers
PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        default_base_url="https://api.openai.com/v1",
        api_key_env_vars=("OPENAI_API_KEY",),
        base_url_env_vars=("OPENAI_BASE_URL",),
        model_keywords=("gpt-", "o1-", "o3-", "chatgpt-"),
        url_keywords=("api.openai.com",),
    ),
    "gemini": ProviderSpec(
        name="gemini",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        base_url_env_vars=("GEMINI_BASE_URL", "GOOGLE_BASE_URL"),
        model_keywords=("gemini-", "gemini"),
        url_keywords=("generativelanguage.googleapis.com", "googleapis.com"),
    ),
    "groq": ProviderSpec(
        name="groq",
        default_base_url="https://api.groq.com/openai/v1",
        api_key_env_vars=("GROQ_API_KEY",),
        base_url_env_vars=("GROQ_BASE_URL",),
        model_keywords=("groq", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"),
        url_keywords=("api.groq.com", "groq.com"),
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        default_base_url="https://api.deepseek.com/v1",
        api_key_env_vars=("DEEPSEEK_API_KEY",),
        base_url_env_vars=("DEEPSEEK_BASE_URL",),
        model_keywords=("deepseek-chat", "deepseek-reasoner", "deepseek"),
        url_keywords=("api.deepseek.com", "deepseek.com"),
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        default_base_url="https://openrouter.ai/api/v1",
        api_key_env_vars=("OPENROUTER_API_KEY",),
        base_url_env_vars=("OPENROUTER_BASE_URL",),
        model_keywords=("openrouter",),
        url_keywords=("openrouter.ai",),
    ),
    "together": ProviderSpec(
        name="together",
        default_base_url="https://api.together.xyz/v1",
        api_key_env_vars=("TOGETHER_API_KEY",),
        base_url_env_vars=("TOGETHER_BASE_URL",),
        model_keywords=("together",),
        url_keywords=("api.together.xyz", "together.xyz"),
    ),
    "huggingface": ProviderSpec(
        name="huggingface",
        default_base_url="https://router.huggingface.co/v1",
        api_key_env_vars=("HF_TOKEN", "HUGGINGFACE_API_KEY"),
        base_url_env_vars=("HF_BASE_URL", "HUGGINGFACE_BASE_URL"),
        model_keywords=("huggingface",),
        url_keywords=("router.huggingface.co", "huggingface.co"),
    ),
}

DEFAULT_PROVIDER_BASE_URLS: dict[str, str] = {p.name: p.default_base_url for p in PROVIDERS.values()}
PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {p.name: p.api_key_env_vars for p in PROVIDERS.values()}


def is_local_base_url(base_url: Optional[str]) -> bool:
    """Return True if base_url points to localhost/loopback."""
    if not base_url:
        return False
    try:
        hostname = urlparse(base_url).hostname
        return hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
    except Exception:
        return False


def is_official_openai(base_url: Optional[str]) -> bool:
    """Return True if base_url points to official OpenAI API."""
    if not base_url:
        return False
    try:
        hostname = urlparse(base_url).hostname
        return hostname in {"api.openai.com"}
    except Exception:
        return False


def detect_provider(model_name: Optional[str] = None, base_url: Optional[str] = None) -> Optional[ProviderSpec]:
    """Detect provider specification from model name or base URL."""
    if base_url:
        b = base_url.lower()
        for provider in PROVIDERS.values():
            if any(k in b for k in provider.url_keywords):
                return provider
        if is_official_openai(base_url):
            return PROVIDERS["openai"]
        return None

    if model_name:
        m = model_name.lower()
        for provider in PROVIDERS.values():
            if any(k in m for k in provider.model_keywords):
                return provider

    return None


def resolve_credentials(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve base_url and api_key with clear precedence rules.

    Precedence:
    1. Base URL:
       Explicit argument -> Provider-specific env var -> Generic LLM_BASE_URL/OPENAI_BASE_URL -> Provider default
    2. API Key:
       Explicit argument -> Provider-specific env var -> Generic LLM_API_KEY -> Local dummy ("none") -> None
    """
    load_dotenv_if_present()
    provider = detect_provider(model_name, base_url)

    # 1. Resolve base_url
    if base_url is None:
        if provider:
            for env_var in provider.base_url_env_vars:
                val = os.environ.get(env_var)
                if val:
                    base_url = val
                    break
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
        if base_url is None and provider:
            base_url = provider.default_base_url

    # 2. Resolve api_key
    if api_key is None:
        if provider:
            for env_var in provider.api_key_env_vars:
                val = os.environ.get(env_var)
                if val:
                    api_key = val
                    break

        if api_key is None and os.environ.get("LLM_API_KEY"):
            api_key = os.environ.get("LLM_API_KEY")

        if (
            api_key is None
            and not os.environ.get("OPENAI_API_KEY")
            and base_url is not None
            and is_local_base_url(base_url)
        ):
            api_key = "none"

    return base_url, api_key
