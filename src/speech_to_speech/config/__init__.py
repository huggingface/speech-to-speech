"""Configuration modules for speech-to-speech."""

from speech_to_speech.config.providers import (
    DEFAULT_PROVIDER_BASE_URLS,
    PROVIDER_ENV_KEYS,
    PROVIDERS,
    ProviderSpec,
    detect_provider,
    is_local_base_url,
    is_official_openai,
    resolve_credentials,
)

__all__ = [
    "DEFAULT_PROVIDER_BASE_URLS",
    "PROVIDER_ENV_KEYS",
    "PROVIDERS",
    "ProviderSpec",
    "detect_provider",
    "is_local_base_url",
    "is_official_openai",
    "resolve_credentials",
]
