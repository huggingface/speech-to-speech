"""Unit tests for centralized provider configuration and credential resolution."""


from speech_to_speech.config.providers import (
    detect_provider,
    is_local_base_url,
    is_official_openai,
    resolve_credentials,
)


def test_provider_detection_by_model_name():
    assert detect_provider(model_name="gemini-2.5-flash").name == "gemini"
    assert detect_provider(model_name="gemini-2.0-flash").name == "gemini"
    assert detect_provider(model_name="gpt-4o-mini").name == "openai"
    assert detect_provider(model_name="o3-mini").name == "openai"
    assert detect_provider(model_name="llama-3.3-70b-versatile").name == "groq"
    assert detect_provider(model_name="deepseek-chat").name == "deepseek"
    assert detect_provider(model_name="openrouter/auto").name == "openrouter"
    assert detect_provider(model_name="togethercomputer/llama-2").name == "together"
    assert detect_provider(model_name="huggingface/meta-llama").name == "huggingface"
    assert detect_provider(model_name="custom-local-model") is None


def test_provider_detection_by_base_url():
    assert detect_provider(base_url="https://generativelanguage.googleapis.com/v1beta/openai/").name == "gemini"
    assert detect_provider(base_url="https://api.openai.com/v1").name == "openai"
    assert detect_provider(base_url="https://api.groq.com/openai/v1").name == "groq"
    assert detect_provider(base_url="https://api.deepseek.com/v1").name == "deepseek"
    assert detect_provider(base_url="https://openrouter.ai/api/v1").name == "openrouter"
    assert detect_provider(base_url="https://api.together.xyz/v1").name == "together"
    assert detect_provider(base_url="https://router.huggingface.co/v1").name == "huggingface"
    assert detect_provider(base_url="http://127.0.0.1:8080/v1") is None
    assert detect_provider(base_url="https://custom.endpoint.com/v1") is None


def test_is_local_base_url():
    assert is_local_base_url("http://127.0.0.1:8080/v1") is True
    assert is_local_base_url("http://localhost:8000/v1") is True
    assert is_local_base_url("http://0.0.0.0:8765/v1") is True
    assert is_local_base_url("https://api.openai.com/v1") is False
    assert is_local_base_url("https://generativelanguage.googleapis.com/v1beta/openai/") is False
    assert is_local_base_url(None) is False
    assert is_local_base_url("invalid-url") is False


def test_is_official_openai():
    assert is_official_openai("https://api.openai.com/v1") is True
    assert is_official_openai("https://api.openai.com/v1/") is True
    assert is_official_openai("https://generativelanguage.googleapis.com/v1beta/openai/") is False
    assert is_official_openai("http://127.0.0.1:8080/v1") is False
    assert is_official_openai(None) is False


def test_resolve_gemini_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-123")

    base_url, api_key = resolve_credentials(model_name="gemini-2.5-flash")
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert api_key == "test-gemini-key-123"


def test_resolve_openai_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-abc")

    base_url, api_key = resolve_credentials(model_name="gpt-4o-mini")
    assert base_url == "https://api.openai.com/v1"
    # Returns None or OPENAI_API_KEY so OpenAI SDK delegates or reads it
    assert api_key == "test-openai-key-abc"


def test_explicit_arguments_override_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://custom-env-gemini.com/v1")

    base_url, api_key = resolve_credentials(
        model_name="gemini-2.5-flash",
        base_url="https://override-url.com/v1",
        api_key="override-key-explicit",
    )
    assert base_url == "https://override-url.com/v1"
    assert api_key == "override-key-explicit"


def test_local_endpoint_injects_dummy_key_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    base_url, api_key = resolve_credentials(
        model_name=None,
        base_url="http://127.0.0.1:8080/v1",
        api_key=None,
    )
    assert base_url == "http://127.0.0.1:8080/v1"
    assert api_key == "none"


def test_remote_custom_endpoint_does_not_inject_dummy_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    base_url, api_key = resolve_credentials(
        model_name=None,
        base_url="https://my-private-llm.corp.internal/v1",
        api_key=None,
    )
    assert base_url == "https://my-private-llm.corp.internal/v1"
    assert api_key is None


def test_generic_llm_api_key_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "fallback-llm-token")

    base_url, api_key = resolve_credentials(model_name="llama-3.3-70b-versatile")
    assert base_url == "https://api.groq.com/openai/v1"
    assert api_key == "fallback-llm-token"
