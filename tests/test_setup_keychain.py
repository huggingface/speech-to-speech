import hashlib
from types import SimpleNamespace

import pytest

from speech_to_speech.setup.keychain import MacOSKeychain, account_for_url


def test_keychain_account_is_stable_url_digest():
    url = "http://127.0.0.1:8080/v1"
    assert account_for_url(url) == "endpoint-" + hashlib.sha256(url.encode()).hexdigest()[:24]


def test_keychain_prompts_hidden_and_passes_secret_over_stdin():
    calls = []
    prompts = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    keychain = MacOSKeychain(runner=runner, secret_prompt=lambda prompt: prompts.append(prompt) or "my-key")
    secret = keychain.prompt("http://127.0.0.1:8080/v1")
    reference = keychain.store("http://127.0.0.1:8080/v1", secret)

    assert prompts == ["API key for http://127.0.0.1:8080/v1: "]
    command, kwargs = calls[0]
    assert "my-key" not in command
    assert kwargs["input"] == "my-key"
    assert reference.account == account_for_url("http://127.0.0.1:8080/v1")


def test_keychain_errors_never_include_secret():
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad secret=my-key")

    keychain = MacOSKeychain(runner=runner, secret_prompt=lambda _: "my-key")
    with pytest.raises(RuntimeError) as error:
        keychain.store("http://127.0.0.1:8080/v1", "my-key")

    assert "my-key" not in str(error.value)
