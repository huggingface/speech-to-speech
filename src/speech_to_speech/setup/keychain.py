from __future__ import annotations

import getpass
import hashlib
import subprocess
from collections.abc import Callable
from typing import Any

from speech_to_speech.setup.models import CredentialRef

KEYCHAIN_SERVICE = "speech-to-speech"


def account_for_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return f"endpoint-{digest}"


class MacOSKeychain:
    def __init__(
        self,
        *,
        runner: Callable[..., Any] = subprocess.run,
        secret_prompt: Callable[[str], str] = getpass.getpass,
    ) -> None:
        self._runner = runner
        self._secret_prompt = secret_prompt

    def prompt_and_store(self, url: str) -> CredentialRef:
        secret = self._secret_prompt(f"API key for {url}: ")
        if not secret:
            raise ValueError("An API key is required for the selected endpoint.")
        reference = CredentialRef(account_for_url(url))
        result = self._runner(
            [
                "/usr/bin/security",
                "add-generic-password",
                "-U",
                "-a",
                reference.account,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            input=secret,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Could not store the endpoint API key in macOS Keychain.")
        return reference

    def get(self, reference: CredentialRef) -> str:
        result = self._runner(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                reference.account,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Could not read the endpoint API key from macOS Keychain.")
        return result.stdout.rstrip("\n")

    def delete(self, reference: CredentialRef) -> None:
        result = self._runner(
            [
                "/usr/bin/security",
                "delete-generic-password",
                "-a",
                reference.account,
                "-s",
                KEYCHAIN_SERVICE,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Could not remove the endpoint API key from macOS Keychain.")
