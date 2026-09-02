import json
import stat

import pytest

from speech_to_speech.setup.models import CredentialRef, ManagedService, SetupProfile
from speech_to_speech.setup.profiles import load_profile, save_profile


def test_profile_round_trip_is_versioned_private_and_secret_free(tmp_path):
    path = tmp_path / "default.json"
    profile = SetupProfile(
        name="default",
        pipeline={"stt": "parakeet-tdt", "llm_backend": "responses-api", "tts": "kokoro"},
        credentials={
            "llm": CredentialRef(service="speech-to-speech", account="endpoint-127.0.0.1-8080")
        },
        managed_services=[
            ManagedService(kind="llm", model="ggml-org/gemma-4-12B-it-GGUF:Q4_0", runtime="llama.cpp")
        ],
    )

    save_profile(profile, path)

    assert load_profile(path) == profile
    payload = path.read_text()
    assert "secret-value" not in payload
    assert json.loads(payload)["schema_version"] == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_profile_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "future.json"
    path.write_text('{"schema_version": 99, "name": "default", "pipeline": {}}')

    with pytest.raises(ValueError, match="schema version 99"):
        load_profile(path)
