from speech_to_speech.setup.doctor import redact, run_doctor
from speech_to_speech.setup.models import CredentialRef, SetupProfile
from speech_to_speech.setup.system import GIB, SystemSnapshot


def test_doctor_reports_healthy_profile():
    messages = []
    profile = SetupProfile(pipeline={"stt": "parakeet-tdt", "tts": "kokoro"})
    result = run_doctor(
        [],
        profile_loader=lambda: profile,
        inspect=lambda: SystemSnapshot(24 * GIB, 20 * GIB, 1, 1, True),
        credential_getter=lambda ref: "secret",
        emit=messages.append,
    )

    assert result == 0
    assert any("ready" in message.lower() for message in messages)


def test_doctor_reports_missing_keychain_and_audio_without_secret():
    messages = []
    profile = SetupProfile(credentials={"llm": CredentialRef("missing")})
    result = run_doctor(
        [],
        profile_loader=lambda: profile,
        inspect=lambda: SystemSnapshot(16 * GIB, 20 * GIB, 0, 0, True),
        credential_getter=lambda ref: (_ for _ in ()).throw(RuntimeError("secret=top-secret")),
        emit=messages.append,
    )

    assert result == 1
    assert "top-secret" not in " ".join(messages)
    assert any("Privacy & Security" in message for message in messages)
    assert redact("Authorization: Bearer abc123") == "Authorization: [REDACTED]"


def test_doctor_reports_missing_installed_model(tmp_path):
    messages = []
    profile = SetupProfile(pipeline={"parakeet_tdt_model_name": str(tmp_path / "missing")})

    result = run_doctor(
        [],
        profile_loader=lambda: profile,
        inspect=lambda: SystemSnapshot(24 * GIB, 20 * GIB, 1, 1, True),
        emit=messages.append,
    )

    assert result == 1
    assert any("local model is missing" in message for message in messages)


def test_doctor_checks_every_configured_endpoint():
    messages = []
    profile = SetupProfile(
        pipeline={
            "openai_stt_base_url": "http://127.0.0.1:8001/v1",
            "responses_api_base_url": "http://127.0.0.1:8002/v1",
            "openai_tts_base_url": "http://127.0.0.1:8003/v1",
        }
    )

    result = run_doctor(
        [],
        profile_loader=lambda: profile,
        inspect=lambda: SystemSnapshot(24 * GIB, 20 * GIB, 1, 1, True),
        endpoint_urls=lambda: {"http://127.0.0.1:8002/v1"},
        emit=messages.append,
    )

    assert result == 1
    assert any("endpoint is not currently available" in message for message in messages)
