from speech_to_speech.setup.doctor import redact, run_doctor
from speech_to_speech.setup.models import CredentialRef, SetupProfile
from speech_to_speech.setup.system import GIB, SystemSnapshot


def test_doctor_reports_healthy_profile():
    messages = []
    profile = SetupProfile(pipeline={"stt": "parakeet-tdt", "tts": "kokoro"})
    result = run_doctor(
        [],
        profile_loader=lambda: profile,
        inspect=lambda: SystemSnapshot("Darwin", "arm64", False, 24 * GIB, "gemma", 20 * GIB, 1, 1, True),
        credential_getter=lambda ref: "secret",
        emit=messages.append,
    )

    assert result == 0
    assert any("ready" in message.lower() for message in messages)


def test_doctor_reports_missing_keychain_and_audio_without_secret():
    messages = []
    profile = SetupProfile(credentials={"llm": CredentialRef("speech-to-speech", "missing")})
    result = run_doctor(
        [],
        profile_loader=lambda: profile,
        inspect=lambda: SystemSnapshot("Darwin", "arm64", False, 16 * GIB, "small", 20 * GIB, 0, 0, True),
        credential_getter=lambda ref: (_ for _ in ()).throw(RuntimeError("secret=top-secret")),
        emit=messages.append,
    )

    assert result == 1
    assert "top-secret" not in " ".join(messages)
    assert any("Privacy & Security" in message for message in messages)
    assert redact("Authorization: Bearer abc123") == "Authorization: [REDACTED]"
