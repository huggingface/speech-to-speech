"""Unit tests for the voice store module (speech_to_speech.voice_store).

The store is the one new seam introduced by the voice-cloning feature: a
global library of cloned voices persisted as one folder per voice (normalized
reference WAV + metadata JSON). Tests run against temp directories.
"""

import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf

from speech_to_speech.voice_store import (
    MAX_UPLOAD_BYTES,
    STORE_SAMPLE_RATE,
    VoiceStore,
    VoiceValidationError,
)


def _wav_bytes(seconds: float = 4.0, sr: int = 16000, channels: int = 1, freq: float = 440.0) -> bytes:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestCreateAndList:
    def test_add_voice_returns_content_hash_id_and_lists_it(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        audio = _wav_bytes()

        record = store.add_voice(audio, ref_text="hello reference", name="My Voice")

        expected_id = hashlib.sha256(audio).hexdigest()[:16]
        assert record.voice_id == expected_id
        assert record.name == "My Voice"
        assert record.ref_text == "hello reference"
        assert record.created_at

        listed = store.list_voices()
        assert [v.voice_id for v in listed] == [expected_id]
        assert listed[0].name == "My Voice"

    def test_store_layout_is_one_folder_per_voice(self, tmp_path):
        root = tmp_path / "voices"
        store = VoiceStore(root)
        record = store.add_voice(_wav_bytes(), ref_text="ref", name="v")

        voice_dir = root / record.voice_id
        assert (voice_dir / "ref.wav").exists()
        meta = json.loads((voice_dir / "voice.json").read_text())
        assert meta["voice_id"] == record.voice_id
        assert meta["name"] == "v"
        assert meta["ref_text"] == "ref"

    def test_fresh_store_instance_sees_persisted_voices(self, tmp_path):
        root = tmp_path / "voices"
        VoiceStore(root).add_voice(_wav_bytes(), ref_text="ref", name="v")

        reopened = VoiceStore(root)
        assert [v.name for v in reopened.list_voices()] == ["v"]

    def test_empty_store_lists_nothing(self, tmp_path):
        assert VoiceStore(tmp_path / "voices").list_voices() == []


class TestDedup:
    def test_identical_audio_converges_on_one_voice(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        audio = _wav_bytes()

        first = store.add_voice(audio, ref_text="ref", name="a")
        second = store.add_voice(audio, ref_text="ref", name="a")

        assert first.voice_id == second.voice_id
        assert len(store.list_voices()) == 1

    def test_reupload_overwrites_transcript_and_name(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        audio = _wav_bytes()

        store.add_voice(audio, ref_text="typo transcript", name="old")
        updated = store.add_voice(audio, ref_text="fixed transcript", name="new")

        listed = store.list_voices()
        assert len(listed) == 1
        assert listed[0].ref_text == "fixed transcript"
        assert listed[0].name == "new"
        assert updated.ref_text == "fixed transcript"

    def test_different_audio_gets_different_id(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        a = store.add_voice(_wav_bytes(freq=440.0), ref_text="r", name="a")
        b = store.add_voice(_wav_bytes(freq=220.0), ref_text="r", name="b")
        assert a.voice_id != b.voice_id
        assert len(store.list_voices()) == 2


class TestNormalization:
    def test_stereo_48k_input_is_stored_as_mono_24k_pcm16(self, tmp_path):
        root = tmp_path / "voices"
        store = VoiceStore(root)
        record = store.add_voice(
            _wav_bytes(seconds=5.0, sr=48000, channels=2),
            ref_text="ref",
            name="v",
        )

        info = sf.info(str(root / record.voice_id / "ref.wav"))
        assert info.samplerate == STORE_SAMPLE_RATE
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.format == "WAV"
        assert info.duration == pytest.approx(5.0, abs=0.05)


class TestResolve:
    def test_resolve_returns_reference_audio_path_and_transcript(self, tmp_path):
        root = tmp_path / "voices"
        store = VoiceStore(root)
        record = store.add_voice(_wav_bytes(), ref_text="the reference transcript", name="v")

        resolved = store.resolve(record.voice_id)

        assert resolved is not None
        assert resolved.ref_text == "the reference transcript"
        assert resolved.ref_audio == str(root / record.voice_id / "ref.wav")

    def test_resolve_unknown_id_returns_none(self, tmp_path):
        assert VoiceStore(tmp_path / "voices").resolve("deadbeefdeadbeef") is None


class TestValidation:
    def test_rejects_unreadable_audio(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        with pytest.raises(VoiceValidationError) as exc:
            store.add_voice(b"definitely not audio", ref_text="ref", name="v")
        assert exc.value.status_code == 415

    def test_rejects_non_wav_container(self, tmp_path):
        buf = io.BytesIO()
        t = np.linspace(0, 4.0, 4 * 16000, endpoint=False)
        sf.write(buf, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000, format="FLAC")
        store = VoiceStore(tmp_path / "voices")
        with pytest.raises(VoiceValidationError) as exc:
            store.add_voice(buf.getvalue(), ref_text="ref", name="v")
        assert exc.value.status_code == 415

    def test_rejects_oversized_upload(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        with pytest.raises(VoiceValidationError) as exc:
            store.add_voice(b"\x00" * (MAX_UPLOAD_BYTES + 1), ref_text="ref", name="v")
        assert exc.value.status_code == 413

    def test_rejects_clip_outside_duration_clamp(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        with pytest.raises(VoiceValidationError, match="short"):
            store.add_voice(_wav_bytes(seconds=1.0), ref_text="ref", name="v")
        with pytest.raises(VoiceValidationError, match="long"):
            store.add_voice(_wav_bytes(seconds=61.0, sr=8000), ref_text="ref", name="v")

    def test_rejects_missing_or_oversized_name_and_transcript(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        audio = _wav_bytes()
        with pytest.raises(VoiceValidationError, match="name"):
            store.add_voice(audio, ref_text="ref", name="   ")
        with pytest.raises(VoiceValidationError, match="name"):
            store.add_voice(audio, ref_text="ref", name="x" * 200)
        with pytest.raises(VoiceValidationError, match="transcript"):
            store.add_voice(audio, ref_text="", name="v")
        with pytest.raises(VoiceValidationError, match="transcript"):
            store.add_voice(audio, ref_text="x" * 5000, name="v")

    def test_rejected_upload_leaves_no_store_entry(self, tmp_path):
        root = tmp_path / "voices"
        store = VoiceStore(root)
        with pytest.raises(VoiceValidationError):
            store.add_voice(_wav_bytes(seconds=1.0), ref_text="ref", name="v")
        assert store.list_voices() == []
        assert [p for p in root.iterdir()] == []
