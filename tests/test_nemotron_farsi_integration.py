"""Opt-in real-model test. Set FARSI_ASR_TEST_AUDIO to a Persian audio file.

Run with the nemo extra installed. FARSI_ASR_TEST_MODEL can point to a local
.nemo checkpoint; otherwise the checkpoint downloads from Hugging Face.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from speech_to_speech.backend_registry import create_backend_handler
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT.nemo_asr_handler import FARSI_MODEL, SAMPLE_RATE
from tests.test_nemo_asr_handler import _context


@pytest.mark.skipif(
    not os.environ.get("FARSI_ASR_TEST_AUDIO"), reason="Set FARSI_ASR_TEST_AUDIO for real-model inference"
)
def test_real_farsi_checkpoint_through_backend_factory(monkeypatch):
    import soundfile as sf
    from scipy.signal import resample_poly

    local_model = os.environ.get("FARSI_ASR_TEST_MODEL")
    if local_model:
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: local_model)
    args = parse_arguments(
        [
            "--stt",
            "nemotron-streaming",
            "--nemotron_streaming_model_name",
            FARSI_MODEL,
            "--nemotron_streaming_device",
            os.environ.get("FARSI_ASR_TEST_DEVICE", "cpu"),
        ]
    )
    handler = create_backend_handler(args.stt_backend, _context())
    audio, sr = sf.read(os.environ["FARSI_ASR_TEST_AUDIO"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        divisor = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    assert audio.size >= SAMPLE_RATE, "Use at least one second of Persian speech"

    def final():
        event = list(
            handler.process(
                VADAudio(
                    audio=audio,
                    mode="final",
                    turn_id="fa_test",
                    turn_revision=1,
                    speech_end_at_s=123.0,
                )
            )
        )[0]
        assert isinstance(event, Transcription)
        assert event.language_code == "fa"
        assert event.turn_id == "fa_test" and event.turn_revision == 1
        assert event.speech_stopped_at_s == 123.0
        assert event.text
        return event.text

    text = final()
    assert text == final(), "Repeated decoding must be deterministic"
    expected = os.environ.get("FARSI_ASR_TEST_EXPECTED_TEXT")
    if expected:
        assert text == expected
    partial = list(
        handler.process(
            VADAudio(
                audio=audio[: max(SAMPLE_RATE, len(audio) // 2)],
                mode="progressive",
                turn_id="fa_partial",
            )
        )
    )[0]
    assert isinstance(partial, PartialTranscription)
    assert partial.turn_id == "fa_partial"
    silence = list(handler.process(VADAudio(audio=np.zeros(SAMPLE_RATE, dtype=np.float32))))[0]
    assert silence.text == "" and silence.language_code == "fa"
    handler.on_session_end()
    assert handler.last_language == "fa"
