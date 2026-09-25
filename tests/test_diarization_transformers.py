"""Real Transformers API smoke tests; no checkpoint downloads or GPU required.

Run with the supporting Transformers branch/release installed. The regular
application environment may predate that API, in which case this module skips.
"""

import numpy as np
import pytest

pytest.importorskip("transformers.models.nemotron3_diarization")

from transformers import (  # noqa: E402
    Nemotron3DiarizationConfig,
    Nemotron3DiarizationForAudioFrameClassification,
    Nemotron3DiarizationProcessor,
    NemotronAsrStreamingFeatureExtractor,
)

from speech_to_speech.diarization import StreamingDiarizer  # noqa: E402


def make_real_diarizer(mode):
    processor = Nemotron3DiarizationProcessor(NemotronAsrStreamingFeatureExtractor(feature_size=8))
    config = Nemotron3DiarizationConfig(
        audio_config={
            "num_mel_bins": 8,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
        },
        head_config={"hidden_size": 8, "audio_hidden_size": 16, "subsampling_factor": 8, "num_speakers": 2},
        streaming_config={
            "speaker_cache_length": 8,
            "fifo_length": 4,
            "speaker_cache_update_period": 4,
            "num_speakers": 2,
            "subsampling_factor": 8,
        },
    )
    diarizer = StreamingDiarizer(
        processor, Nemotron3DiarizationForAudioFrameClassification(config), streaming_mode=mode
    )
    return diarizer


@pytest.mark.parametrize("mode", ["low_latency", "very_low_latency", "ultra_low_latency"])
@pytest.mark.parametrize("length", [1, 159, 160, 16680, 35041])
def test_real_processor_and_tiny_model_cover_complete_audio(mode, length):
    diarizer = make_real_diarizer(mode)
    audio = np.random.default_rng(0).normal(0, 0.1, length).astype(np.float32)
    segments = []
    for start in range(0, length, 173):
        segments.extend(diarizer.push(audio[start : start + 173], sample_rate=16000))
    segments.extend(diarizer.finish())
    assert diarizer.processed_seconds == length / 16000
    assert all(0 <= segment.start < segment.end <= length / 16000 for segment in segments)
    assert diarizer.buffered_samples == 0
    assert not diarizer.active_speakers


@pytest.mark.parametrize("mode", ["low_latency", "very_low_latency", "ultra_low_latency"])
def test_real_model_reuses_identity_across_centered_utterance_boundaries(mode):
    diarizer = make_real_diarizer(mode)
    caches = []
    hook = diarizer.model.register_forward_pre_hook(
        lambda model, args, kwargs: caches.append(kwargs["speaker_cache"]), with_kwargs=True
    )
    try:
        for length in [173, 8001, 16001]:
            before = len(caches)
            previous_cache = diarizer._cache
            audio = np.random.default_rng(length).normal(0, 0.1, length).astype(np.float32)
            segments = diarizer.push(audio, sample_rate=16000)
            segments.extend(diarizer.finish_utterance())
            assert caches[before] is previous_cache
            assert diarizer._cache is not None
            assert diarizer._cache.num_speakers == 2
            assert diarizer.processed_seconds == 0
            assert all(0 <= s.start < s.end <= length / 16000 for s in segments)
        diarizer.reset()
        assert diarizer._cache is None
    finally:
        hook.remove()
