import json
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_to_speech.diarization import SpeakerSegment, StreamingDiarizer
from speech_to_speech.diarization.alignment import align_words
from speech_to_speech.diarization.demo import build_parser, file_blocks, main, microphone_blocks


class Inputs(dict):
    def to(self, *args, **kwargs):
        return self


class Processor:
    """Small STFT geometry with the same overlap rules as the real processor."""

    feature_extractor = SimpleNamespace(sampling_rate=20, hop_length=2, n_fft=8)
    num_samples_first_audio_chunk = 13
    num_samples_per_audio_chunk = 18
    num_mel_frames_per_step = 4

    def __init__(self):
        self.calls = []

    def set_streaming_mode(self, mode):
        if mode != "low_latency":
            raise ValueError("unsupported mode")

    def audio_chunk_start(self, frame):
        return frame * 2 - 4

    def __call__(self, audio, *, sampling_rate, is_streaming, is_first_audio_chunk, is_last_audio_chunk):
        assert sampling_rate == 20 and is_streaming
        self.calls.append((audio.copy(), is_first_audio_chunk, is_last_audio_chunk))
        frames = len(audio) // 2 if is_first_audio_chunk else (len(audio) - 8) // 2 + 1
        if not is_last_audio_chunk:
            assert frames == 6
        result = Inputs(frames=frames)
        if not is_last_audio_chunk:
            result["num_lookahead_frames"] = 2
        return result


class Model:
    device = "cpu"
    dtype = torch.float32

    def __init__(self):
        self.caches = []

    def eval(self):
        return self

    def __call__(self, *, frames, num_lookahead_frames, speaker_cache):
        assert not torch.is_grad_enabled()
        self.caches.append(speaker_cache)
        start = speaker_cache or 0
        count = frames - num_lookahead_frames
        index = torch.arange(start, start + count)
        logits = torch.stack((index < 8, (index >= 4) & (index < 11)), dim=-1).float() * 20 - 10
        return SimpleNamespace(logits=logits[None], speaker_cache=start + count)


def make_diarizer():
    return StreamingDiarizer(Processor(), Model())


def test_from_pretrained_rejects_incompatible_checkpoint(monkeypatch):
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return object(), {"missing_keys": ["head.proj.weight"], "unexpected_keys": ["model.proj.weight"]}

    monkeypatch.setattr(AutoModelForAudioFrameClassification, "from_pretrained", load)
    with pytest.raises(RuntimeError, match="missing=1, unexpected=1"):
        StreamingDiarizer.from_pretrained("example/model", revision="moving-branch")
    assert calls[0]["output_loading_info"] is True


@pytest.mark.parametrize("block_size", [1, 3, 13, 18, 10000])
def test_arbitrary_boundaries_preserve_cache_overlap_and_timestamps(block_size):
    diarizer = make_diarizer()
    audio = np.arange(61, dtype=np.float32)
    segments = []
    for start in range(0, len(audio), block_size):
        segments.extend(diarizer.push(audio[start : start + block_size], sample_rate=20))
        assert diarizer.buffered_samples < diarizer.processor.num_samples_per_audio_chunk
    segments.extend(diarizer.finish())
    assert segments == [SpeakerSegment(0, 0, 0.8), SpeakerSegment(1, 0.4, 1.1)]
    assert diarizer.processed_seconds == 3.05
    assert diarizer.active_speakers == ()
    assert diarizer.buffered_samples == 0
    calls = diarizer.processor.calls
    assert calls[0][1:] == (True, False)
    assert calls[-1][1:] == (False, True)
    for index, (chunk, first, last) in enumerate(calls[:-1]):
        start = 0 if first else diarizer.processor.audio_chunk_start(index * 4)
        np.testing.assert_array_equal(chunk, audio[start : start + len(chunk)])
    assert diarizer.model.caches == [None] + list(range(4, 4 * len(calls), 4))


@pytest.mark.parametrize("length", [1, 2, 7, 12, 13, 14, 21, 22, 23, 37])
def test_short_and_exact_boundary_endings_are_flushed_once(length):
    diarizer = make_diarizer()
    segments = diarizer.push(np.zeros(length, dtype=np.float32), sample_rate=20) + diarizer.finish()
    assert segments[0] == SpeakerSegment(0, 0, min(length / 20, 0.8))
    assert diarizer.processed_seconds == length / 20
    assert all(0 <= segment.start < segment.end <= length / 20 for segment in segments)
    count = len(diarizer.processor.calls)
    assert diarizer.finish() == []
    assert len(diarizer.processor.calls) == count
    with pytest.raises(RuntimeError, match="reset"):
        diarizer.push(np.zeros(1, dtype=np.float32), sample_rate=20)


def test_reset_discards_audio_and_speaker_identity():
    diarizer = make_diarizer()
    diarizer.push(np.zeros(13, dtype=np.float32), sample_rate=20)
    assert diarizer.active_speakers == (0,)
    diarizer.reset()
    assert diarizer.active_speakers == ()
    assert diarizer.processed_seconds == 0
    diarizer.push(np.zeros(1, dtype=np.float32), sample_rate=20)
    assert diarizer.finish() == [SpeakerSegment(0, 0, 0.05)]
    assert diarizer.model.caches[-1] is None


def test_empty_session_does_not_call_model():
    diarizer = make_diarizer()
    assert diarizer.push(np.empty(0, dtype=np.float32), sample_rate=20) == []
    assert diarizer.finish() == []
    assert not diarizer.model.caches


@pytest.mark.parametrize(
    "audio,rate",
    [(np.zeros(3), 16000), (np.zeros((3, 2)), 20), (np.ones(3, dtype=np.int16), 20), (np.array([np.nan]), 20)],
)
def test_invalid_input_does_not_mutate_session(audio, rate):
    diarizer = make_diarizer()
    with pytest.raises(ValueError):
        diarizer.push(audio, sample_rate=rate)
    assert diarizer.buffered_samples == 0
    assert not diarizer.processor.calls


def test_probability_threshold_and_silence():
    diarizer = StreamingDiarizer(Processor(), Model(), threshold=0.99999)
    diarizer.push(np.zeros(100, dtype=np.float32), sample_rate=20)
    assert diarizer.active_speakers == ()
    assert diarizer.finish() == []
    for threshold in [0, 1, float("nan")]:
        with pytest.raises(ValueError, match="threshold"):
            StreamingDiarizer(Processor(), Model(), threshold=threshold)


def test_alignment_preserves_overlapping_and_unknown_speakers():
    words = [
        {"text": text, "timestamp": times}
        for text, times in [("one", (0, 0.5)), ("both", (0.5, 1)), ("two", (1, 2)), ("unknown", (2, 3))]
    ]
    segments = [SpeakerSegment(1, 0.5, 2), SpeakerSegment(0, 0, 1)]
    assert [word.speakers for word in align_words(words, segments)] == [(0,), (0, 1), (1,), ()]


@pytest.mark.parametrize("timestamp", [(None, 1), (0, None), (2, 1), (-1, 0), (0, float("inf"))])
def test_alignment_requires_valid_word_timestamps(timestamp):
    with pytest.raises(ValueError):
        align_words([{"text": "word", "timestamp": timestamp}], [])


def test_file_demo_preserves_partial_final_block():
    audio = np.arange(13, dtype=np.float32)
    np.testing.assert_array_equal(np.concatenate(list(file_blocks(audio, 20, False))), audio)
    args = build_parser().parse_args(["--microphone", "--model", "local-checkpoint"])
    assert args.microphone and args.model == "local-checkpoint"


def test_file_demo_emits_complete_json_session(monkeypatch, capsys):
    monkeypatch.setattr(StreamingDiarizer, "from_pretrained", lambda *args, **kwargs: make_diarizer())
    monkeypatch.setattr(Processor, "streaming_latency_ms", 650, raising=False)
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(61), 20)))
    main(["--audio", "fixture.wav", "--model", "fixture", "--json"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["final"] is True
    assert records[-1]["processed_seconds"] == 3.05
    assert records[-1]["active_speakers"] == []
    assert sum(len(record["segments"]) for record in records) == 2


@pytest.mark.parametrize("overflow", [False, True])
def test_microphone_stops_and_never_silently_drops_audio(monkeypatch, overflow):
    closed = []

    class InputStream:
        def __init__(self, *, callback, **kwargs):
            self.callback = callback

        def __enter__(self):
            for _ in range(51 if overflow else 1):
                self.callback(np.zeros((2, 1), dtype=np.float32), 2, None, False)

        def __exit__(self, *args):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=InputStream))
    stop = Event()
    blocks = microphone_blocks(20, None, stop)
    if overflow:
        with pytest.raises(RuntimeError, match="dropped"):
            next(blocks)
    else:
        assert next(blocks).shape == (2,)
        stop.set()
        assert list(blocks) == []
    assert closed == [True]


def test_utterance_boundary_preserves_speaker_cache_but_restarts_audio_windows():
    diarizer = make_diarizer()
    diarizer.push(np.ones(13, dtype=np.float32), sample_rate=20)
    diarizer.finish_utterance()
    cache = diarizer._cache
    assert cache is not None
    assert diarizer.processed_seconds == 0
    assert diarizer.buffered_samples == 0
    diarizer.push(np.ones(3, dtype=np.float32), sample_rate=20)
    diarizer.finish_utterance()
    assert diarizer.model.caches[-1] == cache
    assert diarizer.processor.calls[-1][1:] == (True, True)
    diarizer.reset()
    assert diarizer._cache is None
