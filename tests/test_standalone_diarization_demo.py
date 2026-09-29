"""Standalone example tests; the streaming component arrives in PR #583."""

import importlib
import json
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

streaming = pytest.importorskip("speech_to_speech.diarization.streaming")
alignment = importlib.import_module("speech_to_speech.diarization.alignment")
demo = importlib.import_module("speech_to_speech.diarization.demo")


def test_alignment_preserves_overlapping_and_unknown_speakers():
    words = [
        {"text": text, "timestamp": times}
        for text, times in [("one", (0, 0.5)), ("both", (0.5, 1)), ("two", (1, 2)), ("unknown", (2, 3))]
    ]
    segments = [streaming.SpeakerSegment(1, 0.5, 2), streaming.SpeakerSegment(0, 0, 1)]
    assert [word.speakers for word in alignment.align_words(words, segments)] == [(0,), (0, 1), (1,), ()]


@pytest.mark.parametrize("timestamp", [(None, 1), (0, None), (2, 1), (-1, 0), (0, float("inf"))])
def test_alignment_requires_valid_word_timestamps(timestamp):
    with pytest.raises(ValueError):
        alignment.align_words([{"text": "word", "timestamp": timestamp}], [])


def test_file_blocks_preserve_partial_final_block():
    audio = np.arange(13, dtype=np.float32)
    np.testing.assert_array_equal(np.concatenate(list(demo.file_blocks(audio, 20, False))), audio)
    args = demo.build_parser().parse_args(["--microphone", "--model", "local-checkpoint"])
    assert args.microphone and args.model == "local-checkpoint"


def test_file_demo_emits_complete_json_session(monkeypatch, capsys):
    class FakeDiarizer:
        sample_rate = 20
        processor = SimpleNamespace(streaming_latency_ms=650)
        active_speakers = ()
        processed_seconds = 0.0

        def push(self, audio, *, sample_rate):
            assert sample_rate == self.sample_rate
            self.processed_seconds += len(audio) / sample_rate
            return []

        def finish(self):
            return [streaming.SpeakerSegment(0, 0, self.processed_seconds)]

    monkeypatch.setattr(demo.StreamingDiarizer, "from_pretrained", lambda *args, **kwargs: FakeDiarizer())
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(61), 20)))
    demo.main(["--audio", "fixture.wav", "--model", "fixture", "--json"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["final"] is True
    assert records[-1]["processed_seconds"] == pytest.approx(3.05)
    assert records[-1]["active_speakers"] == []
    assert len(records[-1]["segments"]) == 1
    assert records[-1]["segments"][0]["speaker"] == 0
    assert records[-1]["segments"][0]["start"] == 0
    assert records[-1]["segments"][0]["end"] == pytest.approx(3.05)


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
    blocks = demo.microphone_blocks(20, None, stop)
    if overflow:
        with pytest.raises(RuntimeError, match="dropped"):
            next(blocks)
    else:
        assert next(blocks).shape == (2,)
        stop.set()
        assert list(blocks) == []
    assert closed == [True]
