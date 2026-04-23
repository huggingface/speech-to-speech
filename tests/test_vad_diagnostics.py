from pathlib import Path
from queue import Queue
from threading import Event
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VAD.vad_diagnostics import VADDiagnosticsRecorder
from VAD.vad_handler import VADHandler
from VAD.vad_iterator import VADStateSnapshot


class _FakeVADModel:
    def __init__(self, probs: list[float]) -> None:
        self._probs = iter(probs)

    def reset_states(self) -> None:
        pass

    def __call__(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        return torch.tensor(next(self._probs), dtype=torch.float32)


def test_vad_diagnostics_recorder_writes_artifacts(tmp_path):
    recorder = VADDiagnosticsRecorder(
        tmp_path,
        sample_rate=16000,
        config={"threshold": 0.5},
    )
    snapshot = VADStateSnapshot(
        current_sample=512,
        window_size_samples=512,
        speech_prob=0.9,
        threshold=0.5,
        negative_threshold=0.35,
        triggered=True,
        pre_speech_samples=0,
        active_speech_samples=512,
        prefix_samples=0,
        temp_end_sample=0,
        transition="speech_start",
    )

    recorder.record_chunk(np.ones(512, dtype=np.int16), snapshot)
    recorder.record_event("speech_started", 32.0, audio_start_ms=0.0)
    session_dir = recorder.flush_session("session_end")

    assert session_dir is not None
    assert (session_dir / "audio.wav").exists()
    assert (session_dir / "diagnostics.json").exists()
    assert (session_dir / "report.html").exists()

    payload = json.loads((session_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert payload["flush_reason"] == "session_end"
    assert payload["chunk_count"] == 1
    assert payload["frames"][0]["transition"] == "speech_start"
    report = (session_dir / "report.html").read_text(encoding="utf-8")
    assert "audio.wav" in report
    assert "VAD Session Diagnostics" in report


def test_vad_handler_flushes_session_diagnostics(tmp_path, monkeypatch):
    fake_model = _FakeVADModel([0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
    monkeypatch.setattr(
        "VAD.vad_handler.torch.hub.load",
        lambda *args, **kwargs: (fake_model, None),
    )

    should_listen = Event()
    should_listen.set()
    handler = VADHandler(
        stop_event=Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={
            "should_listen": should_listen,
            "thresh": 0.5,
            "sample_rate": 16000,
            "min_silence_ms": 64,
            "min_speech_ms": 0,
            "speech_pad_ms": 32,
            "vad_diagnostics_dir": str(tmp_path),
            "text_output_queue": Queue(),
        },
    )

    chunks = [
        np.zeros(512, dtype=np.int16),
        np.ones(512, dtype=np.int16),
        np.ones(512, dtype=np.int16) * 2,
        np.zeros(512, dtype=np.int16),
        np.zeros(512, dtype=np.int16),
        np.zeros(512, dtype=np.int16),
    ]
    for chunk in chunks:
        list(handler.process(chunk.tobytes()))

    handler.on_session_end()

    session_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(session_dirs) == 1

    session_dir = session_dirs[0]
    payload = json.loads((session_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert payload["chunk_count"] >= 5
    assert any(event["kind"] == "speech_started" for event in payload["events"])
    assert any(event["kind"] == "speech_stopped" for event in payload["events"])
    assert any(frame["transition"] == "speech_start" for frame in payload["frames"])
    assert any(frame["triggered"] for frame in payload["frames"])
    assert any(frame["ending"] for frame in payload["frames"])
