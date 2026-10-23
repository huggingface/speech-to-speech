"""Filler budgets must not block knowledge, cancellation or legacy scheduling."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from convfill_demo.runtime.run_convfill import ConvFillConfig

from .test_reasoner_cancel import _bare_system


class GatedBackend:
    def __init__(self):
        self.release = threading.Event()
        self.closed = threading.Event()

    def cancel(self):
        self.release.set()

    def backend_infer(self):
        try:
            assert self.release.wait(2), "test did not release the Reasoner"
            yield "First useful sentence."
            yield "Second useful sentence."
        finally:
            self.closed.set()


def limited_system(limit):
    backend = GatedBackend()
    system = _bare_system(backend)
    system.convfill_config = SimpleNamespace(max_filler_phrases=limit)
    system.audio_in_flight_fn = lambda: 0
    return system, backend


@pytest.mark.parametrize("limit", [0, 1])
def test_exhausted_budget_waits_without_repeating_and_streams_all_knowledge(limit):
    system, backend = limited_system(limit)
    stream = system.backend_fn()
    if limit:
        assert next(stream) == "<sil>"
    remaining = []
    worker = threading.Thread(target=lambda: remaining.extend(stream), daemon=True)
    worker.start()
    try:
        worker.join(0.15)
        assert worker.is_alive(), "filler budget ended the Reasoner prematurely"
        assert remaining == [], "another filler escaped after the budget was exhausted"
    finally:
        backend.release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert remaining == ["First useful sentence.", "Second useful sentence."]
    assert backend.closed.is_set()


def test_cancellation_after_filler_budget_is_exhausted_closes_the_stream():
    system, backend = limited_system(1)
    stream = system.backend_fn()
    assert next(stream) == "<sil>"
    remaining = []
    worker = threading.Thread(target=lambda: remaining.extend(stream), daemon=True)
    worker.start()
    system.cancel_turn()
    worker.join(2)
    assert not worker.is_alive()
    assert remaining == []
    assert backend.closed.wait(1)


@pytest.mark.parametrize("configured", [False, True])
def test_unset_budget_preserves_multiple_fillers(configured):
    system, backend = limited_system(None)
    if not configured:
        del system.convfill_config
    stream = system.backend_fn()
    try:
        assert next(stream) == "<sil>"
        assert next(stream) == "<sil>"
    finally:
        system.cancel_turn()
        stream.close()
    assert backend.closed.wait(1)


def test_limited_tool_continuation_does_not_add_another_acknowledgment():
    system, backend = limited_system(1)
    system.convfill_frontend = SimpleNamespace(turn_state_manager=SimpleNamespace(responses=["Let me check that."]))
    backend.release.set()
    assert list(system.backend_fn()) == ["First useful sentence.", "Second useful sentence."]


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_invalid_filler_budget_fails_before_model_setup(tmp_path, value):
    config = json.loads(CONFIG_PATH.read_text())
    config["max_filler_phrases"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="max_filler_phrases"):
        ConvFillConfig(path, "normal")


def test_hosted_budget_is_opt_in_and_resets_for_each_reply(tmp_path):
    assert ConvFillConfig(CONFIG_PATH, "normal").max_filler_phrases == 1
    config = json.loads(CONFIG_PATH.read_text())
    del config["max_filler_phrases"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(config))
    assert ConvFillConfig(path, "normal").max_filler_phrases is None
    system, backend = limited_system(1)
    for _ in range(2):
        backend.release.clear()
        stream = system.backend_fn()
        assert next(stream) == "<sil>"
        backend.release.set()
        assert list(stream) == ["First useful sentence.", "Second useful sentence."]


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/demo_mode/convfill_hosted_config.json"
