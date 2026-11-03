"""Exercise the real lease broker with simulated allocation lifetimes, no models."""

import ast
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

import numpy as np
import pytest


@pytest.fixture
def worker_definitions():
    # Execute the actual broker and atomic publication helper without importing
    # optional CUDA dependencies or allocating a GPU.
    tree = ast.parse((Path(__file__).parents[1] / "src/speech_to_speech/zerogpu/_worker.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {"LeaseBroker", "_atomic_json"}
    ]
    namespace = dict(
        Path=Path,
        Lock=Lock,
        Thread=Thread,
        np=np,
        time=time,
        json=json,
        uuid4=uuid4,
        log=logging.getLogger("lease-test"),
    )
    exec(compile(tree, "worker.py", "exec"), namespace)
    return namespace


@pytest.fixture
def broker_type(worker_definitions):
    return worker_definitions["LeaseBroker"]


@pytest.fixture
def atomic_json(worker_definitions):
    return worker_definitions["_atomic_json"]


def requests(directory, count=1):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        paths = list(directory.glob("*.request.json"))
        if len(paths) >= count:
            return paths
        time.sleep(0.002)
    raise AssertionError("Requests never reached simulated GPU worker")


def complete(directory, request, value):
    ident = request.name.split(".")[0]
    (directory / f"{ident}.accepted").touch()
    (directory / f"{ident}.events").write_text(
        json.dumps({"kind": "result", "value": value}) + "\n" + json.dumps({"kind": "end"}) + "\n"
    )


def test_concurrent_stages_share_allocation_and_keep_results_separate(broker_type, tmp_path):
    def runner(ident):
        directory = tmp_path / ident
        paths = requests(directory, 2)
        for request in reversed(paths):
            payload = json.loads(request.read_text())["payload"]
            complete(directory, request, payload["messages"][0])
        yield {"status": "ready"}

    broker = broker_type(runner, tmp_path, wait_timeout=3)
    with ThreadPoolExecutor(2) as pool:
        jobs = [
            pool.submit(lambda value=value: list(broker.submit("tokens", {"messages": [value]})))
            for value in ["first", "second"]
        ]
        assert [job.result(timeout=4)[0]["value"] for job in jobs] == ["first", "second"]
    assert broker.leases_started == 1


def test_retired_lease_retries_only_unaccepted_command(broker_type, tmp_path):
    leases = []

    def runner(ident):
        directory = tmp_path / ident
        leases.append(ident)
        request = requests(directory)[0]
        if len(leases) == 1:
            (directory / "closed").touch()
        else:
            complete(directory, request, "renewed")
        yield {"status": "ready"}

    broker = broker_type(runner, tmp_path, wait_timeout=3)
    assert list(broker.submit("tokens", {})) == [{"kind": "result", "value": "renewed"}]
    assert broker.leases_started == 2


def test_accepted_work_is_never_replayed_when_worker_dies(broker_type, tmp_path):
    def runner(ident):
        directory = tmp_path / ident
        request = requests(directory)[0]
        (directory / (request.name.split(".")[0] + ".accepted")).touch()
        yield {"status": "ready"}

    broker = broker_type(runner, tmp_path, wait_timeout=3)
    with pytest.raises(RuntimeError, match="before completing an accepted operation"):
        list(broker.submit("tokens", {}))
    assert broker.leases_started == 1


def test_allocation_failure_reaches_every_waiter(broker_type, tmp_path):
    def runner(ident):
        requests(tmp_path / ident, 2)
        raise RuntimeError("quota unavailable")
        yield  # This runner has the decorated generator's contract.

    broker = broker_type(runner, tmp_path, wait_timeout=3)
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(lambda: list(broker.submit("tokens", {}))) for _ in range(2)]
        for job in jobs:
            with pytest.raises(RuntimeError, match="quota unavailable"):
                job.result(timeout=4)
    assert broker.leases_started == 1


def test_silent_worker_has_bounded_wait(broker_type, tmp_path):
    release = Event()

    def runner(ident):
        yield {"status": "ready"}
        release.wait(2)

    broker = broker_type(runner, tmp_path, wait_timeout=0.05)
    try:
        with pytest.raises(TimeoutError, match="Timed out waiting"):
            list(broker.submit("tokens", {}))
    finally:
        release.set()


def test_prepare_waits_for_true_marker_and_reuses_live_allocation(broker_type, atomic_json, tmp_path, monkeypatch):
    warming, release, finish = Event(), Event(), Event()
    writing, publish = Event(), Event()
    write_text = Path.write_text

    def paused_write(path, data, *args, **kwargs):
        if path.name.startswith("ready.json.") and path.suffix == ".tmp":
            # Hold an incomplete temporary file open while prepare() polls.
            path.touch()
            writing.set()
            assert publish.wait(2)
        return write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", paused_write)

    def runner(ident):
        warming.set()
        assert release.wait(2)
        atomic_json(tmp_path / ident / "ready.json", {"models": ["stt", "llm", "tts"]})
        yield {"status": "ready"}
        finish.wait(2)

    broker = broker_type(runner, tmp_path, wait_timeout=1)
    try:
        with ThreadPoolExecutor(1) as pool:
            job = pool.submit(broker.prepare)
            assert warming.wait(1)
            assert not job.done()
            release.set()
            assert writing.wait(1)
            assert not (broker.current / "ready.json").exists()
            assert not job.done()
            publish.set()
            assert job.result(timeout=2)["models"] == ["stt", "llm", "tts"]
            assert broker.prepare()["models"] == ["stt", "llm", "tts"]
            assert broker.leases_started == 1
    finally:
        release.set()
        publish.set()
        finish.set()
        # Let the broker finish writing lifecycle markers before tmp_path cleanup.
        deadline = time.monotonic() + 2
        while broker.current is not None and not (broker.current / "finished").exists():
            assert time.monotonic() < deadline, "Simulated worker did not finish"
            time.sleep(0.002)


def test_prepare_surfaces_failure_and_bounds_wait(broker_type, tmp_path):
    def fail(ident):
        raise RuntimeError("warmup failed")
        yield

    with pytest.raises(RuntimeError, match="warmup failed"):
        broker_type(fail, tmp_path).prepare()
    finish = Event()

    def silent(ident):
        yield {"status": "starting"}
        finish.wait(2)

    broker = broker_type(silent, tmp_path, wait_timeout=0.03)
    try:
        with pytest.raises(TimeoutError, match="Timed out preparing"):
            broker.prepare()
        assert (broker.current / "stop").exists()
        assert broker.leases_started == 1
    finally:
        finish.set()
