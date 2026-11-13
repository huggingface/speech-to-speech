"""Exercise real readiness publication with fake, observable GPU operations."""

import ast
import json
import logging
import time
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace
from uuid import uuid4

import pytest


def executor(tmp_path, fail=None):
    tree = ast.parse((Path(__file__).parents[1] / "src/speech_to_speech/zerogpu/_worker.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_atomic_json", "lease_worker"}
    ]
    for node in tree.body:
        node.decorator_list = []
    calls = []

    def stage(name):
        assert not (tmp_path / "ready.json").exists()
        calls.append(name)
        if name == fail:
            raise RuntimeError(name + " failed")
        return ["hello"]

    ns = dict(
        Path=Path,
        Lock=Lock,
        Thread=Thread,
        time=time,
        json=json,
        uuid4=uuid4,
        log=logging.getLogger("startup-test"),
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda: "fake GPU")),
        _lease_directory=lambda ident: tmp_path,
        _gpu_models=lambda: {},
        warmup_stt=lambda handler: stage("stt"),
        stt=object(),
        tts=SimpleNamespace(initialize_lease=lambda: stage("graphs"), warmup=lambda: stage("tts")),
        _tokens=lambda *args, **kwargs: iter(stage("llm")),
        LEASE_ADMISSION_SECONDS=5,
        LEASE_IDLE_SECONDS=5,
    )
    exec(compile(tree, "worker.py", "exec"), ns)
    return ns, calls


def test_ready_only_after_exclusive_graphs_and_all_real_model_warmups(tmp_path):
    ns, calls = executor(tmp_path)
    stream = ns["lease_worker"]("lease")
    assert next(stream)["status"] == "ready"
    assert calls == ["graphs", "stt", "llm", "tts"]
    assert json.loads((tmp_path / "ready.json").read_text())["models"] == ["stt", "llm", "tts"]
    stream.close()


@pytest.mark.parametrize("stage", ["graphs", "stt", "llm", "tts"])
def test_failed_warmup_never_reports_ready(tmp_path, stage):
    ns, calls = executor(tmp_path, stage)
    with pytest.raises(RuntimeError, match=stage + " failed"):
        next(ns["lease_worker"]("lease"))
    assert not (tmp_path / "ready.json").exists()
