# ruff: noqa: I001
"""Stateless GPU operations; conversation and cancellation authority stay in parent."""

import spaces  # noqa: F401  # Must precede torch/model imports.

import re
import json
import logging
import time
from uuid import uuid4
from pathlib import Path
from queue import Empty
from threading import Event, Thread, Lock

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from ._model_stt import create_stt, warmup_stt, transcribe as transcribe_stt
from ._tts_provider import create_provider


# Populated only by runtime.initialize(), before the deployment starts ZeroGPU.
stt = llm_model = llm_tokenizer = tts = None


def initialize_models(stt_model, llm_model_id, tts_backend, tts_model, tts_voice):
    global stt, llm_model, llm_tokenizer, tts
    stt = create_stt(stt_model)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_model = (
        AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            dtype=torch.bfloat16,
        )
        .to("cuda")
        .eval()
    )
    tts = create_provider(tts_backend, tts_model, tts_voice)


def _assert_gpu(model):
    device = str(next(model.parameters()).device)
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("ZeroGPU CUDA allocation required; CPU fallback is disabled")


def _transcribe(audio):
    _assert_gpu(stt.model)
    with torch.inference_mode():
        return transcribe_stt(stt, audio)


def _tokens(messages, gen_kwargs, deadline=None, abandoned_path=None):
    _assert_gpu(llm_model)
    stop = Event()

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return (
                stop.is_set()
                or (deadline is not None and time.monotonic() >= deadline)
                or (abandoned_path is not None and abandoned_path.exists())
            )

    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.2)
    errors = []
    options = {
        key: value
        for key, value in gen_kwargs.items()
        if key in {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty"}
    }
    options.setdefault("max_new_tokens", 128)
    options.setdefault("do_sample", False)

    def generate():
        try:
            with torch.inference_mode():
                llm_model.generate(
                    **inputs,
                    **options,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([Deadline()]),
                    pad_token_id=llm_tokenizer.eos_token_id,
                )
        except BaseException as exc:
            errors.append(exc)
            streamer.end()

    thread = Thread(target=generate, daemon=True)
    thread.start()
    try:
        while True:
            try:
                chunk = next(streamer)
            except Empty:
                if not thread.is_alive():
                    break
                continue
            except StopIteration:
                break
            if chunk:
                yield chunk
        if errors:
            raise errors[0]
    finally:
        stop.set()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("GPU generation thread did not stop")


def _gpu_models():
    return {"stt": stt.model, "llm": llm_model, **tts.gpu_models()}


LEASE_ROOT = Path("/tmp/s2s-zerogpu-leases")
LEASE_ROOT.mkdir(mode=0o700, exist_ok=True)
LEASE_ADMISSION_SECONDS = 65.0
LEASE_IDLE_SECONDS = 15.0
log = logging.getLogger("s2s.zerogpu.executor")


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def _lease_directory(lease_id):
    if not re.fullmatch(r"[0-9a-f]{32}", lease_id):
        raise ValueError("Invalid lease identifier")
    return LEASE_ROOT / lease_id


@spaces.GPU(duration=90)
def lease_worker(lease_id):
    """Bounded inference executor, with no session/history/turn state in child.

    File IPC is private to this container. It avoids carrying Python locks or
    live handler state over ZeroGPU's fork boundary. Each model gets one lane;
    the LLM lane can stream while the independent TTS lane speaks a sentence.
    """
    directory = _lease_directory(lease_id)
    if not directory.is_dir():
        raise ValueError("Lease directory does not exist")
    for model in _gpu_models().values():
        _assert_gpu(model)
    started = time.monotonic()
    # CUDA graph capture must be exclusive across this entire process. Once it
    # completes, independent STT/LLM/TTS lanes may safely execute concurrently.
    graph_started = time.monotonic()
    tts.initialize_lease()
    log.info("lease %s TTS graph initialization %.4fs", lease_id, time.monotonic() - graph_started)
    # First connection explicitly waits here before listening. These are real
    # GPU operations, including codec decoding, not just model/device loading.
    for name, warm in (
        ("STT", lambda: warmup_stt(stt)),
        (
            "LLM",
            lambda: list(
                _tokens(
                    [{"role": "user", "content": "Say hello."}],
                    {"max_new_tokens": 8, "do_sample": False},
                    deadline=started + 83,
                )
            ),
        ),
        ("TTS", tts.warmup),
    ):
        warmup_started = time.monotonic()
        warm()
        log.info("lease %s %s warmup %.4fs", lease_id, name, time.monotonic() - warmup_started)
        if time.monotonic() - started >= LEASE_ADMISSION_SECONDS:
            raise TimeoutError("Model preparation exceeded GPU lease admission window")
    last_activity = time.monotonic()
    locks = {kind: Lock() for kind in ("transcribe", "tokens", "synthesize")}
    threads = []
    seen = set()
    _atomic_json(
        directory / "ready.json",
        {"gpu": torch.cuda.get_device_name(), "ready_at": time.time(), "models": ["stt", "llm", "tts"]},
    )
    log.info("lease %s ready on %s", lease_id, torch.cuda.get_device_name())

    def execute(request_path):
        ident = request_path.name.split(".")[0]
        request = json.loads(request_path.read_text())
        operation = request["operation"]
        payload = request["payload"]
        events_path = directory / (ident + ".events")
        abandoned = directory / (ident + ".abandoned")
        with events_path.open("a", buffering=1) as events:

            def emit(kind, value=None):
                events.write(json.dumps({"kind": kind, "value": value}) + "\n")
                events.flush()

            try:
                with locks[operation]:
                    if abandoned.exists():
                        emit("end")
                        return
                    inference_start = time.monotonic()
                    log.info("lease %s command %s %s inference_start", lease_id, ident, operation)
                    if operation == "transcribe":
                        audio = np.load(directory / (ident + ".input.npy"), allow_pickle=False)
                        emit("result", _transcribe(audio))
                    elif operation == "tokens":
                        for token in _tokens(
                            payload["messages"],
                            payload["gen_kwargs"],
                            started + 83,
                            abandoned,
                        ):
                            emit("text", token)
                    elif operation == "synthesize":
                        for number, chunk in enumerate(
                            tts.synthesize_chunks(payload["text"], payload.get("language"), payload.get("speaker"))
                        ):
                            if abandoned.exists():
                                break
                            filename = ident + "." + str(number) + ".output.npy"
                            np.save(directory / filename, chunk, allow_pickle=False)
                            emit("audio", filename)
                    log.info(
                        "lease %s command %s %s inference_end %.4fs",
                        lease_id,
                        ident,
                        operation,
                        time.monotonic() - inference_start,
                    )
                emit("end")
            except BaseException as exc:
                emit("error", f"{type(exc).__name__}: {exc}")
                emit("end")

    try:
        yield {"lease": lease_id, "status": "ready"}
        last_heartbeat = started
        while time.monotonic() - started < LEASE_ADMISSION_SECONDS:
            if (directory / "stop").exists():
                break
            requests = sorted(directory.glob("*.request.json"))
            for request in requests:
                if request.name in seen:
                    continue
                seen.add(request.name)
                # Acceptance is published synchronously before dispatch so the
                # parent can never retry a command whose thread is starting.
                (directory / (request.name.split(".")[0] + ".accepted")).touch()
                thread = Thread(target=execute, args=(request,), daemon=True)
                threads.append(thread)
                thread.start()
                last_activity = time.monotonic()
            active = any(thread.is_alive() for thread in threads)
            if active:
                last_activity = time.monotonic()
            elif time.monotonic() - last_activity >= LEASE_IDLE_SECONDS:
                break
            if time.monotonic() - last_heartbeat >= 1:
                yield {"lease": lease_id, "status": "active" if active else "idle"}
                last_heartbeat = time.monotonic()
            time.sleep(0.01)
    finally:
        # Mark closed before draining: unaccepted commands can safely move to a
        # new allocation, but accepted commands must never be replayed.
        (directory / "closed").touch()
        for thread in threads:
            thread.join(timeout=max(0, 85 - (time.monotonic() - started)))
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("Inference exceeded bounded GPU allocation")
        log.info("lease %s released after %.3fs", lease_id, time.monotonic() - started)


class LeaseBroker:
    """Host-only broker. Several pipeline threads share one live allocation."""

    def __init__(self, runner=None, base_directory=None, wait_timeout=180):
        self.runner = runner or lease_worker
        self.base_directory = Path(base_directory or LEASE_ROOT)
        self.base_directory.mkdir(mode=0o700, exist_ok=True)
        self.wait_timeout = wait_timeout
        self.lock = Lock()
        self.current = None
        self.leases_started = 0

    def _ensure_lease(self):
        with self.lock:
            if self.current is not None and not (self.current / "closed").exists():
                return self.current
            ident = uuid4().hex
            directory = self.base_directory / ident
            directory.mkdir(mode=0o700)
            self.current = directory
            self.leases_started += 1
            Thread(target=self._drive, args=(directory,), daemon=True).start()
            return directory

    def _drive(self, directory):
        started = time.monotonic()
        log.info("lease %s requested", directory.name)
        try:
            for event in self.runner(directory.name):
                if event.get("status") == "ready":
                    log.info("lease %s allocation_ready %.4fs", directory.name, time.monotonic() - started)
        except BaseException as exc:
            _atomic_json(directory / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            (directory / "closed").touch()
            (directory / "finished").touch()

    def prepare(self):
        """Wait for actual model inference warmup in the shared allocation."""
        directory = self._ensure_lease()
        started = time.monotonic()
        while True:
            failure = directory / "failure.json"
            if failure.exists():
                raise RuntimeError(json.loads(failure.read_text())["error"])
            if (directory / "closed").exists():
                raise RuntimeError("GPU allocation ended before models were ready")
            ready = directory / "ready.json"
            if ready.exists():
                return json.loads(ready.read_text())
            if time.monotonic() - started >= self.wait_timeout:
                (directory / "stop").touch()
                raise TimeoutError("Timed out preparing models on ZeroGPU")
            time.sleep(0.01)

    def submit(self, operation, payload, *, cancelled=None):
        """Poll host cancellation without sending callbacks across the fork boundary."""
        if operation not in ("transcribe", "tokens", "synthesize"):
            raise ValueError("Unknown GPU operation")
        started = time.monotonic()
        for attempt in range(3):
            if cancelled is not None and cancelled():
                return
            directory = self._ensure_lease()
            ident = uuid4().hex
            request_payload = dict(payload)
            if operation == "transcribe":
                np.save(directory / (ident + ".input.npy"), request_payload.pop("audio"), allow_pickle=False)
            request = directory / (ident + ".request.json")
            _atomic_json(request, {"operation": operation, "payload": request_payload})
            cursor = 0
            pending = ""
            completed = False
            try:
                while True:
                    if cancelled is not None and cancelled():
                        return
                    events = directory / (ident + ".events")
                    if events.exists():
                        with events.open() as stream:
                            stream.seek(cursor)
                            pending += stream.read()
                            cursor = stream.tell()
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            event = json.loads(line)
                            if event["kind"] == "error":
                                raise RuntimeError(event["value"])
                            if event["kind"] == "end":
                                completed = True
                                log.info(
                                    "lease %s command %s %s host_complete %.4fs",
                                    directory.name,
                                    ident,
                                    operation,
                                    time.monotonic() - started,
                                )
                                return
                            if event["kind"] == "audio" and operation == "synthesize":
                                event["value"] = np.load(directory / event["value"], allow_pickle=False)
                            if cancelled is not None and cancelled():
                                return
                            yield event
                    failure = directory / "failure.json"
                    if failure.exists():
                        raise RuntimeError(json.loads(failure.read_text())["error"])
                    if (directory / "closed").exists() and not (directory / (ident + ".accepted")).exists():
                        break  # Worker retired without starting this operation.
                    if (directory / "finished").exists():
                        # Events may have been written between the read above
                        # and the final marker. Read them before declaring loss.
                        if events.exists() and events.stat().st_size > cursor:
                            continue
                        raise RuntimeError("GPU worker ended before completing an accepted operation")
                    if time.monotonic() - started > self.wait_timeout:
                        raise TimeoutError("Timed out waiting for shared ZeroGPU allocation")
                    time.sleep(0.005)
            finally:
                # Unlike cancellation UUIDs this tombstone is not removed by
                # the caller. A timed-out request cannot start later unnoticed.
                if not completed:
                    (directory / (ident + ".abandoned")).touch()
        raise RuntimeError("GPU allocation repeatedly expired before accepting operation")


_broker = LeaseBroker()


def transcribe(audio):
    result = None
    for event in _broker.submit("transcribe", {"audio": np.asarray(audio, dtype=np.float32)}):
        if event["kind"] == "result":
            result = event["value"]
    if result is not None:
        return result
    raise RuntimeError("GPU transcription returned no result")


def tokens(messages, gen_kwargs, *, cancelled=None):
    for event in _broker.submit("tokens", {"messages": messages, "gen_kwargs": gen_kwargs}, cancelled=cancelled):
        if event["kind"] == "text":
            yield event["value"]


def synthesize_chunks(text, language=None, speaker=None, *, cancelled=None):
    for event in _broker.submit(
        "synthesize", {"text": text, "language": language, "speaker": speaker}, cancelled=cancelled
    ):
        if event["kind"] == "audio":
            yield event["value"]


def prefetch():
    """Acquire asynchronously on native speech start, before final audio exists."""
    _broker._ensure_lease()


def prepare():
    """Blocking connection readiness; reuse the same lease for live speech."""
    return _broker.prepare()


def stop():
    """Release idle capacity on host shutdown without allocating a new lease."""
    with _broker.lock:
        directory = _broker.current
        if directory is not None:
            (directory / "stop").touch()
