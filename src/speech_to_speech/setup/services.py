from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import httpx

from speech_to_speech.setup.models import ManagedService


def available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def endpoint_ready(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/models", timeout=0.25).status_code < 500
    except httpx.HTTPError:
        return False


@dataclass
class ManagedProcess:
    process: Any
    base_url: str
    log_handle: TextIO | None = None

    def stop(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        finally:
            if self.log_handle:
                self.log_handle.close()


class ManagedServiceRunner:
    def __init__(
        self,
        *,
        llama_server: str | Path,
        popen: Callable[..., Any] = subprocess.Popen,
        port_picker: Callable[[], int] = available_loopback_port,
        readiness: Callable[[str], bool] = endpoint_ready,
        sleep: Callable[[float], None] = time.sleep,
        log_path: Path | None = None,
        startup_timeout_s: float = 120,
    ) -> None:
        self.llama_server = str(llama_server)
        self._popen = popen
        self._port_picker = port_picker
        self._readiness = readiness
        self._sleep = sleep
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s

    def start(self, spec: ManagedService) -> ManagedProcess:
        port = self._port_picker()
        base_url = f"http://127.0.0.1:{port}/v1"
        model_arguments = ["-m", spec.model_path] if spec.model_path else ["-hf", spec.model]
        command = [
            self.llama_server,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *model_arguments,
            "-c",
            "16384",
            "-np",
            "1",
            "-fa",
            "on",
        ]
        log_handle = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_path.open("a", encoding="utf-8")
        try:
            process = self._popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=log_handle or subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            if log_handle:
                log_handle.close()
            raise
        managed = ManagedProcess(process, base_url, log_handle)
        for _ in range(max(1, int(self.startup_timeout_s / 0.1))):
            status = process.poll()
            if status is not None:
                if log_handle:
                    log_handle.close()
                raise RuntimeError(f"Managed llama.cpp exited with status {status} before becoming ready.")
            if self._readiness(base_url):
                return managed
            self._sleep(0.1)
        managed.stop()
        raise RuntimeError(f"Managed llama.cpp did not become ready within {self.startup_timeout_s:g} seconds.")
