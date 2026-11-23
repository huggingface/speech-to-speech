"""Stand-in for the OpenAI-compatible STT and TTS services, for text-only LLM evals.

It answers ``POST /v1/audio/speech`` with a short stretch of silent 24 kHz PCM and
``POST /v1/audio/transcriptions`` with an empty transcript, so a server started with
``--stt openai --tts openai`` pointing here runs with only its LLM doing real work.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_SILENCE = b"\x00\x00" * 2400  # 0.1 s at 24 kHz, 16-bit mono


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:  # quiet
        pass

    def _reply(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("content-length") or 0))
        if self.path.endswith("/audio/speech"):
            self._reply(200, "audio/pcm", _SILENCE)
        elif self.path.endswith("/audio/transcriptions"):
            self._reply(200, "application/json", json.dumps({"text": ""}).encode())
        else:
            self._reply(404, "application/json", b"{}")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._reply(200, "application/json", b'{"object": "list", "data": []}')


def serve(host: str = "127.0.0.1", port: int = 8090) -> None:
    ThreadingHTTPServer((host, port), _Handler).serve_forever()
