"""MiniMax t2a_v2 SSE adapter using the pipeline's cancellable HTTP transport."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from math import isfinite
from threading import Event
from typing import Any

import httpx
import numpy as np

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.openai_compatible_handler import (
    PIPELINE_SAMPLE_RATE,
    HttpSpeechOperation,
    OpenAICompatibleTTSHandler,
    SpeechRequestError,
)

logger = logging.getLogger(__name__)


class MiniMaxSpeechOperation(HttpSpeechOperation):
    def _validate_content_type(self, response: httpx.Response) -> None:
        media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        # MiniMax documents streaming responses under both media types.
        # The decoder still requires SSE data frames and a terminal status.
        if media_type not in {"text/event-stream", "application/json"}:
            raise SpeechRequestError("MiniMax endpoint returned a non-SSE response")


class MiniMaxTTSHandler(OpenAICompatibleTTSHandler):
    """Reuse remote TTS cancellation, response identity, and PCM block handling.

    Only the request schema and SSE-to-PCM decoding differ from the existing
    HTTP speech adapter. No model download or paid warmup request is needed.
    """

    # Provider setup options differ; processing and transport hooks are shared.
    def setup(  # type: ignore[override]
        self,
        should_listen: Event,
        api_key: str | None = None,
        base_url: str = "https://api.minimax.io",
        model: str = "speech-2.8-hd",
        voice: str = "English_Graceful_Lady",
        speed: float = 1.0,
        vol: float = 1.0,
        pitch: int = 0,
        blocksize: int = 512,
        timeout: float = 30.0,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        api_key = api_key or os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MiniMax API key is required. Set MINIMAX_API_KEY or --minimax_tts_api_key.")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("MiniMax speed must be between 0.5 and 2.0")
        if not 0 < vol <= 10:
            raise ValueError("MiniMax volume must be in (0, 10]")
        if not -12 <= pitch <= 12:
            raise ValueError("MiniMax pitch must be between -12 and 12")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("MiniMax timeout must be finite and > 0")
        self.vol = vol
        self.pitch = pitch
        # The parent stream flag describes a vLLM-specific request extension.
        # MiniMax sets its own stream flag in _request_payload below.
        super().setup(
            should_listen,
            base_url=base_url,
            api_key=api_key,
            model=model,
            voice=voice,
            speed=speed,
            sample_rate=PIPELINE_SAMPLE_RATE,
            timeout=timeout,
            blocksize=blocksize,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
            gen_kwargs=gen_kwargs,
        )
        self.endpoint_url = f"{self.base_url}/v1/t2a_v2"

    def warmup(self) -> None:
        logger.info("MiniMax TTS handler ready (model=%s, voice=%s)", self.model, self.voice)

    def _request_payload(
        self,
        *,
        text: str,
        voice: str | dict[str, str],
        language: str | None = None,
        use_setup_language: bool = True,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": voice if isinstance(voice, str) else voice["id"],
                "speed": self.speed,
                "vol": self.vol,
                "pitch": self.pitch,
            },
            "audio_setting": {"sample_rate": PIPELINE_SAMPLE_RATE, "format": "pcm", "channel": 1},
            "stream_options": {"exclude_aggregated_audio": True},
        }

    def _make_operation(
        self,
        *,
        text: str,
        voice: str | dict[str, str],
        language: str | None = None,
        use_setup_language: bool = True,
    ) -> HttpSpeechOperation:
        return MiniMaxSpeechOperation(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            payload=self._request_payload(text=text, voice=voice),
            timeout_s=self.timeout,
        )

    def _decode_pcm_stream(
        self,
        encoded_chunks: Iterator[bytes],
        *,
        on_first_source_audio: Callable[[], None] | None = None,
    ) -> Iterator[np.ndarray]:
        yield from super()._decode_pcm_stream(
            self._sse_audio_bytes(encoded_chunks), on_first_source_audio=on_first_source_audio
        )

    @staticmethod
    def _sse_audio_bytes(encoded_chunks: Iterator[bytes]) -> Iterator[bytes]:
        buffer = b""
        try:
            for chunk in encoded_chunks:
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        event = json.loads(payload)
                        base_resp = event.get("base_resp") or {}
                        status_code = base_resp.get("status_code", 0)
                        if status_code not in (0, None):
                            raise SpeechRequestError(f"MiniMax API error (status {status_code})")
                        data = event.get("data") or {}
                        if data.get("status") == 2:
                            # Aggregated audio is excluded; never replay the final packet.
                            return
                        if data.get("status") == 1 and data.get("audio"):
                            yield bytes.fromhex(data["audio"])
                    except (ValueError, TypeError, AttributeError):
                        raise SpeechRequestError("MiniMax endpoint returned an invalid SSE event") from None
            raise SpeechRequestError("MiniMax stream ended before synthesis completed")
        finally:
            close = getattr(encoded_chunks, "close", None)
            if close is not None:
                close()

    def cleanup(self) -> None:
        if hasattr(self, "_operation_lock"):
            super().cleanup()
