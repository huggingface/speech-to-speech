from __future__ import annotations

import io
import json
import logging
import os
import wave
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

import httpx
import numpy as np

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import (
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
OPENAI_BASE_URL = "https://api.openai.com/v1"


class TranscriptionRequestError(RuntimeError):
    """Sanitized HTTP/protocol failure safe to surface to a client."""


@dataclass(frozen=True)
class HttpTranscriptionResult:
    text: str
    language: str | None = None


@dataclass(frozen=True)
class HttpTranscriptionOperation:
    """One synchronous transcription request."""

    endpoint_url: str
    api_key: str | None
    model: str | None
    wav_bytes: bytes
    language: str | None
    response_format: str
    timeout_s: float
    extra_fields: dict[str, Any] | None = None

    def run(self) -> HttpTranscriptionResult:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data: dict[str, Any] = {
            "response_format": self.response_format,
            **(self.extra_fields or {}),
        }
        if self.model:
            data["model"] = self.model
        if self.language:
            data["language"] = self.language

        try:
            response = httpx.post(
                self.endpoint_url,
                headers=headers,
                data=data,
                files={"file": ("audio.wav", self.wav_bytes, "audio/wav")},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TranscriptionRequestError(f"transcription server returned HTTP {exc.response.status_code}") from exc
        except httpx.TimeoutException as exc:
            raise TranscriptionRequestError("transcription request timed out") from exc
        except httpx.HTTPError as exc:
            raise TranscriptionRequestError(f"transcription transport failed: {type(exc).__name__}") from exc

        return self._parse_response(response.content, response.headers.get("content-type", ""))

    def _parse_response(self, body: bytes, content_type: str) -> HttpTranscriptionResult:
        if self.response_format == "text" or "text/plain" in content_type:
            return HttpTranscriptionResult(text=body.decode("utf-8").strip(), language=self.language)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranscriptionRequestError("transcription server returned an invalid JSON response") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise TranscriptionRequestError("transcription response is missing a string 'text' field")
        language = payload.get("language")
        return HttpTranscriptionResult(
            text=text,
            language=language if isinstance(language, str) else self.language,
        )


class OpenAICompatibleSTTHandler(BaseSTTHandler):
    """Serial client handler for POST /v1/audio/transcriptions."""

    def setup(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model: str | None = "nvidia/parakeet-tdt-0.6b-v3",
        language: str | None = None,
        response_format: str = "json",
        timeout: float = 60.0,
        speculative_turns: SpeculativeTurnTracker | None = None,
        final_revision_settle_s: float = 0.0,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if response_format not in {"json", "text"}:
            raise ValueError("OpenAI-compatible STT response_format must be 'json' or 'text'")
        if timeout <= 0:
            raise ValueError("OpenAI-compatible STT timeout must be > 0")
        model = model.strip() if model else None
        language = language.strip() if language else None
        if model is None and language is None:
            raise ValueError("OpenAI-compatible STT requires either a model or language")

        self.base_url = base_url.rstrip("/")
        self.endpoint_url = f"{self.base_url}/audio/transcriptions"
        self.api_key = api_key
        if self.api_key is None and self.base_url == OPENAI_BASE_URL:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.response_format = response_format
        self.timeout = timeout
        self.speculative_turns = speculative_turns
        self.final_revision_settle_s = final_revision_settle_s
        self.gen_kwargs = gen_kwargs or {}

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if not self._is_request_relevant(vad_audio):
            return

        started_at_s = perf_counter()
        try:
            result = self._make_operation(vad_audio.audio).run()
        except Exception as exc:
            if not self._is_request_relevant(vad_audio):
                return
            message = str(exc) if isinstance(exc, TranscriptionRequestError) else "transcription request failed"
            if vad_audio.mode == "progressive":
                logger.warning(
                    "OpenAI-compatible progressive STT failed turn=%s rev=%s: %s",
                    vad_audio.turn_id,
                    vad_audio.turn_revision,
                    message,
                )
                return
            logger.error(
                "OpenAI-compatible STT failed turn=%s rev=%s: %s",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                message,
            )
            yield TranscriptionFailure(
                message=message,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
            return

        if not self._is_request_relevant(vad_audio):
            return

        if vad_audio.mode == "progressive":
            output: STTOut = PartialTranscription(
                text=result.text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            output = Transcription(
                text=result.text,
                language_code=result.language,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )

        yield output
        logger.info(
            "OpenAI-compatible STT completed turn=%s rev=%s mode=%s in %.3fs",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            vad_audio.mode,
            perf_counter() - started_at_s,
        )

    def _make_operation(self, audio: np.ndarray) -> HttpTranscriptionOperation:
        return HttpTranscriptionOperation(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            model=self.model,
            wav_bytes=self._encode_wav(audio),
            language=self.language,
            response_format=self.response_format,
            timeout_s=self.timeout,
            extra_fields=self.gen_kwargs,
        )

    @staticmethod
    def _encode_wav(audio: np.ndarray) -> bytes:
        waveform = np.asarray(audio).squeeze()
        if waveform.ndim != 1:
            raise ValueError(f"STT audio must be mono, got shape {waveform.shape}")
        if np.issubdtype(waveform.dtype, np.floating):
            pcm = np.clip(waveform, -1.0, 1.0)
            pcm = np.round(pcm * 32767.0).astype("<i2")
        else:
            pcm = np.clip(waveform, -32768, 32767).astype("<i2")

        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(PIPELINE_SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return output.getvalue()

    def _is_request_relevant(self, source: VADAudio) -> bool:
        if self.stop_event.is_set():
            return False
        tracker = self.speculative_turns
        return tracker is None or tracker.is_latest(source.turn_id, source.turn_revision)
