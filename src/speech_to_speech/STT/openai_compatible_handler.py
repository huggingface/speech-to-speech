from __future__ import annotations

import io
import json
import logging
import os
import wave
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

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


class TranscriptionRequestCancelled(RuntimeError):
    def __init__(self, request_id: str, reason: str) -> None:
        self.request_id = request_id
        self.reason = reason
        super().__init__(f"transcription request {request_id} cancelled: {reason}")


class TranscriptionRequestError(RuntimeError):
    """Sanitized HTTP/protocol failure safe to surface to a client."""


@dataclass(frozen=True)
class HttpTranscriptionResult:
    text: str
    language: str | None = None


class HttpTranscriptionOperation:
    """Exactly one transcription request and its transport lifecycle."""

    def __init__(
        self,
        *,
        request_id: str,
        endpoint_url: str,
        api_key: str | None,
        model: str | None,
        wav_bytes: bytes,
        language: str | None,
        response_format: str,
        timeout_s: float,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        self.request_id = request_id
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.wav_bytes = wav_bytes
        self.language = language
        self.response_format = response_format
        self.timeout_s = timeout_s
        self.extra_fields = extra_fields or {}
        self._cancelled = Event()
        self._transport_lock = Lock()
        self._client: httpx.Client | None = None
        self._response: httpx.Response | None = None
        self._cancel_reason: str | None = None

    def run(self, cancel_check: Callable[[], bool] | None = None) -> HttpTranscriptionResult:
        cancel_check = cancel_check or (lambda: False)
        if self._cancelled.is_set() or cancel_check():
            self.cancel("stale")
            raise self._cancellation_exception()

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data: dict[str, Any] = {"response_format": self.response_format, **self.extra_fields}
        if self.model:
            data["model"] = self.model
        if self.language:
            data["language"] = self.language

        client = httpx.Client(timeout=self.timeout_s)
        with self._transport_lock:
            cancelled_before_dispatch = self._cancelled.is_set() or cancel_check()
            if cancelled_before_dispatch:
                self._cancelled.set()
                self._cancel_reason = self._cancel_reason or "stale"
            else:
                self._client = client
        if cancelled_before_dispatch:
            client.close()
            raise self._cancellation_exception()

        monitor_stop = Event()
        monitor = Thread(
            target=self._monitor_cancellation,
            args=(cancel_check, monitor_stop),
            name="stt-http-cancel",
            daemon=True,
        )
        monitor.start()

        try:
            if self._cancelled.is_set() or cancel_check():
                self.cancel("stale")
                raise self._cancellation_exception()
            with client.stream(
                "POST",
                self.endpoint_url,
                headers=headers,
                data=data,
                files={"file": ("audio.wav", self.wav_bytes, "audio/wav")},
            ) as response:
                with self._transport_lock:
                    self._response = response
                if self._cancelled.is_set():
                    raise self._cancellation_exception()
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if self._cancelled.is_set():
                        raise self._cancellation_exception() from exc
                    raise TranscriptionRequestError(
                        f"transcription server returned HTTP {exc.response.status_code}"
                    ) from exc

                body = response.read()
                if self._cancelled.is_set() or cancel_check():
                    self.cancel("stale")
                    raise self._cancellation_exception()
                return self._parse_response(body, response.headers.get("content-type", ""))
        except TranscriptionRequestCancelled:
            raise
        except TranscriptionRequestError:
            raise
        except httpx.TimeoutException as exc:
            if self._cancelled.is_set():
                raise self._cancellation_exception() from exc
            raise TranscriptionRequestError("transcription request timed out") from exc
        except httpx.HTTPError as exc:
            if self._cancelled.is_set():
                raise self._cancellation_exception() from exc
            raise TranscriptionRequestError(f"transcription transport failed: {type(exc).__name__}") from exc
        except RuntimeError as exc:
            if self._cancelled.is_set():
                raise self._cancellation_exception() from exc
            raise
        finally:
            monitor_stop.set()
            monitor.join(timeout=0.2)
            with self._transport_lock:
                self._response = None
                self._client = None
            client.close()

    def cancel(self, reason: str) -> None:
        with self._transport_lock:
            if self._cancel_reason is None:
                self._cancel_reason = reason
            self._cancelled.set()
            response = self._response
            client = self._client
        if response is not None:
            try:
                response.close()
            except Exception:
                logger.debug("Error closing transcription response", exc_info=True)
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Error closing transcription client", exc_info=True)

    def _cancellation_exception(self) -> TranscriptionRequestCancelled:
        with self._transport_lock:
            reason = self._cancel_reason or "cancelled"
        return TranscriptionRequestCancelled(self.request_id, reason)

    def _monitor_cancellation(self, cancel_check: Callable[[], bool], stop: Event) -> None:
        while not stop.wait(0.025):
            if cancel_check():
                self.cancel("stale")
                return

    def _parse_response(self, body: bytes, content_type: str) -> HttpTranscriptionResult:
        if self.response_format == "text" or "text/plain" in content_type:
            return HttpTranscriptionResult(text=body.decode("utf-8").strip(), language=self.language)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranscriptionRequestError("transcription server returned an invalid JSON response") from exc
        text = payload.get("text")
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
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.response_format = response_format
        self.timeout = timeout
        self.speculative_turns = speculative_turns
        self.final_revision_settle_s = final_revision_settle_s
        self.gen_kwargs = gen_kwargs or {}
        self._state_lock = Lock()
        self._session_generation = 0
        self._active_operation: HttpTranscriptionOperation | None = None
        self._progressive_hypotheses: dict[tuple[int, str, int], str] = {}

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        request_id = uuid4().hex
        started_at_s = perf_counter()
        with self._state_lock:
            session_generation = self._session_generation
        operation = self._make_operation(request_id, vad_audio.audio)
        with self._state_lock:
            self._active_operation = operation

        def cancel_check() -> bool:
            return self.stop_event.is_set() or not self._is_request_relevant(vad_audio, session_generation)

        try:
            result = operation.run(cancel_check)
        except TranscriptionRequestCancelled:
            logger.debug(
                "OpenAI-compatible STT request cancelled turn=%s rev=%s mode=%s",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                vad_audio.mode,
            )
            return
        except Exception as exc:
            if not self._is_request_relevant(vad_audio, session_generation):
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
        finally:
            with self._state_lock:
                if self._active_operation is operation:
                    self._active_operation = None

        if cancel_check():
            operation.cancel("stale")
            return

        output: STTOut | None
        if vad_audio.mode == "progressive":
            output = self._progressive_delta(vad_audio, result.text, session_generation)
        else:
            self._clear_progressive_hypothesis(vad_audio, session_generation)
            output = Transcription(
                text=result.text,
                language_code=result.language,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
        if output is not None:
            yield output
            logger.info(
                "OpenAI-compatible STT completed turn=%s rev=%s mode=%s in %.3fs",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                vad_audio.mode,
                perf_counter() - started_at_s,
            )

    def _make_operation(self, request_id: str, audio: np.ndarray) -> HttpTranscriptionOperation:
        return HttpTranscriptionOperation(
            request_id=request_id,
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

    def _is_request_relevant(self, source: VADAudio, session_generation: int) -> bool:
        if self.stop_event.is_set():
            return False
        with self._state_lock:
            if session_generation != self._session_generation:
                return False
        tracker = self.speculative_turns
        return tracker is None or tracker.is_latest(source.turn_id, source.turn_revision)

    def _progressive_delta(
        self,
        source: VADAudio,
        hypothesis: str,
        session_generation: int,
    ) -> PartialTranscription | None:
        if source.turn_id is None or source.turn_revision is None:
            return PartialTranscription(text=hypothesis)
        key = (session_generation, source.turn_id, source.turn_revision)
        with self._state_lock:
            previous = self._progressive_hypotheses.get(key, "")
            if hypothesis == previous:
                return None
            if previous and not hypothesis.startswith(previous):
                logger.debug(
                    "Suppressing rewritten progressive STT hypothesis turn=%s rev=%s",
                    source.turn_id,
                    source.turn_revision,
                )
                return None
            self._progressive_hypotheses[key] = hypothesis
        return PartialTranscription(
            text=hypothesis[len(previous) :],
            turn_id=source.turn_id,
            turn_revision=source.turn_revision,
        )

    def _clear_progressive_hypothesis(self, source: VADAudio, session_generation: int) -> None:
        if source.turn_id is None or source.turn_revision is None:
            return
        with self._state_lock:
            self._progressive_hypotheses.pop((session_generation, source.turn_id, source.turn_revision), None)

    def on_session_end(self) -> None:
        with self._state_lock:
            self._session_generation += 1
            self._progressive_hypotheses.clear()
            operation = self._active_operation
        if operation is not None:
            operation.cancel("session_end")
        super().on_session_end()

    def cleanup(self) -> None:
        with self._state_lock:
            operation = self._active_operation
        if operation is not None:
            operation.cancel("shutdown")
