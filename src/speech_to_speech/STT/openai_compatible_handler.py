from __future__ import annotations

import io
import json
import logging
import os
import wave
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
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
from speech_to_speech.STT.endpoint_admission import (
    AdmissionRejected,
    CancellationReason,
    CancelTranscription,
    EndpointAdmissionLease,
    TranscriptionAdmissionRequest,
    TranscriptionCancelled,
    TranscriptionMode,
)

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000


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
        endpoint_url: str,
        api_key: str | None,
        model: str | None,
        wav_bytes: bytes,
        language: str | None,
        response_format: str,
        timeout_s: float,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.wav_bytes = wav_bytes
        self.language = language
        self.response_format = response_format
        self.timeout_s = timeout_s
        self._cancelled = Event()
        self._transport_lock = Lock()
        self._client: httpx.Client | None = None
        self._response: httpx.Response | None = None

    def run(self) -> HttpTranscriptionResult:
        if self._cancelled.is_set():
            raise TranscriptionCancelled("undispatched", "superseded")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = {"response_format": self.response_format}
        if self.model:
            data["model"] = self.model
        if self.language:
            data["language"] = self.language

        client = httpx.Client(timeout=self.timeout_s)
        with self._transport_lock:
            self._client = client

        try:
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
                    raise TranscriptionCancelled("active", "superseded")
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise TranscriptionRequestError(
                        f"transcription server returned HTTP {exc.response.status_code}"
                    ) from exc

                body = response.read()
                if self._cancelled.is_set():
                    raise TranscriptionCancelled("active", "superseded")
                return self._parse_response(body, response.headers.get("content-type", ""))
        except TranscriptionCancelled:
            raise
        except TranscriptionRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise TranscriptionRequestError("transcription request timed out") from exc
        except httpx.HTTPError as exc:
            raise TranscriptionRequestError(f"transcription transport failed: {type(exc).__name__}") from exc
        finally:
            with self._transport_lock:
                self._response = None
                self._client = None
            client.close()

    def cancel(self, reason: CancellationReason) -> None:
        del reason
        self._cancelled.set()
        with self._transport_lock:
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


@dataclass(frozen=True)
class _CompletedRequest:
    source: VADAudio
    future: Future[HttpTranscriptionResult]
    session_generation: int
    started_at_s: float


class OpenAICompatibleSTTHandler(BaseSTTHandler):
    """Asynchronous client handler for POST /v1/audio/transcriptions."""

    def setup(
        self,
        admission_lease: EndpointAdmissionLease,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model: str | None = "nvidia/parakeet-tdt-0.6b-v3",
        language: str | None = None,
        response_format: str = "json",
        timeout: float = 60.0,
        max_concurrency: int = 1,
        max_queue_size: int = 8,
        progressive_min_interval: float = 0.75,
        speculative_turns: SpeculativeTurnTracker | None = None,
        final_revision_settle_s: float = 0.0,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del max_concurrency, max_queue_size, progressive_min_interval, gen_kwargs
        if response_format not in {"json", "text"}:
            raise ValueError("OpenAI-compatible STT response_format must be 'json' or 'text'")
        if timeout <= 0:
            raise ValueError("OpenAI-compatible STT timeout must be > 0")
        model = model.strip() if model else None
        language = language.strip() if language else None
        if model is None and language is None:
            raise ValueError("OpenAI-compatible STT requires either a model or language")

        self.admission_lease = admission_lease
        self.admission = admission_lease.controller
        self.base_url = base_url.rstrip("/")
        self.endpoint_url = f"{self.base_url}/audio/transcriptions"
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.response_format = response_format
        self.timeout = timeout
        self.speculative_turns = speculative_turns
        self.final_revision_settle_s = final_revision_settle_s

        self._owner_id = uuid4().hex
        self._session_generation = 0
        self._generation_lock = Lock()
        self._completion_queue: Queue[_CompletedRequest | None] = Queue()
        self._delivery_thread = Thread(
            target=self._delivery_loop,
            name=f"openai-stt-delivery-{self._owner_id[:8]}",
            daemon=True,
        )
        self._delivery_thread.start()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        mode: TranscriptionMode = "progressive" if vad_audio.mode == "progressive" else "final"
        turn_id = vad_audio.turn_id or f"untagged-{uuid4().hex}"
        turn_revision = vad_audio.turn_revision if vad_audio.turn_revision is not None else 0
        request_id = uuid4().hex
        started_at_s = perf_counter()
        with self._generation_lock:
            session_generation = self._session_generation

        request = TranscriptionAdmissionRequest(
            request_id=request_id,
            owner_id=self._owner_id,
            turn_id=turn_id,
            turn_revision=turn_revision,
            mode=mode,
            operation_factory=lambda: self._make_operation(vad_audio.audio),
            is_relevant=lambda: self._is_request_relevant(vad_audio, session_generation),
        )
        future = self.admission.submit(request)
        completed = _CompletedRequest(
            source=vad_audio,
            future=future,
            session_generation=session_generation,
            started_at_s=started_at_s,
        )
        future.add_done_callback(lambda _future: self._completion_queue.put(completed))

        # Completion is delivered by _delivery_loop so this handler can keep
        # accepting and coalescing newer progressive windows while HTTP runs.
        yield from ()

    def _make_operation(self, audio: np.ndarray) -> HttpTranscriptionOperation:
        return HttpTranscriptionOperation(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            model=self.model,
            wav_bytes=self._encode_wav(audio),
            language=self.language,
            response_format=self.response_format,
            timeout_s=self.timeout,
        )

    def _encode_wav(self, audio: np.ndarray) -> bytes:
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
        with self._generation_lock:
            if session_generation != self._session_generation:
                return False
        tracker = self.speculative_turns
        if tracker is None:
            return True
        return tracker.is_latest(source.turn_id, source.turn_revision)

    def _delivery_loop(self) -> None:
        while True:
            try:
                completed = self._completion_queue.get(timeout=0.1)
            except Empty:
                continue
            if completed is None:
                return

            source = completed.source
            try:
                result = completed.future.result()
            except (TranscriptionCancelled, AdmissionRejected):
                logger.debug(
                    "OpenAI-compatible STT request cancelled turn=%s rev=%s mode=%s",
                    source.turn_id,
                    source.turn_revision,
                    source.mode,
                )
                continue
            except Exception as exc:
                if not self._is_request_relevant(source, completed.session_generation):
                    continue
                message = str(exc) if isinstance(exc, TranscriptionRequestError) else "transcription request failed"
                if source.mode == "progressive":
                    logger.warning(
                        "OpenAI-compatible progressive STT failed turn=%s rev=%s: %s",
                        source.turn_id,
                        source.turn_revision,
                        message,
                    )
                    continue
                logger.error(
                    "OpenAI-compatible STT failed turn=%s rev=%s: %s",
                    source.turn_id,
                    source.turn_revision,
                    message,
                )
                failure = TranscriptionFailure(
                    message=message,
                    turn_id=source.turn_id,
                    turn_revision=source.turn_revision,
                    speech_stopped_at_s=source.created_at_s,
                )
                if self.should_emit_output(failure):
                    self.queue_out.put(failure)
                continue

            with self._generation_lock:
                if completed.session_generation != self._session_generation:
                    continue
            output: STTOut
            if source.mode == "progressive":
                output = PartialTranscription(
                    text=result.text,
                    turn_id=source.turn_id,
                    turn_revision=source.turn_revision,
                )
            else:
                output = Transcription(
                    text=result.text,
                    language_code=result.language,
                    turn_id=source.turn_id,
                    turn_revision=source.turn_revision,
                    speech_stopped_at_s=source.created_at_s,
                )

            if not self.should_emit_output(output):
                continue
            elapsed_s = perf_counter() - completed.started_at_s
            logger.info(
                "OpenAI-compatible STT completed turn=%s rev=%s mode=%s in %.3fs",
                source.turn_id,
                source.turn_revision,
                source.mode,
                elapsed_s,
            )
            self.before_emit_output(output)
            self.queue_out.put(output)

    def on_session_end(self) -> None:
        with self._generation_lock:
            self._session_generation += 1
        self.admission.cancel(CancelTranscription(owner_id=self._owner_id, reason="session_end"))
        super().on_session_end()

    def cleanup(self) -> None:
        self.admission.cancel(CancelTranscription(owner_id=self._owner_id, reason="shutdown"))
        self._completion_queue.put(None)
        self._delivery_thread.join(timeout=2.0)
        self.admission_lease.release()
