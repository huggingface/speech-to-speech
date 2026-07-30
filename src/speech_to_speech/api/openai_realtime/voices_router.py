"""Session-scoped HTTP routes for the cloned-voice library.

Mounted by ``create_app`` next to the realtime WebSocket route. The GET
route doubles as the capability probe: 200 means the backend supports voice
cloning for this session, 409 means it does not (no voice store is wired —
the pipeline's TTS is not Qwen3). No custom fields are added to any OpenAI
Realtime event; capability is discovered by probing these routes.

Voices are selected afterwards with a plain ``session.update`` carrying the
voice id in the session audio output voice field.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.voice_store import VoiceStore, VoiceSyncError, VoiceValidationError

logger = logging.getLogger(__name__)


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    created_at: str


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]


class VoiceRouteError(BaseModel):
    message: str
    code: str


class VoiceRouteErrorResponse(BaseModel):
    error: VoiceRouteError


def _error_response(status_code: int, message: str, code: str) -> JSONResponse:
    body = VoiceRouteErrorResponse(error=VoiceRouteError(message=message, code=code))
    return JSONResponse(content=body.model_dump(), status_code=status_code)


def build_voices_router(pool: list[PipelineUnit], voice_store: Optional[VoiceStore]) -> APIRouter:
    router = APIRouter()

    def _gate(session_id: str) -> Optional[JSONResponse]:
        """Session addressing + capability gating shared by both routes."""
        for unit in pool:
            session = unit.session
            if session is not None and session.session_id == session_id and session.released_at is None:
                break
        else:
            return _error_response(404, "Unknown session", "unknown_session")
        if voice_store is None:
            return _error_response(
                409,
                "Voice cloning is not available on this server's TTS backend.",
                "voice_cloning_unsupported",
            )
        return None

    @router.get("/v1/realtime/sessions/{session_id}/voices")
    async def list_voices(session_id: str) -> JSONResponse:
        err = _gate(session_id)
        if err is not None:
            return err
        assert voice_store is not None
        records = await run_in_threadpool(voice_store.list_voices)
        body = VoiceListResponse(
            voices=[VoiceInfo(voice_id=r.voice_id, name=r.name, created_at=r.created_at) for r in records]
        )
        return JSONResponse(content=body.model_dump(), status_code=200)

    @router.post("/v1/realtime/sessions/{session_id}/voices")
    async def create_voice(
        session_id: str,
        audio: UploadFile = File(...),
        ref_text: str = Form(...),
        name: str = Form(...),
    ) -> JSONResponse:
        err = _gate(session_id)
        if err is not None:
            return err
        assert voice_store is not None
        audio_bytes = await audio.read()
        try:
            # Decode/resample/store off the event loop: this loop also carries
            # every live conversation's audio.
            record = await run_in_threadpool(voice_store.add_voice, audio_bytes, ref_text=ref_text, name=name)
        except VoiceValidationError as e:
            return _error_response(e.status_code, str(e), e.code)
        except VoiceSyncError as e:
            # The store rolled the local copy back; nothing was created.
            return _error_response(502, str(e), "voice_store_sync_failed")
        body = VoiceInfo(voice_id=record.voice_id, name=record.name, created_at=record.created_at)
        return JSONResponse(content=body.model_dump(), status_code=201)

    return router
