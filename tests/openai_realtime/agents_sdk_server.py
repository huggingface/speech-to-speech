"""Deterministic local server used by the official Agents SDK compatibility tests."""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from threading import Event

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from openai.types.realtime.realtime_conversation_item_function_call import (
    RealtimeConversationItemFunctionCall,
)

from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    AssistantTextPart,
    AssistantToolCallPart,
    AudioOutput,
)

from .test_openai_client import _ServerEnv


def _response_key(env: _ServerEnv, prefix: str) -> str:
    return f"{prefix}_{time.monotonic_ns()}"


def _audio(samples: int = 3_200) -> bytes:
    wave = np.resize(np.array([8_000, -8_000], dtype=np.int16), samples)
    return wave.tobytes()


def _queue_completed_response(env: _ServerEnv, response_key: str, text: str) -> None:
    env.output_queue.put(
        AssistantOutputEvent(
            response_key=response_key,
            parts=[AssistantTextPart(text=text)],
        )
    )
    env.output_queue.put(AudioOutput(audio=_audio(), response_key=response_key))
    env.output_queue.put(AssistantResponseDoneEvent(response_key=response_key))
    env.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=response_key))


def main() -> None:
    env = _ServerEnv()
    sdk_bundle = (
        Path(__file__).parents[2]
        / "demo/node_modules/@openai/agents-realtime/dist/bundle/openai-realtime-agents.umd.js"
    )

    @env.app.get("/test/sdk-page", response_class=HTMLResponse)
    async def sdk_page():
        return '<!doctype html><html><body><script src="/test/sdk.js"></script></body></html>'

    @env.app.get("/test/sdk.js")
    async def sdk_javascript():
        if not sdk_bundle.is_file():
            raise HTTPException(status_code=503, detail="Run npm ci in demo/")
        return FileResponse(sdk_bundle, media_type="text/javascript")

    @env.app.get("/test/state")
    async def state():
        if not env.service.connection_ids:
            return {"connected": False, "inputChunks": env.input_queue.qsize()}
        conn_id = env.service.connection_ids[0]
        st = env.service._state(conn_id)
        session = st.runtime_config.session
        audio_input = session.audio.input
        audio_output = session.audio.output
        outputs = [
            getattr(item, "output", None)
            for item in st.runtime_config.chat.buffer
            if getattr(item, "type", None) == "function_call_output"
        ]
        return {
            "connected": True,
            "inputChunks": env.input_queue.qsize(),
            "instructions": session.instructions,
            "voice": session.audio.output.voice,
            "turnDetection": getattr(audio_input.turn_detection, "type", None),
            "inputRate": getattr(audio_input.format, "rate", None),
            "outputRate": getattr(audio_output.format, "rate", None),
            "tools": [tool.name for tool in session.tools],
            "toolOutputs": outputs,
            "responseRequests": env.text_prompt_queue.qsize(),
            "inResponse": st.in_response,
        }

    @env.app.post("/test/voice")
    async def voice():
        if not env.service.connection_ids:
            raise HTTPException(status_code=409, detail="No SDK session")
        env.text_output_queue.put(SpeechStartedEvent())
        env.text_output_queue.put(PartialTranscriptionEvent(delta="hello"))
        env.text_output_queue.put(SpeechStoppedEvent())
        env.text_output_queue.put(TranscriptionCompletedEvent(transcript="hello"))
        _queue_completed_response(env, _response_key(env, "voice"), "Hello from the compatibility test.")
        return {"ok": True}

    @env.app.post("/test/start-audio")
    async def start_audio():
        if not env.service.connection_ids:
            raise HTTPException(status_code=409, detail="No SDK session")
        response_key = _response_key(env, "interrupt")
        env.output_queue.put(
            AssistantOutputEvent(
                response_key=response_key,
                parts=[AssistantTextPart(text="This response will be interrupted.")],
            )
        )
        env.output_queue.put(AudioOutput(audio=_audio(16_000), response_key=response_key))
        return {"ok": True}

    @env.app.post("/test/settle-audio")
    async def settle_audio():
        # Mimic the cancelled pipeline generation's terminal. It is unkeyed so
        # the already-closed public response cannot discard the bookkeeping
        # sentinel before CancelScope observes it.
        env.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE))
        return {"ok": True}

    @env.app.post("/test/barge-in")
    async def barge_in():
        if not env.service.connection_ids:
            raise HTTPException(status_code=409, detail="No SDK session")
        env.text_output_queue.put(SpeechStartedEvent())
        return {"ok": True}

    @env.app.post("/test/tool")
    async def tool_call():
        if not env.service.connection_ids:
            raise HTTPException(status_code=409, detail="No SDK session")
        response_key = _response_key(env, "tool")
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id="call_agents_sdk",
            name="lookup",
            arguments='{"query":"sdk"}',
        )
        conn_id = env.service.connection_ids[0]
        env.service._state(conn_id).runtime_config.chat.add_item(call)
        env.text_output_queue.put(
            AssistantOutputEvent(
                response_key=response_key,
                parts=[
                    AssistantToolCallPart(
                        tool={
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                    )
                ],
            )
        )
        env.output_queue.put(AssistantResponseDoneEvent(response_key=response_key))
        env.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=response_key))
        return {"ok": True}

    @env.app.post("/test/finish-response")
    async def finish_response():
        if not env.service.connection_ids:
            raise HTTPException(status_code=409, detail="No SDK session")
        conn_id = env.service.connection_ids[0]
        response_key = env.service._state(conn_id).current_response_key
        if response_key is None:
            raise HTTPException(status_code=409, detail="No active response")
        _queue_completed_response(env, response_key, "Tool result received.")
        return {"ok": True}

    stopped = Event()

    def stop(*_args: object) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env.start()
    print(
        json.dumps({"http": f"http://127.0.0.1:{env.port}", "ws": f"ws://127.0.0.1:{env.port}/v1/realtime"}), flush=True
    )
    try:
        stopped.wait()
    finally:
        env.stop()


if __name__ == "__main__":
    main()
