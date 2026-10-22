"""Loopback-only Reasoner fault injection for manual/live integration tests.

Do not deploy this helper publicly. Control applies to subsequent requests;
normal mode forwards real model generation without reading or logging content.
"""

import asyncio
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
state = {"delay_s": 0, "http_error": False, "interrupt_stream": False}


@app.post("/control")
async def control(request: Request):
    values = await request.json()
    state.update({key: values[key] for key in state if key in values})
    return state


@app.post("/v1/chat/completions")
async def completions(request: Request):
    mode = dict(state)
    payload = await request.body()
    await asyncio.sleep(float(mode["delay_s"]))
    if mode["http_error"]:
        return JSONResponse({"error": "Injected test failure"}, status_code=503)

    async def stream():
        headers = {
            "Authorization": "Bearer " + os.environ["CONVFILL_REASONER_API_KEY"],
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=45, transport=httpx.AsyncHTTPTransport(retries=2)) as client:
            async with client.stream(
                "POST", os.environ["CONVFILL_REAL_REASONER_URL"] + "/chat/completions", headers=headers, content=payload
            ) as response:
                if response.status_code != 200:
                    yield b'data: {"error":{"message":"upstream unavailable"}}\n\n'
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk
                    if mode["interrupt_stream"]:
                        return

    return StreamingResponse(stream(), media_type="text/event-stream")
