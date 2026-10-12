"""Minimal LiveKit Agents voice agent backed by a local speech-to-speech server.

speech-to-speech serves the OpenAI Realtime API, so LiveKit's OpenAI Realtime
plugin can use it as the realtime model by pointing ``base_url`` at it.
"""

import os

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import openai
from openai.types.realtime import AudioTranscription
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

S2S_BASE_URL = os.getenv("S2S_BASE_URL", "http://127.0.0.1:8765/v1")
S2S_VOICE = os.getenv("S2S_VOICE", "Aiden")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice assistant. Keep answers short and conversational.")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession[None] = AgentSession(
        llm=openai.realtime.RealtimeModel(
            base_url=S2S_BASE_URL,
            api_key="not-needed",
            voice=S2S_VOICE,
            modalities=["audio"],
            # speech-to-speech runs its own VAD; these are the fields it reads.
            turn_detection=ServerVad(
                type="server_vad",
                create_response=True,
                interrupt_response=True,
                silence_duration_ms=500,
            ),
            # The server always transcribes with its configured STT; the model name is ignored.
            input_audio_transcription=AudioTranscription(model="local"),
            input_audio_noise_reduction=None,
            max_session_duration=None,
        )
    )
    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
