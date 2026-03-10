from typing import Literal

class RuntimeConfig:
    """
    Shared mutable configuration written by the RealtimeService on
    session.update and read by pipeline handlers (VAD, LLM, TTS) during
    processing.  Python's GIL makes simple attribute reads/writes atomic,
    so no explicit locking is needed for primitive values.
    """

    def __init__(self):
        # Session-level config (persistent until next session.update)
        self.voice: str | None = None
        self.instructions: str | None = None
        self.turn_detection: dict | None = None
        self.tools: list | None = None
        self.tool_choice: Literal["auto", "required", "none"] | None = None
        self.input_audio_transcription: dict | None = None

        # Client audio sample rate: OpenAI Realtime API standard is 24kHz,
        # but the internal pipeline operates at 16kHz.  The service layer
        # resamples between these rates transparently.
        self.client_audio_rate: int = 24000

        # Per-response overrides (consumed once per response, then cleared)
        self.response_instructions: str | None = None
        self.response_tool_choice: str | None = None