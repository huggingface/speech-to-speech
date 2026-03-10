from typing import Literal, Optional

from pydantic import BaseModel


# ── Transcription events ─────────────────────

# This event is not supported by the OpenAI Realtime API, but we need for Reachy mini.
# We should check if it's still support in the future.
# Used ConversationItemInputAudioTranscriptionDeltaEvent object attributes for partial transcription.
# See: https://github.com/openai/openai-python/blob/main/src/openai/types/realtime/conversation_item_input_audio_transcription_delta_event.py

class ConversationItemInputAudioTranscriptionPartial(BaseModel):
    """
    Returned when the text value of an input audio transcription content part is updated with incremental transcription results.
    """

    event_id: str
    """The unique ID of the server event."""

    item_id: str
    """The ID of the item containing the audio that is being transcribed."""

    type: Literal["conversation.item.input_audio_transcription.partial"] = (
        "conversation.item.input_audio_transcription.partial"
    )

    content_index: Optional[int] = None
    """The index of the content part in the item's content array."""

    transcript: str
    """The text delta."""

# event that doesn't exist in the OpenAI Realtime API, but we need for Reachy mini:
# - response.completed
# - response.audio.delta -> response.output_audio.delta
# - conversation.item.input_audio_transcription.partial -> used conversation.item.input_audio_transcription.delta
# - response.audio.done -> response.output_audio.done
# - response.audio.completed
# - response.audio_transcript.done -> response.output_audio_transcript.done
