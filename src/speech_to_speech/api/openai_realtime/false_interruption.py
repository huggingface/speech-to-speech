"""Barge-in that waits for words: coughs and backchannels do not cancel a reply.

With ``turn_detection.false_interruption`` set, user speech that starts while a
reply is active, queued, or still playing does not cancel that reply right
away. The router holds the new turn's pipeline events instead, and the reply
keeps generating. The interruption is committed once the user's transcript has
``min_words`` words that are not fillers or backchannels ("uh", "mm-hmm",
"yeah"): the held events then run through the ordinary barge-in path, so the
client sees exactly what an immediate interruption would have produced, only
later. If the turn ends empty, filler-only, or untranscribable, its events are
dropped. The client never hears about that turn, nothing enters the
conversation, and the reply carries on (or resumes, when playback was paused).
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

if TYPE_CHECKING:
    # RuntimeConfig imports this module, and the pipeline package imports RuntimeConfig.
    from speech_to_speech.pipeline.events import PipelineEvent, SpeechStartedEvent

INTERRUPTION_EVENTS_EXTENSION = "speech_to_speech.interruption"

PlaybackMode = Literal["continue", "pause", "duck"]
PLAYBACK_MODES: tuple[PlaybackMode, ...] = ("continue", "pause", "duck")

# Words and phrases that never count toward an interruption on their own.
# Hyphens and punctuation are normalized away, so "mm-hmm" matches "mm hmm".
DEFAULT_IGNORE_WORDS: tuple[str, ...] = (
    # Hesitations and noises an STT writes for coughs and throat clearing.
    "uh",
    "uhh",
    "um",
    "umm",
    "uhm",
    "er",
    "erm",
    "ah",
    "ahh",
    "eh",
    "oh",
    "ooh",
    "hm",
    "hmm",
    "hmmm",
    "mm",
    "mmm",
    "mhm",
    "mm hmm",
    "mm hm",
    "uh huh",
    "huh",
    "aha",
    "ha",
    # Throat clearing, as STT models spell it.
    "ahem",
    "a hem",
    "a ham",
    "uh hum",
    "hem",
    # Backchannels: the listener is following, not taking the turn.
    "yeah",
    "yea",
    "yep",
    "yup",
    "yes",
    "ya",
    "okay",
    "ok",
    "right",
    "alright",
    "all right",
    "sure",
    "i see",
    "got it",
    "gotcha",
    "cool",
    "nice",
    "wow",
    "great",
    "exactly",
    "true",
    "indeed",
    "makes sense",
    "go on",
    "thanks",
    "thank you",
)

# Each of these characters is one word: their scripts do not separate words with spaces.
_CHARACTER_WORDS = re.compile(r"[฀-๿぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
# Sound annotations some STT models emit for non-speech, e.g. "[coughs]" or "(laughing)".
_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")


def interruption_words(text: str) -> list[str]:
    """Split a transcript into normalized words for interruption counting."""
    text = _ANNOTATION.sub(" ", text).casefold()
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if char == "'" or char == "’":
            chars.append("'")
        elif category.startswith(("P", "S")) or char.isspace():
            chars.append(" ")
        elif _CHARACTER_WORDS.match(char):
            chars.append(f" {char} ")
        else:
            chars.append(char)
    return [word.strip("'") for word in "".join(chars).split() if word.strip("'")]


def _phrases(ignore_words: tuple[str, ...] | list[str]) -> list[tuple[str, ...]]:
    phrases = {tuple(interruption_words(entry)) for entry in ignore_words}
    phrases.discard(())
    return sorted(phrases, key=len, reverse=True)


class FalseInterruptionConfig(BaseModel):
    """Session settings under ``audio.input.turn_detection.false_interruption``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_words: int = Field(default=1, ge=1, le=50)
    """Words that are not fillers or backchannels needed to commit an interruption."""
    playback: PlaybackMode = "continue"
    """What happens to the assistant's audio while the decision is pending.

    ``continue`` keeps playing. ``pause`` stops playback and resumes from the
    same point on a false interruption. ``duck`` keeps playing at low volume.
    The server applies ``pause`` and ``duck`` on WebRTC, where it paces
    playback. WebSocket clients own their player: they learn the mode from the
    opt-in ``speech_to_speech.interruption.*`` events and apply it themselves.
    """
    timeout_ms: int = Field(default=2000, ge=0, le=30000)
    """Give up and resume if no transcript arrives this long after speech stops."""
    ignore_words: tuple[str, ...] | None = None
    """Words and phrases that never count. ``None`` uses the built-in English list."""

    _phrases: list[tuple[str, ...]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._phrases = _phrases(DEFAULT_IGNORE_WORDS if self.ignore_words is None else self.ignore_words)

    def word_count(self, transcript: str) -> int:
        """Count the words of *transcript* that are not ignored fillers or backchannels."""
        words = interruption_words(transcript)
        phrases = self._phrases
        count = 0
        index = 0
        while index < len(words):
            for phrase in phrases:
                if tuple(words[index : index + len(phrase)]) == phrase:
                    index += len(phrase)
                    break
            else:
                count += 1
                index += 1
        return count


_UNSET = object()


def turn_detection_setting(turn_detection: Any) -> Any:
    """Return the raw ``false_interruption`` value a client set, or ``_UNSET``.

    ``turn_detection`` arrives as the SDK's ``ServerVad`` model, which keeps
    unknown fields as extras, or as a plain dict.
    """
    if turn_detection is None:
        return _UNSET
    if isinstance(turn_detection, dict):
        return turn_detection.get("false_interruption", _UNSET)
    extra = getattr(turn_detection, "model_extra", None) or {}
    return extra.get("false_interruption", _UNSET)


def is_unset(value: Any) -> bool:
    return value is _UNSET


def parse_false_interruption(value: Any) -> FalseInterruptionConfig | None:
    """Read the session field: ``None``/``false`` is off, ``true`` uses the defaults.

    Raises ``ValueError`` with a client-facing message for an invalid value.
    """
    if value is None or value is False:
        return None
    if value is True:
        return FalseInterruptionConfig()
    if isinstance(value, FalseInterruptionConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("turn_detection.false_interruption must be an object, true, false, or null.")
    try:
        return FalseInterruptionConfig.model_validate(value)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'value'}: {error['msg']}" for error in exc.errors()
        )
        raise ValueError(f"Invalid turn_detection.false_interruption ({problems}).") from None


class Verdict(str, Enum):
    """What the router does with one event of the pending turn."""

    HOLD = "hold"
    """Keep the event for replay if the interruption is committed."""
    DROP = "drop"
    """Consume the event: it belongs to a revision the turn has moved past."""
    CONFIRM = "confirm"
    """Commit the interruption: replay every held event, this one included."""
    DISMISS = "dismiss"
    """A false interruption: drop the held events and let the reply continue."""


@dataclass
class PendingInterruption:
    """User speech held while it is unclear whether it interrupts the reply."""

    turn_id: str
    revision: int | None
    config: FalseInterruptionConfig
    audio_start_ms: int = 0
    events: list[PipelineEvent] = field(default_factory=list)
    revisions: set[int | None] = field(default_factory=set)
    started_at: float = field(default_factory=time.monotonic)
    stopped_at: float | None = None
    previous_words: int = 0
    transcript: str = ""

    @classmethod
    def begin(cls, event: SpeechStartedEvent, config: FalseInterruptionConfig) -> PendingInterruption:
        assert event.turn_id is not None
        pending = cls(
            turn_id=event.turn_id,
            revision=event.turn_revision,
            config=config,
            audio_start_ms=event.audio_start_ms,
        )
        pending.events.append(event)
        pending.revisions.add(event.turn_revision)
        return pending

    def owns(self, event: PipelineEvent) -> bool:
        return getattr(event, "turn_id", None) == self.turn_id

    def deadline(self) -> float | None:
        """When to give up waiting for a transcript, once speech has stopped."""
        if self.stopped_at is None:
            return None
        return self.stopped_at + self.config.timeout_ms / 1000.0

    def expired(self, now: float | None = None) -> bool:
        deadline = self.deadline()
        return deadline is not None and (time.monotonic() if now is None else now) >= deadline

    def observe(self, event: PipelineEvent) -> Verdict:
        """Decide one event of this turn. The caller checks :meth:`owns` first."""
        from speech_to_speech.pipeline.events import (
            AudioInputCompletedEvent,
            PartialTranscriptionEvent,
            SpeechStartedEvent,
            SpeechStoppedEvent,
            TranscriptionCompletedEvent,
            TranscriptionFailedEvent,
        )

        revision = getattr(event, "turn_revision", None)
        if isinstance(event, SpeechStartedEvent):
            # The user resumed after a pause: VAD reopened the same turn.
            self.revision = revision
            self.revisions.add(revision)
            self.stopped_at = None
            self.previous_words = 0
            self.events.append(event)
            return Verdict.HOLD
        if revision != self.revision:
            return Verdict.DROP
        self.events.append(event)
        if isinstance(event, PartialTranscriptionEvent):
            # One hypothesis can hallucinate a word into a cough; two in a row rarely do.
            words = self.config.word_count(event.delta)
            confirmed = words >= self.config.min_words and self.previous_words >= self.config.min_words
            self.previous_words = words
            return Verdict.CONFIRM if confirmed else Verdict.HOLD
        if isinstance(event, SpeechStoppedEvent):
            self.stopped_at = time.monotonic()
            return Verdict.HOLD
        if isinstance(event, TranscriptionCompletedEvent):
            self.transcript = event.transcript
            if self.config.word_count(event.transcript) >= self.config.min_words:
                return Verdict.CONFIRM
            return Verdict.DISMISS
        if isinstance(event, TranscriptionFailedEvent):
            return Verdict.DISMISS
        if isinstance(event, AudioInputCompletedEvent):
            # Speech goes to the model as audio: no words to judge it by.
            return Verdict.CONFIRM
        return Verdict.HOLD


class SpeechToSpeechInterruptionPendingEvent(BaseModel):
    """User speech started during assistant audio; it may not be an interruption."""

    type: Literal["speech_to_speech.interruption.pending"] = "speech_to_speech.interruption.pending"
    event_id: str
    audio_start_ms: int
    playback: PlaybackMode


class SpeechToSpeechInterruptionResolvedEvent(BaseModel):
    """The pending interruption was committed (``interrupted``) or dismissed."""

    type: Literal["speech_to_speech.interruption.resolved"] = "speech_to_speech.interruption.resolved"
    event_id: str
    interrupted: bool
    transcript: str = ""
