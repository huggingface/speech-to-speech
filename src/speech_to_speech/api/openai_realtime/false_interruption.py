"""Barge-in that waits for words: coughs and backchannels do not cancel a reply.

With ``turn_detection.false_interruption`` set, user speech that starts while a
reply is active, queued, or still playing does not cancel that reply right
away. Playback pauses at once and the router holds the new turn's pipeline
events while the reply keeps generating. The interruption is committed once the
user's transcript has ``min_words`` words that are not fillers or backchannels
("uh", "mm-hmm", "yeah"): the held events then run through the ordinary
barge-in path, so the client sees what an immediate interruption would have
produced. If the turn ends empty, filler-only, or untranscribable, its events
are dropped, nothing enters the conversation, and the reply resumes from the
last gap between words before the pause.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

if TYPE_CHECKING:
    # RuntimeConfig imports this module, and the pipeline package imports RuntimeConfig.
    from speech_to_speech.pipeline.events import PipelineEvent, SpeechStartedEvent

INTERRUPTION_EVENTS_EXTENSION = "speech_to_speech.interruption"

# Resuming a paused reply replays from the last gap between words within this
# much of the audio played before the pause, rather than from the middle of a word.
RESUME_REWIND_MAX_MS = 600
_GAP_WINDOW_MS = 10
_GAP_MIN_MS = 50
_GAP_LEVEL = 0.1


def resume_rewind_samples(played: np.ndarray, sample_rate: int) -> int:
    """How many of the last *played* samples to replay when a paused reply resumes.

    Replay starts just before the next word at the latest quiet stretch of at
    least 50 ms (quieter than a tenth of the loudest 10 ms) in the last
    ``RESUME_REWIND_MAX_MS``. Returns 0 when playback stopped in such a gap,
    or when there is none.
    """
    window = max(1, sample_rate * _GAP_WINDOW_MS // 1000)
    count = min(len(played), sample_rate * RESUME_REWIND_MAX_MS // 1000) // window
    if count == 0:
        return 0
    tail = played[len(played) - count * window :].astype(np.float32).reshape(count, window)
    level = np.sqrt(np.mean(tail * tail, axis=1))
    quiet = level < level.max() * _GAP_LEVEL
    min_run = math.ceil(_GAP_MIN_MS / _GAP_WINDOW_MS)
    run = 0
    for index in range(count - 1, -1, -1):
        if quiet[index]:
            run += 1
            continue
        if run >= min_run:
            if index + run == count - 1:
                return 0  # playback stopped between words
            # Keep one window of the gap ahead of the next word.
            return (count - (index + run)) * window
        run = 0
    if run >= min_run and run < count:
        return (count - (run - 1)) * window
    return 0


# Non-lexical sounds across languages (hesitations, throat clears, laughter).
# Punctuation and repeated letters are normalized away ("Hmmmm" -> "hm",
# "mm-hmm" -> "mm hm", "haha"/"hehehe" -> "ha").
DEFAULT_FILLERS: tuple[str, ...] = (
    "uh",
    "um",
    "uhm",
    "er",
    "erm",
    "ah",
    "eh",
    "oh",
    "ooh",
    "hm",
    "hmm",
    "mm",
    "mhm",
    "mm hmm",
    "uh huh",
    "ahem",
    "ha",
    "hah",
    "heh",
    "aha",
)

# English listener backchannels. Direct answers ("yes", "sure") are omitted so
# early answers interrupt; set per session for other languages or [] to disable.
DEFAULT_BACKCHANNELS: tuple[str, ...] = (
    "yeah",
    "yea",
    "yep",
    "yup",
    "ya",
    "okay",
    "ok",
    "right",
    "alright",
    "all right",
    "i see",
    "got it",
    "gotcha",
    "makes sense",
    "go on",
)

_CHARACTER_WORDS = re.compile(r"[฀-๿぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
_LAUGHTER = re.compile(r"a?(?:h[ae]){2,}h?")
_REPEATED_LETTER = re.compile(r"(.)\1+")


def interruption_words(text: str) -> list[str]:
    """Split a transcript into normalized words for interruption counting."""
    text = _ANNOTATION.sub(" ", text).casefold()
    chars = []
    for char in text:
        if char in ("'", "’"):
            chars.append("'")
        elif unicodedata.category(char).startswith(("P", "S")) or char.isspace():
            chars.append(" ")
        elif _CHARACTER_WORDS.match(char):
            chars.append(f" {char} ")
        else:
            chars.append(char)
    return [word.strip("'") for word in "".join(chars).split() if word.strip("'")]


def _matching_words(text: str) -> tuple[str, ...]:
    words = []
    for word in interruption_words(text):
        collapsed = _REPEATED_LETTER.sub(r"\1", word)
        words.append("ha" if _LAUGHTER.fullmatch(collapsed) else collapsed)
    return tuple(words)


def _phrases(ignored: tuple[str, ...] | list[str]) -> list[tuple[str, ...]]:
    phrases = {_matching_words(entry) for entry in ignored}
    phrases.discard(())
    return sorted(phrases, key=len, reverse=True)


class FalseInterruptionConfig(BaseModel):
    """Session settings under ``audio.input.turn_detection.false_interruption``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_words: int = Field(default=1, ge=1, le=50)
    """Words other than fillers or backchannels needed to commit an interruption."""
    timeout_ms: int = Field(default=2000, gt=0, le=30000)
    """Give up and resume if no transcript arrives this long after speech stops.

    Keep it above the STT's slowest final transcript: a short real
    interruption ("No.") whose transcript misses it is dropped as noise.
    """
    fillers: tuple[str, ...] | None = None
    """Non-lexical sounds that never count ("uh", "mm-hmm", "ahem"). ``None`` uses ``DEFAULT_FILLERS``."""
    backchannels: tuple[str, ...] | None = None
    """Listener words that never count on their own ("yeah", "okay", "I see"). ``None`` uses ``DEFAULT_BACKCHANNELS``."""

    _phrases: list[tuple[str, ...]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        fillers = DEFAULT_FILLERS if self.fillers is None else self.fillers
        backchannels = DEFAULT_BACKCHANNELS if self.backchannels is None else self.backchannels
        self._phrases = _phrases((*fillers, *backchannels))

    def word_count(self, transcript: str) -> int:
        """Count the words of *transcript* that are not fillers or backchannels."""
        words = _matching_words(transcript)
        phrases = self._phrases
        count = 0
        index = 0
        while index < len(words):
            for phrase in phrases:
                if words[index : index + len(phrase)] == phrase:
                    index += len(phrase)
                    break
            else:
                count += 1
                index += 1
        return count


_UNSET = object()


def turn_detection_setting(turn_detection: Any) -> Any:
    """Return the raw ``false_interruption`` value a client set, or ``_UNSET``."""
    if turn_detection is None:
        return _UNSET
    if isinstance(turn_detection, dict):
        return turn_detection.get("false_interruption", _UNSET)
    extra = getattr(turn_detection, "model_extra", None) or {}
    return extra.get("false_interruption", _UNSET)


def is_unset(value: Any) -> bool:
    return value is _UNSET


def parse_false_interruption(value: Any) -> FalseInterruptionConfig | None:
    """Read the session field: ``None``/``false`` is off, ``true`` uses the defaults."""
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
    DROP = "drop"
    CONFIRM = "confirm"
    DISMISS = "dismiss"


@dataclass
class PendingInterruption:
    """User speech held while it is unclear whether it interrupts the reply."""

    turn_id: str
    revision: int | None
    config: FalseInterruptionConfig
    audio_start_ms: int = 0
    announced: bool = False
    events: list[PipelineEvent] = field(default_factory=list)
    revisions: set[int | None] = field(default_factory=set)
    started_at: float = field(default_factory=time.monotonic)
    stopped_at: float | None = None
    previous_words: int = 0
    transcript: str = ""
    releasing: bool = False
    released: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    def begin(
        cls,
        event: SpeechStartedEvent,
        config: FalseInterruptionConfig,
        *,
        announced: bool = False,
    ) -> PendingInterruption:
        assert event.turn_id is not None
        pending = cls(
            turn_id=event.turn_id,
            revision=event.turn_revision,
            config=config,
            audio_start_ms=event.audio_start_ms,
            announced=announced,
        )
        pending.events.append(event)
        pending.revisions.add(event.turn_revision)
        return pending

    def owns(self, event: PipelineEvent) -> bool:
        return getattr(event, "turn_id", None) == self.turn_id

    def deadline(self) -> float | None:
        if self.releasing or self.stopped_at is None:
            return None
        return self.stopped_at + self.config.timeout_ms / 1000.0

    def due(self, now: float | None = None) -> Verdict | None:
        deadline = self.deadline()
        if deadline is None:
            return None
        now = time.monotonic() if now is None else now
        return Verdict.DISMISS if now >= deadline else None

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
            if (
                isinstance(event, TranscriptionCompletedEvent)
                and self.config.word_count(event.transcript) >= self.config.min_words
            ):
                # "Wait." before the user went on: the words were said, whatever follows.
                self.transcript = event.transcript
                return Verdict.CONFIRM
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
            return (
                Verdict.CONFIRM
                if self.config.word_count(event.transcript) >= self.config.min_words
                else Verdict.DISMISS
            )
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


class SpeechToSpeechInterruptionResolvedEvent(BaseModel):
    """The pending interruption was committed (``interrupted``) or dismissed."""

    type: Literal["speech_to_speech.interruption.resolved"] = "speech_to_speech.interruption.resolved"
    event_id: str
    interrupted: bool
    transcript: str = ""
