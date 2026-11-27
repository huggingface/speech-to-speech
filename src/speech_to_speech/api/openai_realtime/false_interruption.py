"""Barge-in that waits for words: coughs and backchannels do not cancel a reply.

With ``turn_detection.false_interruption`` set, user speech that starts while a
reply is active, queued, or still playing does not cancel that reply right
away. Playback pauses at once (by default) and the router holds the new turn's
pipeline events, while the reply keeps generating. The interruption is
committed once the user's transcript has ``min_words`` words that are not
fillers or backchannels ("uh", "mm-hmm", "yeah"): the held events then run
through the ordinary barge-in path, so the client sees exactly what an
immediate interruption would have produced. If the turn ends empty,
filler-only, or untranscribable, its events are dropped. The client never
hears about that turn, nothing enters the conversation, and the reply resumes
from the last gap between words before the pause.
"""

from __future__ import annotations

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

PlaybackMode = Literal["continue", "pause", "duck"]
PLAYBACK_MODES: tuple[PlaybackMode, ...] = ("continue", "pause", "duck")

# Speech that commits an interruption before its words are known, while the
# reply keeps playing over it.
DEFAULT_MIN_SPEECH_MS = 1200

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
    or when there is none: replaying from the middle of another word would be
    no better than resuming where playback stopped.
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
    return 0


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
    min_speech_ms: int | None = Field(default=None, ge=0, le=30000)
    """Active speech that commits an interruption even before its words are known; 0 turns it off.

    By default (``None``) it is off while the reply is paused, which costs
    nothing to wait on and keeps a long cough from cancelling it, and
    ``DEFAULT_MIN_SPEECH_MS`` while the reply keeps playing over the user:
    words can come late, since VAD splits "Wait, wait, can you..." into short
    segments and a busy STT may skip partial transcripts while the reply is
    being synthesized.
    """
    playback: PlaybackMode = "pause"
    """What happens to the assistant's audio while the decision is pending.

    ``pause`` stops playback at once and, on a false interruption, resumes
    from the last gap between words. ``continue`` keeps playing. ``duck``
    keeps playing at low volume. The server applies ``pause`` and ``duck`` on
    WebRTC, where it paces playback. WebSocket clients own their player: they
    learn the mode from the opt-in ``speech_to_speech.interruption.*`` events
    and apply it themselves; without those events their playback continues.
    """
    timeout_ms: int = Field(default=2000, ge=0, le=30000)
    """Give up and resume if no transcript arrives this long after speech stops."""
    ignore_words: tuple[str, ...] | None = None
    """Words and phrases that never count. ``None`` uses the built-in English list."""

    _phrases: list[tuple[str, ...]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._phrases = _phrases(DEFAULT_IGNORE_WORDS if self.ignore_words is None else self.ignore_words)

    def speech_threshold_ms(self, paused: bool) -> int:
        """The ``min_speech_ms`` that applies when playback is (or is not) paused."""
        if self.min_speech_ms is not None:
            return self.min_speech_ms
        return 0 if paused else DEFAULT_MIN_SPEECH_MS

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
    min_speech_ms: int = 0
    events: list[PipelineEvent] = field(default_factory=list)
    revisions: set[int | None] = field(default_factory=set)
    started_at: float = field(default_factory=time.monotonic)
    stopped_at: float | None = None
    previous_words: int = 0
    transcript: str = ""
    # Active speech of finished segments, and of the one in progress.
    closed_speech_ms: float = 0.0
    speaking_since: float | None = None
    speaking_active_ms: float = 0.0

    @classmethod
    def begin(
        cls,
        event: SpeechStartedEvent,
        config: FalseInterruptionConfig,
        *,
        paused: bool = False,
    ) -> PendingInterruption:
        assert event.turn_id is not None
        pending = cls(
            turn_id=event.turn_id,
            revision=event.turn_revision,
            config=config,
            audio_start_ms=event.audio_start_ms,
            min_speech_ms=config.speech_threshold_ms(paused),
        )
        pending.events.append(event)
        pending.revisions.add(event.turn_revision)
        pending._speech_started(event, pending.started_at)
        return pending

    def _speech_started(self, event: SpeechStartedEvent, now: float) -> None:
        self.speaking_since = now
        self.speaking_active_ms = event.active_speech_ms

    def speech_ms(self, now: float | None = None) -> float:
        """Active speech heard so far in this turn."""
        if self.speaking_since is None:
            return self.closed_speech_ms
        now = time.monotonic() if now is None else now
        return self.closed_speech_ms + self.speaking_active_ms + (now - self.speaking_since) * 1000

    def owns(self, event: PipelineEvent) -> bool:
        return getattr(event, "turn_id", None) == self.turn_id

    def _long_enough(self, now: float | None = None) -> bool:
        return bool(self.min_speech_ms) and self.speech_ms(now) >= self.min_speech_ms

    def deadline(self) -> float | None:
        """When :meth:`due` next changes without a new event, if ever."""
        if self.speaking_since is not None and self.min_speech_ms:
            remaining_ms = self.min_speech_ms - self.speech_ms(self.speaking_since)
            return self.speaking_since + max(0.0, remaining_ms) / 1000.0
        if self.stopped_at is not None:
            return self.stopped_at + self.config.timeout_ms / 1000.0
        return None

    def due(self, now: float | None = None) -> Verdict | None:
        """A decision that time alone has made: long speech, or no transcript after the timeout."""
        now = time.monotonic() if now is None else now
        if self.speaking_since is not None:
            return Verdict.CONFIRM if self._long_enough(now) else None
        if self.stopped_at is not None and now >= self.stopped_at + self.config.timeout_ms / 1000.0:
            return Verdict.DISMISS
        return None

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
            self._speech_started(event, time.monotonic())
            return Verdict.CONFIRM if self._long_enough() else Verdict.HOLD
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
            now = time.monotonic()
            if self.speaking_since is not None:
                heard = self.speaking_active_ms + (now - self.speaking_since) * 1000
                self.closed_speech_ms += event.active_speech_ms or heard
                self.speaking_since = None
            self.stopped_at = now
            return Verdict.CONFIRM if self._long_enough(now) else Verdict.HOLD
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
