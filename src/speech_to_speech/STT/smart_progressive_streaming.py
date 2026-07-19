#!/usr/bin/env python3
"""
Smart Progressive Streaming Handler (Whisper)

Provides frequent partial transcriptions with:
- Growing window up to 15s for accuracy
- Sentence-boundary-aware window sliding for audio > 15s
- Fixed sentences + active transcription

Uses a transformers Whisper ASR pipeline with word-level timestamps
(``return_timestamps="word"``) to locate sentence boundaries, so audio
before a completed sentence can be frozen and never re-transcribed.
Sentence segmentation uses NLTK punkt, which covers all languages this
project targets (en, de, fr, it, tr).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generator, Optional

import numpy as np

try:
    from lingua import Language, LanguageDetectorBuilder

    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ISO 639-1 -> NLTK punkt language names for the languages this project targets.
NLTK_LANGUAGES = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "tr": "turkish",
}

_lingua_detector = None


def _get_lingua_detector():
    """Lazily build a lingua detector over the target languages."""
    global _lingua_detector
    if not LINGUA_AVAILABLE:
        return None
    if _lingua_detector is None:
        iso_to_language = {
            lang.iso_code_639_1.name.lower(): lang for lang in Language.all() if lang.iso_code_639_1 is not None
        }
        languages = [iso_to_language[code] for code in NLTK_LANGUAGES if code in iso_to_language]
        _lingua_detector = LanguageDetectorBuilder.from_languages(*languages).build()
    return _lingua_detector


def detect_text_language(text: str) -> Optional[str]:
    """Detect the language of a text snippet with lingua, if available."""
    detector = _get_lingua_detector()
    if detector is None or len(text.strip()) < 20:
        return None
    detected = detector.detect_language_of(text)
    if detected is None:
        return None
    return detected.iso_code_639_1.name.lower()


def ensure_punkt() -> None:
    """Make sure the NLTK punkt sentence tokenizer data is available."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str, language_code: Optional[str]) -> list[str]:
    """Split text into sentences with NLTK punkt.

    Unknown language codes fall back to the English punkt model, which is
    close enough for boundary detection on punctuated Whisper output.
    """
    from nltk.tokenize import sent_tokenize

    language = NLTK_LANGUAGES.get(language_code or "", "english")
    return sent_tokenize(text, language=language)


@dataclass
class SentenceSpan:
    """A sentence with its end time relative to the decoded window start."""

    text: str
    end: float


@dataclass
class WindowResult:
    """Decoded transcription of one audio window."""

    text: str
    sentences: list[SentenceSpan]


@dataclass
class PartialTranscription:
    """Result from progressive streaming."""

    fixed_text: str  # Sentences that won't change
    active_text: str  # Current partial transcription
    timestamp: float  # Current position in audio
    is_final: bool  # True if this is the last update


class SmartProgressiveStreamingHandler:
    """
    Smart progressive streaming with sentence-aware window management.

    Strategy:
    1. Emit partial transcriptions every 500ms
    2. Use growing window (up to 15s) for better accuracy
    3. When audio > 15s, slide window using sentence boundaries:
       - Keep completed sentences as "fixed"
       - Only re-transcribe the "active" portion
    """

    def __init__(
        self,
        asr_pipeline: Any,
        emission_interval: float = 0.5,
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
        language: Optional[str] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            asr_pipeline: transformers automatic-speech-recognition pipeline
                wrapping a Whisper model with word-timestamp support
            emission_interval: Emit partial transcription every N seconds (default 500ms)
            max_window_size: Maximum window size before sliding (default 15s)
            sentence_buffer: Keep last N seconds of sentences in active window (default 2s)
            language: User-fixed language code (e.g. "de"). None lets Whisper
                auto-detect per window, which supports code-switching speech
                (e.g. an interpreter alternating languages every sentence)
            gen_kwargs: Extra generate kwargs forwarded to the pipeline
        """
        self.asr_pipeline = asr_pipeline
        self.emission_interval = emission_interval
        self.max_window_size = max_window_size
        self.sentence_buffer = sentence_buffer
        self.language = language
        self.gen_kwargs = dict(gen_kwargs or {})
        self.sample_rate = 16000

        ensure_punkt()

        # State for incremental streaming
        self.reset()

    def reset(self) -> None:
        """Reset state for new streaming session."""
        self.fixed_sentences: list[str] = []
        self.fixed_end_time: float = 0.0
        self.last_transcribed_length: int = 0

    def _decode_window(self, audio_window: np.ndarray) -> WindowResult:
        """Decode one audio window with word timestamps and sentence spans."""
        generate_kwargs = dict(self.gen_kwargs)
        if self.language:
            generate_kwargs["language"] = self.language

        result = self.asr_pipeline(
            {"raw": audio_window, "sampling_rate": self.sample_rate},
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
        )

        chunks = result.get("chunks") or []
        # Join word chunks ourselves so character offsets line up exactly with
        # the word timestamps when mapping sentence boundaries below.
        text = "".join(chunk.get("text", "") for chunk in chunks) if chunks else result.get("text", "")

        # Pick the punkt model from the window's dominant language; only
        # affects abbreviation handling at sentence boundaries.
        language = self.language or detect_text_language(text)
        sentences = self._sentence_spans(text, chunks, language)
        return WindowResult(text=text.strip(), sentences=sentences)

    def _sentence_spans(
        self,
        text: str,
        chunks: list[dict[str, Any]],
        language_code: Optional[str],
    ) -> list[SentenceSpan]:
        """Map NLTK sentences onto word timestamps to get sentence end times."""
        if not text.strip():
            return []

        # Character end offset and end time of each word within `text`
        word_ends: list[tuple[int, float]] = []
        cursor = 0
        last_time = 0.0
        for chunk in chunks:
            word = chunk.get("text", "")
            idx = text.find(word, cursor)
            if idx == -1:
                idx = cursor
            cursor = idx + len(word)
            start_t, end_t = chunk.get("timestamp") or (None, None)
            if end_t is None:
                # Whisper occasionally emits an open-ended final word
                end_t = start_t if start_t is not None else last_time
            last_time = end_t
            word_ends.append((cursor, end_t))

        spans: list[SentenceSpan] = []
        search_from = 0
        word_idx = 0
        for sentence in split_sentences(text, language_code):
            idx = text.find(sentence, search_from)
            if idx == -1:
                idx = search_from
            sentence_end_char = idx + len(sentence)
            search_from = sentence_end_char

            end_time = last_time
            while word_idx < len(word_ends):
                char_end, time_end = word_ends[word_idx]
                if char_end >= sentence_end_char:
                    end_time = time_end
                    word_idx += 1
                    break
                word_idx += 1
            spans.append(SentenceSpan(text=sentence.strip(), end=end_time))
        return spans

    def _maybe_freeze_sentences(self, result: WindowResult, window_duration: float) -> str:
        """Freeze completed sentences once the window grows too large.

        Returns the active (non-frozen) text for the current window.
        """
        if window_duration < self.max_window_size or len(result.sentences) <= 1:
            return result.text

        cutoff_time = window_duration - self.sentence_buffer

        freeze_count = 0
        for sentence in result.sentences:
            if sentence.end < cutoff_time:
                freeze_count += 1
            else:
                break

        if freeze_count == 0:
            return result.text

        frozen = result.sentences[:freeze_count]
        self.fixed_sentences.extend(sentence.text for sentence in frozen)
        # Sentence end times are relative to the current window start
        self.fixed_end_time += frozen[-1].end

        # Derive active text from the remaining sentences instead of
        # re-decoding: the next incremental call re-transcribes from the new
        # fixed point anyway, and Whisper inference is too costly to run twice.
        return " ".join(sentence.text for sentence in result.sentences[freeze_count:])

    def transcribe_incremental(self, audio: np.ndarray, is_final: bool = False) -> PartialTranscription:
        """
        Transcribe audio incrementally (for live streaming).

        Call this repeatedly with growing audio buffer.
        Returns a single PartialTranscription for current state.
        """
        current_length = len(audio)
        # Skip if not enough new audio, or no new audio since last transcription
        if current_length < self.sample_rate * 0.5 or current_length == self.last_transcribed_length:
            return PartialTranscription(
                fixed_text=" ".join(self.fixed_sentences),
                active_text="",
                timestamp=current_length / self.sample_rate,
                is_final=is_final,
            )

        self.last_transcribed_length = current_length

        # Extract window for transcription (from last fixed sentence to end)
        window_start_samples = int(self.fixed_end_time * self.sample_rate)
        audio_window = audio[window_start_samples:]

        result = self._decode_window(audio_window)
        active_text = self._maybe_freeze_sentences(result, len(audio_window) / self.sample_rate)

        return PartialTranscription(
            fixed_text=" ".join(self.fixed_sentences),
            active_text=active_text.strip(),
            timestamp=current_length / self.sample_rate,
            is_final=is_final,
        )

    def transcribe_progressive(self, audio: np.ndarray) -> Generator[PartialTranscription, None, None]:
        """
        Transcribe audio with smart progressive emissions.

        Simulates live streaming by feeding growing prefixes of `audio` to
        :meth:`transcribe_incremental` every `emission_interval` seconds.
        """
        step = max(1, int(self.emission_interval * self.sample_rate))
        position = 0

        while position < len(audio):
            position = min(position + step, len(audio))
            is_final = position >= len(audio)
            yield self.transcribe_incremental(audio[:position], is_final=is_final)


def demo_smart_progressive() -> None:
    """Demonstrate smart progressive streaming with Whisper."""
    import time

    import soundfile as sf
    from transformers import pipeline

    print("=" * 80)
    print("SMART PROGRESSIVE STREAMING DEMO (Whisper)")
    print("=" * 80)

    print("Loading model...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
    )

    audio, sr = sf.read("reachy-voice-test.wav")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy import signal

        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    audio = audio.astype(np.float32)

    test_cases = [
        (audio, "Short (5.8s)"),
        (np.concatenate([audio] * 3), "Medium (17.5s) - exceeds 15s window"),
        (np.concatenate([audio] * 5), "Long (29.1s) - multiple sentence fixings"),
    ]

    for test_audio, label in test_cases:
        duration = len(test_audio) / 16000

        print(f"\n{'=' * 80}")
        print(f"TEST: {label} - Duration: {duration:.1f}s")
        print("=" * 80)

        handler = SmartProgressiveStreamingHandler(
            asr_pipeline,
            emission_interval=0.5,
            max_window_size=15.0,
            sentence_buffer=2.0,
        )

        update_count = 0
        start_time = time.perf_counter()

        for result in handler.transcribe_progressive(test_audio):
            update_count += 1
            if update_count % 2 == 0 or result.is_final:
                elapsed = time.perf_counter() - start_time
                marker = "FINAL" if result.is_final else f"Update {update_count}"

                print(f"[{result.timestamp:5.1f}s | {elapsed:.3f}s elapsed] {marker}:")
                if result.fixed_text:
                    print(f"  Fixed:  {result.fixed_text[:70]}...")
                print(f"  Active: {result.active_text[:70]}...")
                print()

        total_time = time.perf_counter() - start_time
        print(f"Total updates: {update_count}")
        print(f"Total time: {total_time:.3f}s ({total_time / duration:.3f}s per audio second)")


if __name__ == "__main__":
    demo_smart_progressive()
