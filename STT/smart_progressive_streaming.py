#!/usr/bin/env python3
"""
Smart Progressive Streaming Handler

Provides frequent partial transcriptions (every 250ms) with:
- Growing window up to 15s for accuracy
- Sentence-boundary-aware window sliding for audio > 15s
- Fixed sentences + active transcription
"""

import numpy as np
from typing import Generator, Tuple, List, Callable
from dataclasses import dataclass

@dataclass
class PartialTranscription:
    """Result from progressive streaming."""
    fixed_text: str  # Sentences that won't change
    active_text: str  # Current partial transcription
    timestamp: float  # Current position in audio
    is_final: bool  # True if this is the last update


def _join_segments(segments: List[dict]) -> str:
    return " ".join(seg.get("segment", "").strip() for seg in segments if seg.get("segment")).strip()


def _split_fixed_segments(
    segments: List[dict],
    total_duration: float,
    max_window_size: float,
    sentence_buffer: float,
) -> Tuple[List[dict], List[dict]]:
    if total_duration < max_window_size:
        return [], segments

    cutoff = max(0.0, total_duration - sentence_buffer)
    fixed = []
    for seg in segments:
        if seg.get("end", 0.0) <= cutoff:
            fixed.append(seg)
        else:
            break
    return fixed, segments[len(fixed):]


class SmartProgressiveStreamingHandler:
    """
    Smart progressive streaming with sentence-aware window management.

    Strategy:
    1. Emit partial transcriptions every 250ms
    2. Use growing window (up to 15s) for better accuracy
    3. When audio > 15s, slide window using sentence boundaries:
       - Keep completed sentences as "fixed"
       - Only re-transcribe the "active" portion
    """

    def __init__(self,
                 model,
                 emission_interval: float = 0.25,
                 max_window_size: float = 15.0,
                 sentence_buffer: float = 2.0):
        """
        Args:
            model: Parakeet model with sentence alignment
            emission_interval: Emit partial transcription every N seconds (default 250ms)
            max_window_size: Maximum window size before sliding (default 15s)
            sentence_buffer: Keep last N seconds of sentences in active window (default 2s)
        """
        self.model = model
        self.emission_interval = emission_interval
        self.max_window_size = max_window_size
        self.sentence_buffer = sentence_buffer
        self.sample_rate = 16000

        # State for incremental streaming
        self.reset()

    def reset(self):
        """Reset state for new streaming session."""
        self.fixed_sentences = []
        self.fixed_end_time = 0.0
        self.last_transcribed_length = 0

    def transcribe_incremental(self, audio: np.ndarray) -> PartialTranscription:
        """
        Transcribe audio incrementally (for live streaming).

        Call this repeatedly with growing audio buffer.
        Returns a single PartialTranscription for current state.
        """
        # Skip if not enough new audio
        current_length = len(audio)
        if current_length < self.sample_rate * 0.5:  # Need at least 500ms
            return PartialTranscription(
                fixed_text=" ".join(self.fixed_sentences),
                active_text="",
                timestamp=current_length / self.sample_rate,
                is_final=False
            )

        # Skip if no new audio since last transcription
        if current_length == self.last_transcribed_length:
            return PartialTranscription(
                fixed_text=" ".join(self.fixed_sentences),
                active_text="",
                timestamp=current_length / self.sample_rate,
                is_final=False
            )

        self.last_transcribed_length = current_length

        # Extract window for transcription (from last fixed sentence to end)
        window_start_samples = int(self.fixed_end_time * self.sample_rate)
        audio_window = audio[window_start_samples:]

        # Transcribe current window
        import mlx.core as mx
        audio_mx = mx.array(audio_window, dtype=mx.float32)
        result = self.model.decode_chunk(audio_mx, verbose=False)

        # Check if window exceeds max_window_size
        window_duration = len(audio_window) / self.sample_rate

        segments_rel = [
            {
                "segment": sentence.text.strip(),
                "start": getattr(sentence, "start", 0.0),
                "end": sentence.end,
            }
            for sentence in getattr(result, "sentences", []) or []
        ]

        if window_duration >= self.max_window_size and len(segments_rel) > 1:
            fixed_rel, _active_rel = _split_fixed_segments(
                segments_rel,
                window_duration,
                self.max_window_size,
                self.sentence_buffer,
            )

            if fixed_rel:
                self.fixed_sentences.extend([seg["segment"] for seg in fixed_rel if seg.get("segment")])
                self.fixed_end_time += fixed_rel[-1]["end"]

                # Re-transcribe from new fixed point
                window_start_samples = int(self.fixed_end_time * self.sample_rate)
                audio_window = audio[window_start_samples:]
                import mlx.core as mx
                audio_mx = mx.array(audio_window, dtype=mx.float32)
                result = self.model.decode_chunk(audio_mx, verbose=False)

                segments_rel = [
                    {
                        "segment": sentence.text.strip(),
                        "start": getattr(sentence, "start", 0.0),
                        "end": sentence.end,
                    }
                    for sentence in getattr(result, "sentences", []) or []
                ]

        # Build output
        fixed_text = " ".join(self.fixed_sentences)
        active_text = _join_segments(segments_rel) or result.text.strip()
        timestamp = len(audio) / self.sample_rate

        return PartialTranscription(
            fixed_text=fixed_text,
            active_text=active_text,
            timestamp=timestamp,
            is_final=False
        )

    def transcribe_progressive(self, audio: np.ndarray) -> Generator[PartialTranscription, None, None]:
        """
        Transcribe audio with smart progressive emissions.

        Yields PartialTranscription with:
        - fixed_text: Completed sentences (won't change)
        - active_text: Current partial transcription
        - timestamp: Current position
        - is_final: True for last update
        """
        total_duration = len(audio) / self.sample_rate
        position = 0  # Start of current window (in samples)
        fixed_sentences = []  # List of completed sentence texts
        fixed_end_time = 0.0  # End time of last fixed sentence

        while position < len(audio):
            # Determine current window end
            remaining = (len(audio) - position) / self.sample_rate

            if remaining <= self.emission_interval:
                # Last chunk - process everything remaining
                window_end = len(audio)
                is_final = True
            else:
                # Regular chunk
                window_end = min(len(audio), position + int(self.emission_interval * self.sample_rate))
                is_final = False

            # Extract window for transcription
            # Window includes: fixed_end_time to current position
            # (we re-transcribe from last fixed sentence to get better context)
            window_start_samples = int(fixed_end_time * self.sample_rate)
            audio_window = audio[window_start_samples:window_end]

            # Transcribe current window
            import mlx.core as mx
            audio_mx = mx.array(audio_window, dtype=mx.float32)
            result = self.model.decode_chunk(audio_mx, verbose=False)

            # Check if window exceeds max_window_size
            window_duration = (window_end - window_start_samples) / self.sample_rate

            segments_rel = [
                {
                    "segment": sentence.text.strip(),
                    "start": getattr(sentence, "start", 0.0),
                    "end": sentence.end,
                }
                for sentence in getattr(result, "sentences", []) or []
            ]

            if window_duration >= self.max_window_size and len(segments_rel) > 1:
                fixed_rel, _active_rel = _split_fixed_segments(
                    segments_rel,
                    window_duration,
                    self.max_window_size,
                    self.sentence_buffer,
                )

                if fixed_rel:
                    fixed_sentences.extend([seg["segment"] for seg in fixed_rel if seg.get("segment")])
                    fixed_end_time += fixed_rel[-1]["end"]

                    # Re-transcribe from new fixed_end_time
                    window_start_samples = int(fixed_end_time * self.sample_rate)
                    audio_window = audio[window_start_samples:window_end]
                    import mlx.core as mx
                    audio_mx = mx.array(audio_window, dtype=mx.float32)
                    result = self.model.decode_chunk(audio_mx, verbose=False)

                    segments_rel = [
                        {
                            "segment": sentence.text.strip(),
                            "start": getattr(sentence, "start", 0.0),
                            "end": sentence.end,
                        }
                        for sentence in getattr(result, "sentences", []) or []
                    ]

            # Build output
            fixed_text = " ".join(fixed_sentences)
            active_text = _join_segments(segments_rel) or result.text.strip()
            timestamp = window_end / self.sample_rate

            yield PartialTranscription(
                fixed_text=fixed_text,
                active_text=active_text,
                timestamp=timestamp,
                is_final=is_final
            )

            # Move to next position
            position = window_end

            if is_final:
                break


class SmartProgressiveStreamingTextHandler:
    """
    Progressive streaming for decoders that return timestamped segments.

    If timestamped segments are available, uses them to split fixed vs active
    text. If not, falls back to repeated full-buffer transcriptions and locks
    stable sentence prefixes when they repeat across updates.
    """

    def __init__(
        self,
        transcribe_fn: Callable[[np.ndarray], str],
        emission_interval: float = 0.25,
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
    ):
        self.transcribe_fn = transcribe_fn
        self.emission_interval = emission_interval
        self.max_window_size = max_window_size
        self.sentence_buffer = sentence_buffer
        self.sample_rate = 16000
        self.reset()

    def reset(self):
        self.fixed_text = ""
        self.last_text = ""
        self.last_transcribed_length = 0
        self._pending_fixed = ""
        self._pending_fixed_hits = 0

    def _sentence_cutoff(self, text: str) -> str:
        # Find a stable sentence boundary in a prefix
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct == -1:
            return ""
        return text[: last_punct + 1].strip()

    def _longest_common_prefix(self, a: str, b: str) -> str:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return a[:idx]

    def transcribe_incremental(self, audio: np.ndarray) -> PartialTranscription:
        current_length = len(audio)
        if current_length < self.sample_rate * 0.5:
            return PartialTranscription(
                fixed_text=self.fixed_text,
                active_text="",
                timestamp=current_length / self.sample_rate,
                is_final=False,
            )

        if current_length == self.last_transcribed_length:
            return PartialTranscription(
                fixed_text=self.fixed_text,
                active_text="",
                timestamp=current_length / self.sample_rate,
                is_final=False,
            )

        self.last_transcribed_length = current_length

        # Transcribe the full buffer.
        result = self.transcribe_fn(audio)
        if hasattr(result, "timestamp") and isinstance(result.timestamp, dict):
            segments = result.timestamp.get("segment") or []
            duration = current_length / self.sample_rate

            fixed_segments, active_segments = _split_fixed_segments(
                segments,
                duration,
                self.max_window_size,
                self.sentence_buffer,
            )

            self.fixed_text = _join_segments(fixed_segments)
            active_text = _join_segments(active_segments)
            if not active_text:
                text = getattr(result, "text", "") or ""
                active_text = text[len(self.fixed_text) :].strip() if self.fixed_text else text.strip()
        else:
            text = (result or "").strip()

            # Determine stable prefix
            common = self._longest_common_prefix(self.last_text, text)
            candidate_fixed = self._sentence_cutoff(common)

            if candidate_fixed and candidate_fixed.startswith(self.fixed_text):
                if candidate_fixed == self._pending_fixed:
                    self._pending_fixed_hits += 1
                else:
                    self._pending_fixed = candidate_fixed
                    self._pending_fixed_hits = 1

                if self._pending_fixed_hits >= 2 and len(candidate_fixed) > len(self.fixed_text):
                    self.fixed_text = candidate_fixed

            self.last_text = text

            active_text = text[len(self.fixed_text) :].strip() if self.fixed_text else text

        return PartialTranscription(
            fixed_text=self.fixed_text,
            active_text=active_text,
            timestamp=current_length / self.sample_rate,
            is_final=False,
        )

def demo_smart_progressive():
    """Demonstrate smart progressive streaming."""
    from mlx_audio.stt.generate import load_model
    import soundfile as sf
    import time

    print("="*80)
    print("SMART PROGRESSIVE STREAMING DEMO")
    print("="*80)
    print("\nFeatures:")
    print("  • Partial updates every 250ms")
    print("  • Growing window up to 15s")
    print("  • Sentence-aware window sliding for long audio")
    print()

    # Load model
    print("Loading model...")
    model = load_model('mlx-community/parakeet-tdt-0.6b-v3')

    # Load audio
    audio, sr = sf.read("reachy-voice-test.wav")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    audio = audio.astype(np.float32)

    # Test cases
    test_cases = [
        (audio, "Short (5.8s)"),
        (np.concatenate([audio] * 3), "Medium (17.5s) - exceeds 15s window"),
        (np.concatenate([audio] * 5), "Long (29.1s) - multiple sentence fixings"),
    ]

    for test_audio, label in test_cases:
        duration = len(test_audio) / 16000

        print(f"\n{'='*80}")
        print(f"TEST: {label} - Duration: {duration:.1f}s")
        print('='*80)

        handler = SmartProgressiveStreamingHandler(
            model,
            emission_interval=0.25,  # 250ms updates
            max_window_size=15.0,    # Max 15s window
            sentence_buffer=2.0      # Keep 2s of sentences in active window
        )

        print("\nProgressive transcriptions (showing every 4th update for readability):")
        print()

        update_count = 0
        start_time = time.perf_counter()

        for result in handler.transcribe_progressive(test_audio):
            update_count += 1

            # Show every 4th update (every 1s) to keep output manageable
            if update_count % 4 == 0 or result.is_final:
                elapsed = time.perf_counter() - start_time
                marker = "FINAL" if result.is_final else f"Update {update_count}"

                print(f"[{result.timestamp:5.1f}s | {elapsed:.3f}s elapsed] {marker}:")

                if result.fixed_text:
                    print(f"  Fixed:  {result.fixed_text[:70]}...")
                print(f"  Active: {result.active_text[:70]}...")
                print()

        total_time = time.perf_counter() - start_time
        print(f"Total updates: {update_count}")
        print(f"Total time: {total_time:.3f}s ({total_time/duration:.3f}s per audio second)")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. Frequent updates (250ms):
   - User sees transcription building in real-time
   - Low latency feel

2. Growing window (up to 15s):
   - Better accuracy with more context
   - No arbitrary chunking for short audio

3. Sentence-aware sliding (for audio > 15s):
   - Completed sentences become "fixed"
   - Only active portion re-transcribed
   - Maintains accuracy while managing window size

4. Performance:
   - Processing happens DURING user speaking
   - Final result available immediately when user stops
    """)


if __name__ == "__main__":
    demo_smart_progressive()
