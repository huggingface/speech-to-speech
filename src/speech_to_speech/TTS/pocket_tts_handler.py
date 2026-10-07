from __future__ import annotations

import logging
import re
from threading import Event
from time import perf_counter
from typing import Any, Iterator, cast

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import transcript_for_log
from speech_to_speech.TTS.persian_normalization import normalize
from speech_to_speech.TTS.pocket_tts_farsi import PersianPhonemizer
from speech_to_speech.utils.utils import TORCH_DEVICES, resolve_device

logger = logging.getLogger(__name__)
console = Console()


class PocketTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handler for Pocket TTS model from Kyutai Labs.
    Supports streaming audio generation with voice cloning.
    """

    def setup(
        self,
        should_listen: Event,
        device: str = "cpu",
        voice: str = "alba",  # Default voice from catalog
        language: str = "english",
        sample_rate: int = 16000,  # Match the pipeline's native audio output rate.
        blocksize: int = 512,
        max_tokens: int = 50,
        gen_kwargs: dict[str, Any] | None = None,  # For compatibility with pipeline, not used
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        eos_threshold: float | None = None,
    ) -> None:
        """
        Initialize Pocket TTS handler.

        Args:
            should_listen: Event to control when to start listening again
            device: Device to run model on ('auto', 'cuda', 'npu', 'xpu', 'mps', 'cpu')
            voice: Voice to use. Can be:
                - A preset name: 'alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'
                - A local audio file path
                - A Hugging Face path like "hf://kyutai/tts-voices/..."
            language: PocketTTS language/model configuration to load
            sample_rate: Output sample rate (pocket-tts generates at 24kHz and will be resampled to this rate). Default 16kHz matches the pipeline's audio output.
            blocksize: Size of audio blocks to yield
            max_tokens: Token budget per synthesis chunk, capped at 18 for Farsi.
            model_name: Pocket TTS Farsi v2 model ID, or None for the selected language.
            temperature: Sampling temperature, or the model config default.
            eos_threshold: End-of-speech threshold, defaults to -2 for Farsi.
        """
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self._failed_responses: set[tuple[int | None, str | None, str | None, int | None]] = set()
        self.device = resolve_device(device, TORCH_DEVICES, "Pocket TTS")
        self.voice = voice
        self.language = language
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        if max_tokens <= 0:
            raise ValueError("Pocket TTS max_tokens must be positive")
        if model_name not in {None, "mehdi-hf/pocket-tts-farsi-v2"}:
            raise ValueError("Unsupported Pocket TTS model_name; use mehdi-hf/pocket-tts-farsi-v2 or select a language")
        self.model_name = model_name
        self.is_farsi = model_name == "mehdi-hf/pocket-tts-farsi-v2"
        self.max_tokens = min(max_tokens, 18) if self.is_farsi else max_tokens
        self.phonemizer: PersianPhonemizer | None = None
        if self.is_farsi and voice in {"alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"}:
            raise ValueError(
                "Pocket TTS Farsi requires a reference audio file or URL via --pocket_tts_voice; preset voice states are incompatible with these weights."
            )

        # Suppress verbose logging from pocket_tts library
        logging.getLogger("pocket_tts").setLevel(logging.WARNING)
        logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
        logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)

        # Import and load model
        from pocket_tts import TTSModel

        logger.info("Loading Pocket TTS model: %s", model_name or self.language)
        load_kwargs: dict[str, Any] = {}
        if temperature is not None:
            load_kwargs["temp"] = temperature
        if eos_threshold is not None or self.is_farsi:
            load_kwargs["eos_threshold"] = -2.0 if eos_threshold is None else eos_threshold
        if self.is_farsi:
            from huggingface_hub import hf_hub_download
            from pocket_tts.utils.config import Config

            required_flags = {
                "capitalize_first_letter",
                "append_terminal_punctuation",
                "pad_with_spaces_for_short_inputs",
            }
            if not required_flags.issubset(Config.model_fields):
                raise RuntimeError(
                    "Pocket TTS Farsi requires the mallahyari/pocket-tts fork. See the Pocket TTS Farsi installation instructions in README.md."
                )
            config = hf_hub_download(repo_id="mehdi-hf/pocket-tts-farsi-v2", filename="model.yaml")
            self.model = TTSModel.load_model(config=config, **load_kwargs)
        else:
            self.model = TTSModel.load_model(language=self.language, **load_kwargs)

        if self.is_farsi:
            if not callable(getattr(self.model, "_generate_audio_stream_short_text", None)):
                raise RuntimeError(
                    "Pocket TTS Farsi requires the pinned fork's short-text streaming API. See README.md."
                )
            for flag in ("capitalize_first_letter", "append_terminal_punctuation", "pad_with_spaces_for_short_inputs"):
                if getattr(self.model, flag, True):
                    raise RuntimeError(f"Pocket TTS Farsi config must disable {flag}")

        # Move model to specified device
        if self.device != "cpu":
            self.model = self.model.to(self.device)

        logger.info(f"Pocket TTS model moved to {self.device}")

        # Load voice state
        logger.info(f"Loading voice: {voice}")
        if self.is_farsi:
            from pocket_tts.data.audio import audio_read
            from pocket_tts.data.audio_utils import convert_audio
            from pocket_tts.utils.utils import download_if_necessary

            audio, rate = audio_read(download_if_necessary(voice))
            audio = audio[..., : int(5 * rate)]
            if audio.shape[-1] == 0:
                raise ValueError("Pocket TTS Farsi reference audio is empty")
            audio = convert_audio(audio, rate, self.model.sample_rate, 1)
            self.voice_state = self.model.get_state_for_audio_prompt(audio)
            self.phonemizer = PersianPhonemizer(self.device)
        else:
            self.voice_state = self.model.get_state_for_audio_prompt(voice)

        logger.info(f"Pocket TTS model sample rate: {self.model.sample_rate}")

    def _generate_audio(self, text: str, generation: int | None) -> Iterator[Any]:
        def cancelled() -> bool:
            return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

        phonemizer = getattr(self, "phonemizer", None)
        if phonemizer is None:
            if not cancelled():
                yield from self.model.generate_audio_stream(
                    self.voice_state,
                    text,
                    max_tokens=self.max_tokens,
                    copy_state=True,
                )
            return

        # Spell decimals before splitting, and recognize adjacent sentences too.
        sentences = [part.strip() for part in re.split(r"[.!؟?]+", normalize(text)) if part.strip()]
        for index, sentence in enumerate(sentences):
            if cancelled():
                return
            phonemes = phonemizer(sentence)
            if cancelled():
                return
            if not phonemes:
                continue
            if index:
                import torch

                yield torch.zeros(int(self.model.sample_rate * 0.25))
            for chunk in self._phoneme_chunks(phonemes):
                if cancelled():
                    return
                yield from self._synthesize_farsi_chunk(chunk, generation)

    def _phoneme_chunks(self, phonemes: str) -> Iterator[str]:
        # In this notation ? is a glottal stop and ; is zh, never punctuation.
        if re.search(r"[^a-zA-Z?;1\s]", phonemes):
            raise ValueError("Pocket TTS Farsi G2P returned invalid phoneme symbols")
        sp = self.model.flow_lm.conditioner.tokenizer.sp

        def count(text: str) -> int:
            ids = sp.encode(text.replace("1", ""))
            if sp.unk_id() in ids:
                raise ValueError("Pocket TTS Farsi phonemes contain a symbol outside the model vocabulary")
            return len(ids)

        groups: list[str] = []
        group: list[str] = []
        for word in phonemes.split():
            if "1" in word and (word == "1" or not word.endswith("1") or word.count("1") != 1):
                raise ValueError("Pocket TTS Farsi ezafe marker must occur once at the end of a word")
            group.append(word)
            if not word.endswith("1"):
                groups.append(" ".join(group))
                group = []
        if group:
            raise ValueError("Pocket TTS Farsi ezafe marker needs a following word")
        current = ""
        for linked_phrase in groups:
            if count(linked_phrase) > self.max_tokens:
                raise ValueError("Pocket TTS Farsi linked phrase exceeds the token budget. Shorten the phrase.")
            trial = f"{current} {linked_phrase}".strip()
            if current and count(trial) > self.max_tokens:
                yield current.replace("1", "")
                current = linked_phrase
            else:
                current = trial
        if current:
            yield current.replace("1", "")

    def _synthesize_farsi_chunk(self, phonemes: str, generation: int | None) -> Iterator[Any]:
        import torch

        def cancelled() -> bool:
            return generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)

        tokens = len(self.model.flow_lm.conditioner.tokenizer.sp.encode(phonemes))
        cap_seconds = tokens / 3.0 + 2.0
        for _ in range(2):
            if cancelled():
                return
            parts = []
            # The public API reinterprets glottal stops as punctuation. The pinned
            # fork's short-text method takes the exact, already chunked phonemes.
            for audio in self.model._generate_audio_stream_short_text(
                model_state=self.voice_state,
                text_to_generate=phonemes,
                frames_after_eos=0,
                copy_state=True,
            ):
                if cancelled():
                    return
                parts.append(audio)
            if cancelled():
                return
            if not parts:
                continue
            waveform = torch.cat(parts).detach().reshape(-1)
            duration = waveform.numel() / self.model.sample_rate
            if (
                0.08 < duration < cap_seconds * 0.98
                and torch.isfinite(waveform).all().item()
                and waveform.abs().max().item() > 1e-5
            ):
                yield waveform
                return
        raise RuntimeError(
            "Pocket TTS Farsi generation was silent or reached its length cap twice. Shorten the phrase or try another reference."
        )

    @property
    def min_time_to_debug(self) -> float:
        """
        Override to suppress logging for individual audio chunks.
        Pocket TTS yields many small chunks (~10-20ms each), which would flood logs.
        Only log if a chunk takes unusually long (>100ms).
        """
        return 0.1  # 100ms threshold

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            self._failed_responses.discard(self._response_identity(tts_input))
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if self._response_identity(tts_input) in self._failed_responses:
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        gen = tts_input.cancel_generation
        if gen is None and self.cancel_scope is not None:
            gen = self.cancel_scope.generation
        language_code = tts_input.tts_language_code
        text = tts_input.text
        logger.debug(f"Received language code: {language_code}")

        console.print(f"[green]ASSISTANT: {text}")

        # Generate audio stream
        logger.debug("Generating audio: %s", transcript_for_log(text))

        try:
            yield from self._stream_pcm(text, gen)
        except Exception:
            if getattr(self, "phonemizer", None) is None:
                raise
            if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                logger.info("TTS generation cancelled (interruption)")
                return
            self._failed_responses.add(self._response_identity(tts_input))
            logger.exception("Pocket TTS Farsi synthesis failed")
            self.queue_out.put(
                cast(
                    TTSOut,
                    ResponseFailedEvent(
                        message="Pocket TTS Farsi synthesis failed",
                        turn_id=tts_input.turn_id,
                        turn_revision=tts_input.turn_revision,
                        cancel_generation=gen,
                        response_key=tts_input.response_key,
                    ),
                )
            )

    @staticmethod
    def _response_identity(message: TTSIn) -> tuple[int | None, str | None, str | None, int | None]:
        return (message.cancel_generation, message.response_key, message.turn_id, message.turn_revision)

    def on_session_end(self) -> None:
        self._failed_responses.clear()

    def _stream_pcm(self, text: str, gen: int | None) -> Iterator[TTSOut]:
        pipeline_start = perf_counter()
        first_chunk = True

        from speech_to_speech.api.openai_realtime.utils import StreamingPcm16Resampler

        resampler = StreamingPcm16Resampler(self.model.sample_rate, self.sample_rate)
        pending = np.empty(0, dtype="<i2")

        def cancelled() -> bool:
            return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

        for audio_chunk in self._generate_audio(text, gen):
            if cancelled():
                logger.info("TTS generation cancelled (interruption)")
                return
            if first_chunk:
                logger.debug(f"Time to first audio: {perf_counter() - pipeline_start:.3f}s")
                first_chunk = False

            # Saturate before converting, so peaks cannot wrap around as int16.
            audio_np = audio_chunk.detach().cpu().numpy().reshape(-1)
            pcm = (np.clip(audio_np, -1, 1) * 32767).astype("<i2")
            # Keep filter history across chunks instead of restarting at every
            # playback block, which introduces periodic boundary artifacts.
            converted = np.frombuffer(resampler.push(pcm.tobytes()), dtype="<i2")
            pending = np.concatenate((pending, converted))
            while len(pending) >= self.blocksize:
                if cancelled():
                    return
                yield pending[: self.blocksize].copy()
                pending = pending[self.blocksize :]

        if cancelled():
            return
        tail = np.frombuffer(resampler.flush(), dtype="<i2")
        pending = np.concatenate((pending, tail))
        for index in range(0, len(pending), self.blocksize):
            if cancelled():
                return
            block = pending[index : index + self.blocksize]
            if len(block) < self.blocksize:
                block = np.pad(block, (0, self.blocksize - len(block)))
            yield block.copy()
