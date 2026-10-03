from __future__ import annotations

import logging
import re
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.LLM.utils import WHISPER_LANGUAGE_TO_LLM_LANGUAGE
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.utils import resolve_device

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000
FARSI_MODEL = "mehdi-hf/nemotron-asr-streaming-farsi"

SUPPORTED_LANGUAGES = ["en"]
SUPPORTED_DEVICES = ("cuda", "npu", "cpu")
_NEMOTRON_LANG_TAG = re.compile(r"\s*<([A-Za-z]{2})(?:-[A-Za-z]{2})?>\s*$")


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def _reported_language_code(start_language: str) -> str:
    requested = (start_language or "").strip() or "en"
    if requested.lower() == "auto":
        return "en"
    return requested


def _is_nemotron_multilingual(model_name: str) -> bool:
    return "nemotron-3.5" in model_name.lower()


def _split_lang_tag(text: str) -> tuple[str, str | None]:
    match = _NEMOTRON_LANG_TAG.search(text)
    if match is None:
        return text.strip(), None
    language_code = match.group(1).lower()
    # Nemotron uses Bokmål's "nb"; the pipeline uses Whisper's "no" for Norwegian.
    if language_code == "nb":
        language_code = "no"
    return text[: match.start()].strip(), language_code


def _warn_reported_language(start_language: str) -> None:
    requested = (start_language or "").strip() or "en"
    if requested.lower() == "auto":
        logger.warning(
            "NeMo Parakeet Unified does not detect language; %r is a request, not a code. Reporting %r instead.",
            start_language,
            "en",
        )
        return
    if requested not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE:
        logger.warning(
            "Unknown language code %r for parakeet_unified_language; "
            "--enable_lang_prompt and some TTS backends may not resolve it.",
            requested,
        )


class NemoASRSTTHandler(BaseSTTHandler):
    """Speech to text with a NeMo ASR checkpoint through ASRModel.transcribe."""

    _detects_utterance_language = False

    def setup(
        self,
        model_name: str,
        device: str = "auto",
        language: str = "en",
        gen_kwargs: dict | None = None,
    ) -> None:
        logger.info("Loading NeMo ASR STT model: %s", model_name)
        self.device = resolve_device(device, SUPPORTED_DEVICES, "NeMo ASR")
        self.start_language = (language or "").strip() or "en"
        self.model_name = model_name
        self._is_farsi = model_name == FARSI_MODEL
        self._detects_utterance_language = _is_nemotron_multilingual(model_name)
        if self._is_farsi:
            # This checkpoint has a Persian-only tokenizer and a single trained prompt.
            self.start_language = "fa"
        if not self._detects_utterance_language:
            _warn_reported_language(self.start_language)
        self.language = _reported_language_code(self.start_language)
        self.last_language = self.language
        self.gen_kwargs = dict(gen_kwargs or {})

        from nemo.collections.asr.models import ASRModel

        if self._is_farsi:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(model_name, "nemotron-asr-streaming-farsi.nemo")
            self.model = ASRModel.restore_from(model_path, map_location="cpu")
        else:
            self.model = ASRModel.from_pretrained(model_name=model_name)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        if self._is_farsi:
            import torch
            from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
            from omegaconf import OmegaConf

            self.model = self.model.to(dtype=torch.float32).eval()
            self.model.encoder.set_default_att_context_size([56, 13])
            self.model.change_decoding_strategy(OmegaConf.structured(RNNTDecodingConfig(fused_batch_size=-1)))
            self.model.set_inference_prompt("fa-IR")
        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        try:
            self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except Exception:
            if self._is_farsi:
                raise
            logger.warning("%s: warmup failed", self.__class__.__name__, exc_info=True)

    def _transcribe(self, audio: np.ndarray) -> str:
        if getattr(self, "_is_farsi", False):
            return self._transcribe_farsi(audio)
        if self._detects_utterance_language:
            set_prompt = getattr(self.model, "set_inference_prompt", None)
            if callable(set_prompt):
                set_prompt("auto")
            try:
                results = self.model.transcribe([audio], target_lang="auto")
            except TypeError:
                results = self.model.transcribe([audio])
        else:
            results = self.model.transcribe([audio])
        if not results:
            return ""
        return _extract_text(results[0]).strip()

    def _transcribe_farsi(self, audio: np.ndarray) -> str:
        """Use the trained Persian prompt without NeMo's randomized transcribe dataloader."""
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        if not audio.size:
            return ""
        self.model.set_inference_prompt("fa-IR")
        buffer = CacheAwareStreamingAudioBuffer(
            model=self.model, online_normalization=False, pad_and_drop_preencoded=False
        )
        buffer.append_audio(audio, stream_id=-1)
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        hypotheses, prediction = None, None
        text = ""
        with torch.inference_mode():
            for step, (chunk, length) in enumerate(buffer):
                prediction, texts, *cache, hypotheses = self.model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=length,
                    cache_last_channel=cache[0],
                    cache_last_time=cache[1],
                    cache_last_channel_len=cache[2],
                    keep_all_outputs=buffer.is_buffer_empty(),
                    previous_hypotheses=hypotheses,
                    previous_pred_out=prediction,
                    drop_extra_pre_encoded=0 if step == 0 else self.model.encoder.streaming_cfg.drop_extra_pre_encoded,
                    return_transcription=True,
                )
                if texts:
                    text = _extract_text(texts[0]).strip()
        return text

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        text = self._transcribe(audio)
        language_code = self.language
        if self._detects_utterance_language:
            text, detected = _split_lang_tag(text)
            if detected is not None:
                language_code = detected
                self.last_language = detected
        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return
        console.print(f"[yellow]USER: {text}")
        yield Transcription(
            text=text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.speech_end_at_s,
        )
