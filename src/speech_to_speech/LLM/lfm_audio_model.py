"""LFM3-Audio: one model that hears the user and speaks its reply.

The model takes the conversation, including the user's audio, and generates
interleaved text and speech. Each response chunk carries the text and the
speech generated with it, so ``--tts none`` can play the model's own voice.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import wave
from collections.abc import Iterator
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Any, Optional, cast

import numpy as np
import torch
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from packaging.version import Version
from transformers import PreTrainedTokenizerBase
from transformers.generation.streamers import BaseStreamer

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.utils import StreamingPcm16Resampler
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler, StreamContext
from speech_to_speech.pipeline.messages import LLMResponseChunk
from speech_to_speech.pipeline.turn_latency import active_turn_latency_tracker
from speech_to_speech.utils.utils import response_wants_audio

logger = logging.getLogger(__name__)

MIN_TRANSFORMERS_VERSION = "5.20.2"
MODEL_SAMPLE_RATE = 24_000
# Speech frames are 80 ms. Decode the first half second early to start playback
# sooner, then whole interleaved blocks of twelve frames.
FIRST_DECODE_FRAMES = 6
DECODE_FRAMES = 12
# Fresh attempts when a reply's first speech slot ends speech (rare sampling outcome).
SPEECH_RETRIES = 2
# The model card's instruction for voice chat. Without it the model sometimes
# ends speech in its first speech slot and finishes the reply as text.
INTERLEAVED_INSTRUCTION = "Respond with interleaved text and audio."
# Sampling settings from the model card for interleaved text and speech.
AUDIO_GEN_KWARGS = {
    "audio_temperature": 1.0,
    "audio_top_k": 128,
    "audio_repetition_penalty": 1.1,
    "audio_repetition_window": 4096,
}


class _GenerationCancelled(Exception):
    """Raised inside generate() to stop it; the model has no stopping criteria."""


class _QueueStreamer(BaseStreamer):
    def __init__(self, events: Queue[tuple[str, Any]], kind: str, cancel: Event, *, skip_prompt: bool) -> None:
        self._events = events
        self._kind = kind
        self._cancel = cancel
        self._skip_prompt = skip_prompt

    def put(self, value: torch.Tensor) -> None:
        if self._cancel.is_set():
            raise _GenerationCancelled
        if self._skip_prompt:
            self._skip_prompt = False
            return
        self._events.put((self._kind, value.detach().reshape(-1).clone()))

    def end(self) -> None:
        pass


def _wav_base64_to_float(audio_b64: str) -> np.ndarray:
    with wave.open(io.BytesIO(base64.b64decode(audio_b64)), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16_000):
            raise ValueError("LFM3-Audio expects mono 16 kHz PCM16 input audio")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _model_conversation(chat: Chat) -> list[dict[str, Any]]:
    """Render chat history in the processor's format, keeping user audio as audio."""
    messages: list[dict[str, Any]] = []
    for message in chat.to_transformers_chat():
        content = message.get("content")
        if message.get("role") == "tool":
            messages.append({"role": "tool", "content": content})
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "input_audio" and part.get("audio"):
                    parts.append({"type": "audio", "audio": _wav_base64_to_float(part["audio"])})
                elif part.get("type") == "input_text" and part.get("text"):
                    parts.append({"type": "text", "text": part["text"]})
            messages.append({"role": message["role"], "content": parts})
        else:
            messages.append({"role": message["role"], "content": content or ""})
    return messages


class LFMAudioModelHandler(BaseLanguageModelHandler):
    """Speech-in, speech-out handler for LiquidAI LFM3-Audio checkpoints (transformers)."""

    tokenizer: PreTrainedTokenizerBase

    def setup(self, **kwargs: Any) -> None:  # type: ignore[override]
        super().setup(**kwargs)
        self.warmup()

    def _load_model(self, model_name: str, device: str, torch_dtype: str, gen_kwargs: dict[str, Any]) -> None:
        import transformers
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
            raise ImportError(
                f"LFM3-Audio requires transformers>={MIN_TRANSFORMERS_VERSION}; found {transformers.__version__}."
            )
        from speech_to_speech.api.openai_realtime.handlers.audio import PIPELINE_SAMPLE_RATE

        self.output_sample_rate = PIPELINE_SAMPLE_RATE
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name, dtype=getattr(torch, torch_dtype), device_map=device, trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.codec = self.processor.audio_tokenizer.to(device=device).eval()
        self._codec_cache_class = sys.modules[type(self.codec).__module__].LiquidCodecStreamingCache
        self.gen_kwargs = {"max_new_tokens": gen_kwargs.get("max_new_tokens", 1024)}
        self.interleaved_n_text = self.model.config.interleaved_n_text

    def _apply_instructions(
        self,
        chat: Chat,
        instructions: Optional[str],
        raw_tools: list[Any] | None,
        tool_choice: Optional[str],
        ctx: StreamContext | None = None,
        wants_audio: bool = True,
        *,
        language_name: str | None = None,
    ) -> None:
        if raw_tools and tool_choice != "none":
            logger.warning("The lfm-audio backend does not support tools yet; ignoring %d tool(s)", len(raw_tools))
        super()._apply_instructions(
            chat, instructions, None, tool_choice, ctx, wants_audio, language_name=language_name
        )

    def _start_generation(
        self, messages: list[dict[str, Any]], speak: bool, max_new_tokens: int
    ) -> tuple[Queue[tuple[str, Any]], Event, Thread, int]:
        """Run generate() in a thread that reports text tokens and speech frames on a queue."""
        if speak:
            if messages and messages[0]["role"] == "system":
                system = {"role": "system", "content": f"{messages[0]['content']}\n\n{INTERLEAVED_INSTRUCTION}"}
                messages = [system, *messages[1:]]
            else:
                messages = [{"role": "system", "content": INTERLEAVED_INSTRUCTION}, *messages]
            messages = [*messages, {"role": "assistant", "content": "<|mixed_start|>"}]
        inputs = self.processor.apply_chat_template(
            messages,
            continue_final_message=speak,
            add_generation_prompt=not speak,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        events: Queue[tuple[str, Any]] = Queue()
        cancel = Event()
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": _QueueStreamer(events, "text", cancel, skip_prompt=True),
        }
        if speak:
            generate_kwargs.update(AUDIO_GEN_KWARGS)
            generate_kwargs["audio_streamer"] = _QueueStreamer(events, "audio", cancel, skip_prompt=False)
        else:
            generate_kwargs.update(suppress_mixed=True, suppress_audio_out=True)

        def run() -> None:
            error: BaseException | None = None
            try:
                with self._transformers_lock, torch.inference_mode():
                    self.model.generate(**generate_kwargs)
            except _GenerationCancelled:
                pass
            except BaseException as exc:
                error = exc
            events.put(("done", error))

        thread = Thread(target=run, daemon=True)
        thread.start()
        return events, cancel, thread, int(inputs["input_ids"].shape[1])

    def _generate(
        self,
        chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> Iterator[LLMResponseChunk]:
        speak = response_wants_audio(response)
        messages = _model_conversation(chat)
        max_new_tokens = self.gen_kwargs["max_new_tokens"]
        events, cancel, thread, prompt_tokens = self._start_generation(messages, speak, max_new_tokens)
        ctx.input_tokens += prompt_tokens
        if ctx.prefetch_transaction is not None:
            ctx.prefetch_transaction.register_abort(lambda: cancel.set())
        speech_retries = SPEECH_RETRIES

        text_ids: list[int] = []
        text_sent = 0
        frames: list[torch.Tensor] = []
        frames_seen = 0
        codec_cache = self._codec_cache_class(self.codec.config)
        resampler = StreamingPcm16Resampler(MODEL_SAMPLE_RATE, self.output_sample_rate)

        def chunk(final: bool) -> LLMResponseChunk | None:
            nonlocal text_sent
            text = self._decode_text(text_ids)
            # A multi-byte character can span tokens; wait for the rest of it.
            ready = len(text) if final else len(text.rstrip("�"))
            if speak and not final:
                # Speech-mode text parts are joined with spaces downstream, so
                # only release whole words.
                ready = max(text.rfind(" ", text_sent, ready), text_sent)
            delta = text[text_sent:ready]
            text_sent = ready
            ctx.raw_generated_text = text
            ctx.generated_text = text
            audio = self._decode_speech(frames, codec_cache, resampler, final) if speak else None
            frames.clear()
            if speak:
                delta = delta.strip()
            if not delta and (audio is None or not audio.size):
                return None
            return LLMResponseChunk(
                text=delta,
                audio=audio,
                language_code=language_code,
                runtime_config=runtime_config,
                response=response,
                turn_id=ctx.turn_id,
                turn_revision=ctx.turn_revision,
                speech_stopped_at_s=ctx.speech_stopped_at_s,
                cancel_generation=ctx.cancel_generation,
            )

        try:
            while True:
                try:
                    kind, value = events.get(timeout=0.05)
                except Empty:
                    if not cancel.is_set() and self._check_stop(gen, ctx):
                        cancel.set()
                    continue
                if kind == "done":
                    if value is not None:
                        raise value
                    break
                if cancel.is_set():
                    continue
                if self._check_stop(gen, ctx):
                    cancel.set()
                    continue
                if kind == "text":
                    if not text_ids and ctx.llm_start_s is not None:
                        tracker = active_turn_latency_tracker()
                        if tracker is not None and tracker.llm_ttft_s is None:
                            tracker.record_llm_ttft(perf_counter() - ctx.llm_start_s)
                    text_ids.extend(value.tolist())
                    if speak and not frames_seen and len(text_ids) > self.interleaved_n_text and speech_retries:
                        # The model sampled end-of-speech in its first speech slot
                        # and is finishing the reply as text. Nothing has been
                        # yielded yet, so resample: text is greedy, speech is not.
                        speech_retries -= 1
                        logger.info("LFM3-Audio reply started without speech; resampling")
                        cancel.set()
                        thread.join()
                        events, cancel, thread, _ = self._start_generation(messages, speak, max_new_tokens)
                        text_ids.clear()
                        continue
                    if not speak:
                        item = chunk(final=False)
                        if item is not None:
                            yield item
                    continue
                frames.append(value)
                frames_seen += 1
                if len(frames) >= (DECODE_FRAMES if frames_seen > FIRST_DECODE_FRAMES else FIRST_DECODE_FRAMES):
                    item = chunk(final=False)
                    if item is not None:
                        if not self._turn_output_allowed(ctx.turn_id, ctx.turn_revision):
                            ctx.cancelled = True
                            cancel.set()
                            continue
                        yield item
            if not ctx.interrupted:
                item = chunk(final=True)
                if item is not None:
                    yield item
        finally:
            cancel.set()
            thread.join()

    def _decode_text(self, token_ids: list[int]) -> str:
        return cast(str, self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def _decode_speech(
        self,
        frames: list[torch.Tensor],
        cache: Any,
        resampler: StreamingPcm16Resampler,
        final: bool,
    ) -> np.ndarray:
        if not frames and not final:
            return np.empty(0, dtype="<i2")
        codes = (
            torch.stack(frames, dim=-1)[None]
            if frames
            else torch.empty((1, self.model.config.num_codebooks, 0), dtype=torch.long, device=self.device)
        )
        with torch.inference_mode():
            waveform = self.codec.decode(codes, streaming_cache=cache, finalize=final).audio_values[0]
        pcm = (np.clip(waveform.float().cpu().numpy().reshape(-1), -1, 1) * 32767).astype("<i2")
        out = resampler.push(pcm.tobytes())
        if final:
            out += resampler.flush()
        return np.frombuffer(out, dtype="<i2").copy()

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        def generate(system: str, user: str) -> str:
            events, _cancel, thread, _ = self._start_generation(
                [{"role": "system", "content": system}, {"role": "user", "content": user}], False, 1024
            )
            ids: list[int] = []
            while (event := events.get())[0] != "done":
                ids.extend(event[1].tolist())
            thread.join()
            if event[1] is not None:
                raise event[1]
            return self._decode_text(ids)

        return generate

    def warmup(self) -> None:
        started = perf_counter()
        silence = np.zeros(16_000, dtype=np.float32)
        events, _cancel, thread, _ = self._start_generation(
            [{"role": "user", "content": [{"type": "audio", "audio": silence}]}], True, 48
        )
        while events.get()[0] != "done":
            pass
        thread.join()
        logger.info("LFM3-Audio warmed up in %.2fs", perf_counter() - started)
