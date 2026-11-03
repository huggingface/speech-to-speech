"""Native parent-side Qwen handler with only synthesis offloaded."""

from time import perf_counter
from types import SimpleNamespace

import torch

from speech_to_speech.pipeline.turn_latency import active_turn_latency_tracker
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
from speech_to_speech.zerogpu import runtime


class ZeroGPUTTSHandler(Qwen3TTSHandler):
    def setup(
        self,
        should_listen,
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        speaker="Aiden",
        language="english",
        gen_kwargs=None,
        **kwargs,
    ):
        if model_name != runtime.get_model_ids()["tts"]:
            raise ValueError("ZeroGPU TTS model must match runtime.initialize(tts_model=...).")
        super().setup(
            should_listen,
            model_name=model_name,
            speaker=speaker,
            language=language,
            device="cuda",
            dtype="bfloat16",
            backend="torch",
            streaming_chunk_size=4,
            max_new_tokens=512,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )

    def _setup_faster(self, model_name, dtype, attn_implementation, backend):
        # Session/voice settings still belong to the native handler. No model
        # or CUDA state is required by its parent-side process() method.
        self.dtype = torch.bfloat16
        self.model = SimpleNamespace(sample_rate=24000)

    def warmup(self):
        pass

    def _model_type(self):
        return "custom_voice"

    def _supported_speakers(self):
        # This reads checkpoint metadata, never performs CUDA inference.
        return runtime.get_tts_speakers()

    def _process_custom_voice(self, text, language=None):
        generation = self.cancel_scope.generation if self.cancel_scope else None
        started = perf_counter()
        first = True
        chunks = runtime.synthesize_chunks(text, language or self.language, self.speaker)
        try:
            for chunk in chunks:
                if self.cancel_scope and self.cancel_scope.is_stale(generation):
                    return
                if first:
                    tracker = active_turn_latency_tracker()
                    if tracker is not None:
                        tracker.record_tts_ttfa(perf_counter() - started)
                    first = False
                # Child's native Qwen _stream already converts/resamples to
                # pipeline PCM16/16kHz and preserves streaming chunk boundaries.
                yield chunk
        finally:
            close = getattr(chunks, "close", None)
            if close:
                close()
