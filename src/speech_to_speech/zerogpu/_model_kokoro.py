"""Native Kokoro synthesis with deferred warmup and a shared CUDA model."""

from queue import Queue
from threading import Event

import torch
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.kokoro_handler import KOKORO_NATIVE_LANGUAGES, KokoroTTSHandler


class DeferredKokoro(KokoroTTSHandler):
    def setup(self, should_listen, model_name="hexgrad/Kokoro-82M", **kwargs):
        self.model_name = model_name
        super().setup(should_listen, model_name=model_name, **kwargs)

    def warmup(self):
        # Native setup runs during ZeroGPU emulation, before a real allocation.
        pass

    def _setup_kokoro(self):
        from kokoro import KModel, KPipeline

        self.backend = "kokoro"
        # Explicit model sharing also avoids KPipeline's CUDA capability query
        # when constructing additional language pipelines inside a live lease.
        self.gpu_model = KModel(repo_id=self.model_name).to("cuda").eval()
        self.pipeline = KPipeline(lang_code=self.lang_code, repo_id=self.model_name, model=self.gpu_model)
        self.pipeline.load_voice(self.voice)
        self.pipelines = {self.lang_code: self.pipeline}

    def select_pipeline(self, lang_code):
        if lang_code not in self.pipelines:
            from kokoro import KPipeline

            self.pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id=self.model_name,
                model=self.gpu_model,
            )
        self.pipeline = self.pipelines[lang_code]
        self.lang_code = lang_code


def _language_code(language):
    return {"english": "en", "en-us": "en", "en-gb": "en"}.get(language, language)


class KokoroTTSProvider:
    def __init__(self, model_name, speaker):
        # Kokoro voice names encode their phonemizer language (a/b English,
        # f French, etc.). Native request handling maps ISO language codes.
        lang_code = speaker[0] if speaker and speaker[0] in "abefhijpz" else "b"
        self.model_name = model_name
        self.default_voice = speaker
        self.default_language = {
            "a": "en",
            "b": "en",
            "e": "es",
            "f": "fr",
            "h": "hi",
            "i": "it",
            "j": "ja",
            "p": "pt",
            "z": "zh",
        }[lang_code]
        self.supported_languages = KOKORO_NATIVE_LANGUAGES
        self.handler = DeferredKokoro(
            Event(),
            Queue(),
            Queue(),
            setup_kwargs={
                "should_listen": Event(),
                "model_name": model_name,
                "device": "cuda",
                "voice": speaker,
                "lang_code": lang_code,
            },
        )

    def resolve_voice(self, requested, current):
        return self.handler._resolve_requested_voice(requested) or current or self.default_voice

    def gpu_models(self):
        return {"tts": self.handler.gpu_model}

    def initialize_lease(self):
        if not torch.cuda.is_available() or next(self.handler.gpu_model.parameters()).device.type != "cuda":
            raise RuntimeError("Kokoro TTS requires an allocated CUDA GPU; CPU fallback is disabled")

    def warmup(self):
        for _ in self.synthesize_chunks("Hello."):
            pass

    def synthesize_chunks(self, text, language=None, speaker=None):
        self.initialize_lease()
        handler = self.handler
        language = _language_code(language) or self.default_language
        voice = handler._resolve_requested_voice(speaker) or handler._initial_voice
        # Keep allocation-local caches separate from conversation state: every
        # request starts with configured defaults, then native voice selection.
        handler.voice = handler._initial_voice
        handler.lang_code = handler._initial_lang_code
        lang_code = handler._lang_and_voice_for(language, voice)[0] if language else handler.lang_code
        handler.select_pipeline(lang_code)
        with torch.inference_mode():
            # Reuse native resampling, PCM16 conversion, chunking and voice
            # fallback. Preselecting its pipeline prevents a second model load.
            produced = False
            request = TTSInput(
                text=text,
                language_code=language,
                response=RealtimeResponseCreateParams(audio={"output": {"voice": voice}}),
            )
            for chunk in handler.process(request):
                produced = True
                yield chunk
            if not produced:
                raise RuntimeError("Kokoro TTS returned no audio")
