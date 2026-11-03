"""Worker-side TTS selection, independent of the shared GPU lease executor.

Providers adapt model loading and allocation-time initialization. Audio still
comes from each model's existing native speech-to-speech handler.
"""

from copy import copy
from typing import Protocol

PROVIDER_DEFAULTS = {
    "qwen3": {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "voice": "Aiden"},
    "kokoro": {"model": "hexgrad/Kokoro-82M", "voice": "bm_fable"},
}
SUPPORTED_BACKENDS = frozenset(PROVIDER_DEFAULTS)


class TTSProvider(Protocol):
    model_name: str
    default_voice: str
    default_language: str
    supported_languages: frozenset[str]

    def resolve_voice(self, requested, current): ...
    def initialize_lease(self): ...
    def gpu_models(self): ...
    def warmup(self): ...
    def synthesize_chunks(self, text, language=None, speaker=None): ...
    def supported_speakers(self): ...


class QwenTTSProvider:
    def __init__(self, model_name, speaker, *, language=None, speed=1.0):
        from speech_to_speech.TTS.qwen3_tts_handler import QWEN3_LANGUAGE_ALIASES

        from . import _model_qwen_tts

        if speed != 1.0:
            raise ValueError("Qwen3 TTS does not support the Kokoro speed setting")
        self.native = _model_qwen_tts
        self.handler = _model_qwen_tts.load_tts(model_name, speaker)
        self.model_name = model_name
        self.default_voice = speaker
        self.default_language = self.handler._normalize_language(language or "english")
        self.handler.language = self.default_language
        self.supported_languages = frozenset(code for code in QWEN3_LANGUAGE_ALIASES if len(code) == 2)

    def initialize_lease(self):
        self.native.initialize_tts_graphs(self.handler)

    def gpu_models(self):
        return self.native.gpu_models(self.handler)

    def warmup(self):
        for _ in self.synthesize_chunks("Hello."):
            pass

    def resolve_voice(self, requested, current):
        if not requested:
            return current or self.default_voice
        from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

        # Native metadata validation on a shallow copy avoids mutating the
        # worker's handler from the application's session thread.
        handler = copy(self.handler)
        handler.speaker = current or self.default_voice
        handler._apply_session_voice_override(
            "custom_voice",
            response=RealtimeResponseCreateParams(audio={"output": {"voice": requested}}),
        )
        return handler.speaker

    def synthesize_chunks(self, text, language=None, speaker=None):
        language = self.handler._language_for_utterance(language, language)
        chunks = self.native.synthesize_chunks16k(
            self.handler,
            text,
            language,
            speaker or self.default_voice,
        )
        produced = False
        for chunk in chunks:
            produced = True
            yield chunk
        if not produced:
            raise RuntimeError("Qwen3 TTS returned no audio")

    def supported_speakers(self):
        return self.handler._supported_speakers()


def create_provider(backend, model_name=None, speaker=None, *, language=None, speed=1.0) -> TTSProvider:
    if backend not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported ZeroGPU TTS backend: {backend!r}; choose 'qwen3' or 'kokoro'")
    model_name = model_name or PROVIDER_DEFAULTS[backend]["model"]
    speaker = speaker or PROVIDER_DEFAULTS[backend]["voice"]
    if backend == "qwen3":
        return QwenTTSProvider(model_name, speaker, language=language, speed=speed)
    if backend == "kokoro":
        from ._model_kokoro import KokoroTTSProvider

        return KokoroTTSProvider(model_name, speaker, language=language, speed=speed)
    raise ValueError(f"Unsupported ZeroGPU TTS backend: {backend!r}; choose 'qwen3' or 'kokoro'")
