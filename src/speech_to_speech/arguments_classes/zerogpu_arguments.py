"""Configuration for the optional shared ZeroGPU model worker."""

from dataclasses import dataclass, field

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class ZeroGPUSTTArguments:
    zerogpu_stt_model_name: str = field(
        default="nvidia/parakeet-tdt-0.6b-v3", metadata={"help": "Parakeet model loaded by the ZeroGPU application."}
    )


@dataclass
class ZeroGPULLMArguments(LanguageModelBaseArguments):
    compact_history: bool = field(
        default=False, metadata={"help": "Background compaction is not supported by the experimental ZeroGPU backend."}
    )
    stream_batch_sentences: int = field(default=1, metadata={"help": "Sentences per streamed batch."})
    zerogpu_llm_gen_max_new_tokens: int = field(
        default=128, metadata={"help": "Maximum generated tokens per response."}
    )
    zerogpu_llm_gen_do_sample: bool = field(default=False, metadata={"help": "Sample generated tokens."})


@dataclass
class ZeroGPUTTSArguments:
    zerogpu_tts_backend: str | None = field(
        default=None,
        metadata={"help": "Native TTS backend selected by the initialized ZeroGPU runtime (qwen3 or kokoro)."},
    )
    zerogpu_tts_model_name: str | None = field(
        default=None, metadata={"help": "Must match the model initialized in the worker; defaults to its model."}
    )
    zerogpu_tts_voice: str | None = field(
        default=None, metadata={"help": "Voice ID understood by the selected TTS backend."}
    )
    zerogpu_tts_language: str | None = field(
        default=None, metadata={"help": "Speech language; defaults to the selected backend's language."}
    )
