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
    zerogpu_tts_model_name: str = field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        metadata={"help": "Qwen CustomVoice model loaded by the ZeroGPU application."},
    )
    zerogpu_tts_speaker: str = field(default="Aiden", metadata={"help": "Qwen CustomVoice speaker."})
    zerogpu_tts_language: str = field(default="english", metadata={"help": "Language for speech synthesis."})
