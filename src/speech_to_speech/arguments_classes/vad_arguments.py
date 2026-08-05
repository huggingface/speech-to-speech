from dataclasses import dataclass, field


@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.6,
        metadata={
            "help": "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
        },
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
        },
    )
    min_silence_ms: int = field(
        default=64,
        metadata={
            "help": "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 64 ms."
        },
    )
    min_speech_ms: int = field(
        default=384,
        metadata={
            "help": "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 384 ms."
        },
    )
    min_speech_continuation_ms: int = field(
        default=192,
        metadata={
            "help": "Hysteresis threshold (ms of active speech) for accepting speech that continues a reopenable turn (soft-ended, uncommitted, within the reopen window). Set to 0 to disable the split and use min_speech_ms. Clamped to [100, min_speech_ms]. New turns and barge-ins always require min_speech_ms. Default and recommended: 192 with min_speech_ms 384."
        },
    )
    max_speech_ms: float = field(
        default=float("inf"),
        metadata={
            "help": "Maximum length of continuous speech before forcing a split. Default is infinite, allowing for uninterrupted speech segments."
        },
    )
    speech_pad_ms: int = field(
        default=500,
        metadata={
            "help": "Amount of audio retained before VAD triggers and prepended to detected speech segments. Once speech is detected, audio continues to be kept until VAD declares the segment done. Measured in milliseconds. Default is 500 ms."
        },
    )
    audio_enhancement: bool = field(
        default=False,
        metadata={
            "help": "improves sound quality by applying techniques like noise reduction, equalization, and echo cancellation. Default is False."
        },
    )
    enable_realtime_transcription: bool = field(
        default=False,
        metadata={"help": "Enable progressive audio release for live transcription during speech. Default is False."},
    )
    realtime_processing_pause: float = field(
        default=0.5,
        metadata={
            "help": "Interval (in seconds) for releasing progressive audio chunks during speech. Default is 0.5s."
        },
    )
    speculative_reopen_ms: int = field(
        default=800,
        metadata={
            "help": "In realtime mode, keep a soft-ended turn reopenable for this many milliseconds unless a response commits it. Default is 800 ms."
        },
    )
    unanswered_reopen_ms: int = field(
        default=7000,
        metadata={
            "help": "Sanity cap (ms) for reopening a soft-ended speculative turn that has not yet been answered by any assistant output. While a turn is uncommitted, resumed speech within this window reopens the same turn instead of starting a new one. Has no effect below speculative_reopen_ms and is clamped to smart_turn_max_wait_ms when Smart Turn is enabled."
        },
    )
    short_segment_merge_ms: int = field(
        default=0,
        metadata={
            "help": "When greater than 0, adjacent VAD segments below min_speech_ms are held and stitched for this many milliseconds before being discarded. Fragments shorter than 100 ms of active speech are never held. Useful with very low min_silence_ms values."
        },
    )
    smart_turn: bool = field(
        default=True,
        metadata={
            "help": "In realtime mode, use Smart Turn v3.2 after Silero finalizes a segment to choose how long assistant output remains speculative. Enabled by default; pass --no_smart_turn to disable it."
        },
    )
    smart_turn_model_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional path to a Smart Turn v3.x CPU ONNX model. When omitted, the latest supported v3.2 CPU model is downloaded from pipecat-ai/smart-turn-v3."
        },
    )
    smart_turn_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Smart Turn completion probability threshold. Higher values wait more readily on ambiguous pauses. Default is 0.5."
        },
    )
    smart_turn_max_wait_ms: int = field(
        default=2000,
        metadata={
            "help": "Speculative reopen grace used when Smart Turn reports an incomplete turn. Resumed speech creates a newer turn revision; otherwise output may commit after this delay. Default is 2000 ms."
        },
    )
    smart_turn_incomplete_delay_ms: int = field(
        default=600,
        metadata={
            "help": "Delay STT and LLM processing after Smart Turn reports an incomplete turn, allowing resumed speech to invalidate the revision before expensive work begins. This delay runs within smart_turn_max_wait_ms. Default is 600 ms."
        },
    )
    smart_turn_cpu_count: int = field(
        default=1,
        metadata={"help": "Number of CPU threads ONNX Runtime may use for each Smart Turn inference. Default is 1."},
    )
