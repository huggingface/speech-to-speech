"""Compatibility import for the independently installable Smart Turn analyzer."""

from hf_s2s_smart_turn.analyzer import (
    MAX_AUDIO_SECONDS,
    MODEL_FILENAME,
    MODEL_REPO_ID,
    MODEL_SAMPLE_RATE,
    MODEL_VERSION,
    SmartTurnAnalyzer,
    SmartTurnResult,
)

__all__ = [
    "MAX_AUDIO_SECONDS",
    "MODEL_FILENAME",
    "MODEL_REPO_ID",
    "MODEL_SAMPLE_RATE",
    "MODEL_VERSION",
    "SmartTurnAnalyzer",
    "SmartTurnResult",
]
