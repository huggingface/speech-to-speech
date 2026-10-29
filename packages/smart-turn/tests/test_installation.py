"""COMMITMENT: the standalone API does not require the conversation pipeline.

WHO BREAKS: hosts installing only the Smart Turn component.
SEAM: installed package -> public imports in a fresh Python process.
SPEC: packages/smart-turn/README.md.
DOES NOT PROVE: model accuracy; the full application's analyzer tests cover
preprocessing and ONNX invocation through its preserved import path.
"""

import subprocess
import sys


def test_public_api_imports_without_loading_the_pipeline() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from hf_s2s_smart_turn import SmartTurnAnalyzer, SmartTurnResult; "
            "import importlib.util, sys; "
            "assert importlib.util.find_spec('speech_to_speech') is None; "
            "assert not any(name.startswith('speech_to_speech') for name in sys.modules); "
            "assert callable(SmartTurnAnalyzer); "
            "assert SmartTurnResult(True, 0.9, 1.0).complete",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
