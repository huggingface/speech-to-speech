# Headless Smart Turn

This package exposes the existing Pipecat Smart Turn v3.2 ONNX classifier without
loading the speech-to-speech conversation pipeline. The full application depends
on this package and preserves its `speech_to_speech.VAD.smart_turn` import.
Applications with their own microphone, VAD and turn policy can install only
`packages/smart-turn` and its CPU inference dependencies.

The standalone package supports the Qwen/vLLM backend's Transformers 4.57.6
environment as well as the full framework's Transformers 5 environment.

From this repository, install it with
`uv sync --project packages/smart-turn --no-dev`.

```python
import numpy as np
from hf_s2s_smart_turn import SmartTurnAnalyzer

analyzer = SmartTurnAnalyzer(model_path="/models/smart-turn-v3.2-cpu.onnx")
audio = np.asarray(recent_mono_audio, dtype=np.float32)
result = analyzer.predict(audio, sample_rate=16000)
print(result.complete, result.probability)
```

`recent_mono_audio` is the caller's processed audio. The analyzer
resamples to 16 kHz, keeps at most the latest eight seconds, left-pads shorter
input, extracts Whisper features and returns `complete`, `probability` and
`inference_ms`. `predict` is synchronous CPU work; an async application should
run it off its event loop and reject stale results after cancellation or
resumed speech.

Pass a locally verified `model_path` for a reproducible deployment. Omitting
it downloads the upstream `pipecat-ai/smart-turn-v3` model via Hugging Face
Hub, which is convenient for development but does not pin a revision. The
caller owns model/artifact pinning, VAD boundaries, audio buffering, session
identity, maximum wait and timeout policy. Smart Turn is an end-of-turn
classifier, not a replacement for VAD or STT.

Pin the preprocessing environment too. The upstream Whisper feature extractor
uses Torch when it is installed and NumPy otherwise; these paths can produce
different probabilities. Revalidate application thresholds when changing that
environment. This package preserves the upstream behavior within each runtime.

From the repository root, run the standalone development checks with:

```bash
uv sync --project packages/smart-turn --group dev
uv run --project packages/smart-turn pytest packages/smart-turn/tests -q
uv run --project packages/smart-turn ruff check packages/smart-turn
uv run --project packages/smart-turn ruff format --check packages/smart-turn
```

CI also exercises the full application's compatibility import.
