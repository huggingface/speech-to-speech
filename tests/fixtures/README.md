# Echo-cancellation speech fixture

`aec-speech.wav` contains the first twelve seconds of the reference recording
already shipped at `src/speech_to_speech/TTS/ref_audio.wav`, mixed to mono,
resampled to 16 kHz, and scaled to signed 16-bit PCM. The test reuses a later
segment as an uncorrelated near voice during playback. No new external or
private voice recording is included.

From the repository root, with NumPy, soundfile, SciPy and soxr installed:

```bash
.venv/bin/python - <<'PY'
import soundfile as sf
import soxr
import numpy as np
from scipy.io import wavfile

speech, rate = sf.read("src/speech_to_speech/TTS/ref_audio.wav", dtype="float32")
speech = soxr.resample(speech.mean(axis=1), rate, 16000)
wavfile.write("tests/fixtures/aec-speech.wav", 16000, (speech[:12 * 16000] * 12000).astype(np.int16))
PY
```
