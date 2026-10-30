# Echo cancellation

`--echo-cancellation` makes the packaged `talk` and `local` clients remove
their own speaker playback from the microphone, so you can interrupt the
assistant by voice without headphones. It stays off by default; headphones or
a headset don't need it.

## How it works

The client uses `pywebrtc-audio`, which wraps WebRTC's echo canceller (AEC3).
Noise suppression and gain control stay off. The binding applies its
high-pass filter whenever echo cancellation is on.

On Windows, cancellation uses the default WASAPI input and output devices.
If you set `--input-device` or `--output-device`, select WASAPI devices; MME
does not provide capture timestamps. Shared-mode conversion keeps the client's
sample rate even when the Windows mixer uses another rate.

With cancellation enabled, a separate audio process owns the device streams.
Model and tool code in the parent cannot hold its Python execution lock.
Networking and tool execution stay in the parent. Python scripts that call the
client directly must protect their entry point with `if __name__ == "__main__":`
so the audio process can start safely.

`--echo-cancellation` cannot be combined with
`--local_audio_block_mic_during_playback`, which pauses the microphone during
playback and so disables voice interruptions.

## Limits

- Echo cancellation in the packaged client requires `--send-rate` and
  `--recv-rate` to match: both 16 kHz, their default, or both 24 kHz.
- The client does not correct clock drift between separate input and output
  devices.
- While the assistant speaks, the canceller can turn the user's voice down
  along with the echo. Devices, volume, room acoustics and load affect how
  well cancellation works.
