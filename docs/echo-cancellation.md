# Echo cancellation for local voice chat

By default, `speech-to-speech local` reads the microphone and plays replies
through the default PortAudio devices. It does not cancel acoustic echo itself,
so on laptop speakers the assistant can hear its own voice. The VAD then treats
that voice as a barge-in and cuts the reply off.

You have three options:

- Use headphones.
- Pass `--local_audio_block_mic_during_playback`. This stops the feedback but
  also stops you from interrupting the assistant by voice.
- Run the audio through an echo canceller. This keeps voice interruptions
  working without headphones. On Linux, use the system echo canceller below.

## Linux: PulseAudio or PipeWire WebRTC echo canceller

PulseAudio's `module-echo-cancel` can create a virtual microphone and speaker
pair backed by the WebRTC audio processing library. PipeWire supports the same
module through `pipewire-pulse`. Audio played to the virtual speaker is
subtracted from the virtual microphone. If you make the pair your default
devices, `speech-to-speech local` uses it without extra flags.

### Quick setup

From a source checkout:

```bash
scripts/linux_echo_cancel.sh enable    # wrap the current default mic and speaker
speech-to-speech local ...             # no --local_audio_*_device flags needed
scripts/linux_echo_cancel.sh disable   # unload and restore the previous defaults
```

To wrap a different microphone or speaker, pass `enable <source> <sink>`. Find
device names with `pactl list short sources` and `pactl list short sinks`.
Running `status` shows the current defaults.

Without a checkout, run the equivalent commands directly:

```bash
pactl load-module module-echo-cancel aec_method=webrtc \
    source_name=s2s_echo_cancel_source sink_name=s2s_echo_cancel_sink
pactl set-default-source s2s_echo_cancel_source
pactl set-default-sink s2s_echo_cancel_sink
```

Undo them with `pactl unload-module module-echo-cancel`, then set your old
defaults back.

The setup lasts until you log out or restart the audio server. To load it at
every login on PulseAudio, add these lines to `~/.config/pulse/default.pa`:

```
.include /etc/pulse/default.pa
load-module module-echo-cancel aec_method=webrtc source_name=s2s_echo_cancel_source sink_name=s2s_echo_cancel_sink
set-default-source s2s_echo_cancel_source
set-default-sink s2s_echo_cancel_sink
```

On PipeWire, use the native module instead. Put this in
`~/.config/pipewire/pipewire.conf.d/echo-cancel.conf` and restart PipeWire.
PipeWire releases before 0.3.60 use `aec.method = webrtc` instead of
`library.name`.

```
context.modules = [
  { name = libpipewire-module-echo-cancel
    args = {
      library.name = aec/libspa-aec-webrtc
      source.props = { node.name = "s2s_echo_cancel_source" }
      sink.props = { node.name = "s2s_echo_cancel_sink" }
    }
  }
]
```

### Tips

- All of the assistant's audio must go through the echo-cancelled speaker.
  Other programs playing through the same speaker are cancelled too, but audio
  sent straight to the hardware device is not.
- The canceller needs about half a second of playback to adapt after it is
  loaded. Load it once and keep it loaded between turns and sessions.
- Bluetooth headsets in hands-free mode, and many conference speakerphones,
  already cancel echo in hardware. You don't need this setup with them.

### Test results

Tested on Ubuntu 22.04 with PulseAudio 15.99.1, a Lenovo Legion laptop's
built-in microphone and speakers (ALC287), and speakers at 50% volume. A
12-second speech clip played through the default output device while the default
input recorded through PortAudio, as `speech-to-speech local` does. The raw
microphone was recorded at the same time. Silero VAD ran with the default
`--thresh 0.6`.

| Capture | Echo level during playback | Longest VAD speech run |
| --- | --- | --- |
| Raw microphone | -22.7 to -9.8 dBFS | 4640 ms |
| Echo-cancelled default source, freshly loaded | -44.0 to -40.3 dBFS | 480 ms, in the first 0.7 s |
| Echo-cancelled default source, after adapting | -30.8 dBFS (below the room noise) | 0 ms (max probability 0.23) |

A barge-in needs 384 ms of continuous speech (`--min_speech_ms`). The raw
microphone would interrupt every reply. After the canceller adapts, it produces
no VAD speech at all.

Not yet tested:

- Talking over the assistant (double-talk) with a live voice.
- The PipeWire path. It uses only the `pactl` arguments that `pipewire-pulse`
  supports, but has not been run.
