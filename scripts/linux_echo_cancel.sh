#!/usr/bin/env bash
# Route the default microphone and speaker through the WebRTC acoustic echo
# canceller of PulseAudio or PipeWire (through pipewire-pulse), so
# `speech-to-speech local` can be interrupted by voice without headphones.
#
# Usage:
#   scripts/linux_echo_cancel.sh enable [SOURCE] [SINK]   # defaults: current default devices
#   scripts/linux_echo_cancel.sh disable                  # unload and restore previous defaults
#   scripts/linux_echo_cancel.sh status
#
# The change lasts until `disable`, a logout, or an audio server restart.
set -euo pipefail

SOURCE_NAME=s2s_echo_cancel_source
SINK_NAME=s2s_echo_cancel_sink
STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/s2s-echo-cancel.state"

die() {
    echo "error: $*" >&2
    exit 1
}

command -v pactl >/dev/null || die "pactl not found; install pulseaudio-utils (Debian/Ubuntu) or pipewire-pulse."
pactl info >/dev/null 2>&1 || die "no PulseAudio-compatible server is running."

server_name() {
    pactl info | sed -n 's/^Server Name: //p'
}

module_id() {
    pactl list short modules | awk -v sink="sink_name=$SINK_NAME" '$2 == "module-echo-cancel" && index($0, sink) { print $1; exit }'
}

enable() {
    if [[ -n "$(module_id)" ]]; then
        echo "Echo cancellation is already enabled."
        status
        return
    fi

    local default_source default_sink master_source master_sink
    default_source="$(pactl get-default-source)"
    default_sink="$(pactl get-default-sink)"
    master_source="${1:-$default_source}"
    master_sink="${2:-$default_sink}"
    [[ "$master_source" == *.monitor ]] && die "default source '$master_source' is a monitor; pass a microphone source explicitly."

    local -a pulse_only_args=()
    if [[ "$(server_name)" != *PipeWire* ]]; then
        # PulseAudio only: keep the device format, and disable analog gain control
        # so the canceller does not move the hardware mic slider. pipewire-pulse
        # uses its own WebRTC defaults.
        pulse_only_args=(
            use_master_format=1
            aec_args='"analog_gain_control=0 digital_gain_control=1 noise_suppression=1"'
        )
    fi

    pactl load-module module-echo-cancel \
        aec_method=webrtc \
        "${pulse_only_args[@]}" \
        source_master="$master_source" \
        sink_master="$master_sink" \
        source_name="$SOURCE_NAME" \
        sink_name="$SINK_NAME" \
        source_properties="device.description=Echo-Cancelled-Microphone" \
        sink_properties="device.description=Echo-Cancelled-Speaker" >/dev/null

    printf '%s\n%s\n' "$default_source" "$default_sink" >"$STATE_FILE"
    pactl set-default-source "$SOURCE_NAME"
    pactl set-default-sink "$SINK_NAME"
    echo "Enabled WebRTC echo cancellation on $(server_name)."
    echo "  microphone: $master_source"
    echo "  speaker:    $master_sink"
    echo "Run 'speech-to-speech local ...' without --local_audio_input_device/--local_audio_output_device."
    # Volume changes on the echo-cancelled sink do not clear a mute on the real one.
    if [[ "$(pactl get-sink-mute "$master_sink")" == *yes* ]]; then
        echo "warning: $master_sink is muted; unmute it with: pactl set-sink-mute $master_sink 0" >&2
    fi
}

disable() {
    local id
    id="$(module_id)"
    if [[ -z "$id" ]]; then
        echo "Echo cancellation is not enabled."
        return
    fi
    pactl unload-module "$id"
    if [[ -f "$STATE_FILE" ]]; then
        { read -r source; read -r sink; } <"$STATE_FILE"
        pactl set-default-source "$source" 2>/dev/null || true
        pactl set-default-sink "$sink" 2>/dev/null || true
        rm -f "$STATE_FILE"
    fi
    echo "Disabled echo cancellation."
}

status() {
    echo "Server:         $(server_name)"
    echo "Default source: $(pactl get-default-source)"
    echo "Default sink:   $(pactl get-default-sink)"
    if [[ -n "$(module_id)" ]]; then
        echo "Echo cancel:    enabled (module $(module_id))"
    else
        echo "Echo cancel:    disabled"
    fi
}

case "${1:-}" in
    enable) shift; enable "$@" ;;
    disable) disable ;;
    status) status ;;
    *) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
