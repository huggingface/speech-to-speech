#!/bin/sh
set -eu

SPEECH_TO_SPEECH_VERSION="0.2.13"
UV_VERSION="0.11.30"
PYTHON_VERSION="3.11.13"
UV_SHA256="2b9e582af54f84fa50c115427451a6c13e80f43b52f8282b8af5791077317bbf"
CONSTRAINTS_SHA256="729a123b7a7119f78c4c21eb582bf6f6fe535850346b3eb588af8783ed18b8ab"
RELEASE_REF="v${SPEECH_TO_SPEECH_VERSION}"
REPOSITORY_RAW="https://raw.githubusercontent.com/huggingface/speech-to-speech/${RELEASE_REF}"
UV_ARCHIVE_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-apple-darwin.tar.gz"

fail() {
    printf 'speech-to-speech installer: %s\n' "$1" >&2
    exit 1
}

[ "$(uname -s)" = "Darwin" ] || fail "this installer supports macOS only"
[ "$(uname -m)" = "arm64" ] || fail "native Apple Silicon (arm64) is required"
translated="$(sysctl -in sysctl.proc_translated 2>/dev/null || printf '0')"
[ "$translated" != "1" ] || fail "Rosetta was detected; open a native arm64 terminal and rerun"

available_kib="$(df -Pk "$HOME" | awk 'NR == 2 {print $4}')"
[ -n "$available_kib" ] || fail "could not determine free disk space"
[ "$available_kib" -ge 4194304 ] || fail "at least 4 GiB of free disk space is required to bootstrap"

if [ "${S2S_INSTALL_TEST_ONLY:-0}" = "1" ]; then
    exit 0
fi

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v shasum >/dev/null 2>&1 || fail "shasum is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"

app_root="$HOME/Library/Application Support/speech-to-speech"
log_dir="$HOME/Library/Logs/speech-to-speech"
bin_dir="$HOME/.local/bin"
mkdir -p "$app_root/bin" "$app_root/tools" "$log_dir" "$bin_dir"
log_file="$log_dir/install.log"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/speech-to-speech-install.XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

printf 'Installing speech-to-speech %s with uv %s and Python %s\n' \
    "$SPEECH_TO_SPEECH_VERSION" "$UV_VERSION" "$PYTHON_VERSION" | tee -a "$log_file"

uv_archive="$temporary_dir/uv.tar.gz"
curl -fL --retry 3 --proto '=https' --tlsv1.2 "$UV_ARCHIVE_URL" -o "$uv_archive" >>"$log_file" 2>&1
actual_uv_sha="$(shasum -a 256 "$uv_archive" | awk '{print $1}')"
[ "$actual_uv_sha" = "$UV_SHA256" ] || fail "uv archive checksum verification failed"
tar -xzf "$uv_archive" -C "$temporary_dir"
uv_downloaded="$(find "$temporary_dir" -type f -name uv -perm -u+x -print -quit)"
[ -n "$uv_downloaded" ] || fail "the uv archive did not contain an executable"
cp "$uv_downloaded" "$app_root/bin/uv"
chmod 755 "$app_root/bin/uv"
uv_bin="$app_root/bin/uv"

constraints="$temporary_dir/macos-arm64-constraints.txt"
curl -fL --retry 3 --proto '=https' --tlsv1.2 \
    "$REPOSITORY_RAW/scripts/macos-arm64-constraints.txt" -o "$constraints" >>"$log_file" 2>&1
actual_constraints_sha="$(shasum -a 256 "$constraints" | awk '{print $1}')"
[ "$actual_constraints_sha" = "$CONSTRAINTS_SHA256" ] || fail "dependency constraints checksum verification failed"

"$uv_bin" python install "$PYTHON_VERSION" >>"$log_file" 2>&1
UV_TOOL_DIR="$app_root/tools" UV_TOOL_BIN_DIR="$bin_dir" "$uv_bin" tool install --force \
    --python "$PYTHON_VERSION" \
    --constraints "$constraints" \
    "speech-to-speech==${SPEECH_TO_SPEECH_VERSION}" >>"$log_file" 2>&1

printf 'Environment installed. Starting guided setup…\n' | tee -a "$log_file"
"$bin_dir/speech-to-speech" setup
