#!/usr/bin/env bash
# Launch the Hugging Face realtime speech pipeline and its browser interface.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SPEECH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$SPEECH_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
UI_DIR="$SPEECH_ROOT/demo"
S2S_PORT="${S2S_PORT:-8765}"
UI_PORT="${UI_PORT:-7860}"
HF_S2S_STT="${HF_S2S_STT:-openai}"
HF_S2S_TTS="${HF_S2S_TTS:-openai}"
CONVFILL_CONFIG="${CONVFILL_CONFIG:-configs/demo_mode/convfill_hosted_config.json}"
CONVFILL_MODE="${CONVFILL_MODE:-normal}"

cd "$REPO_ROOT"

# Resolve the interpreter: an explicit override, an already-activated venv, or
# the repository .venv created by uv sync.
if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$SPEECH_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$SPEECH_ROOT/.venv/bin/python"
else
  echo "[error] No Python environment found for the speech demo." >&2
  echo "Run uv sync from the repository root." >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] PYTHON_BIN='$PYTHON_BIN' is not an executable interpreter." >&2
  exit 1
fi

if [ -z "${NLTK_DATA:-}" ]; then
  NLTK_DATA="$("$PYTHON_BIN" -c 'import sys, os; print(os.path.join(sys.prefix, "share", "nltk_data"))')"
fi
export NLTK_DATA

if [ ! -d "$UI_DIR/node_modules" ]; then
  echo "[error] Browser dependencies are missing. Run:" >&2
  echo "  npm ci --prefix demo" >&2
  echo "or, for the browser interface alone:" >&2
  echo "  npm ci --prefix demo" >&2
  exit 1
fi

export CONVFILL_PIPELINE_URL="ws://127.0.0.1:${S2S_PORT}/v1/realtime"
export STARTUP_GREETING="${STARTUP_GREETING-Greet the user warmly in one short sentence, then invite them to talk.}"

S2S_PID=""
UI_PID=""

cleanup() {
  echo
  echo "[shutdown] stopping voice demo..."
  [ -n "$S2S_PID" ] && kill "$S2S_PID" 2>/dev/null || true
  [ -n "$UI_PID" ] && kill "$UI_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[speech] starting Hugging Face speech-to-speech on port ${S2S_PORT}"
echo "[speech] STT=${HF_S2S_STT} LLM=ConvFill(${CONVFILL_CONFIG}:${CONVFILL_MODE}) TTS=${HF_S2S_TTS}"
speech_args=()
if [ "$HF_S2S_STT" = openai ]; then
  speech_args+=(--openai_stt_base_url "${CONVFILL_STT_URL:?Set CONVFILL_STT_URL including /v1}" --openai_stt_model "${CONVFILL_STT_MODEL:-Qwen/Qwen3-ASR-1.7B}" --openai_stt_timeout 30)
  [ -z "${CONVFILL_STT_API_KEY:-}" ] || speech_args+=(--openai_stt_api_key "$CONVFILL_STT_API_KEY")
fi
if [ "$HF_S2S_TTS" = openai ]; then
  speech_args+=(--openai_tts_base_url "${CONVFILL_TTS_URL:?Set CONVFILL_TTS_URL including /v1}" --openai_tts_model "${CONVFILL_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}" --openai_tts_voice "${CONVFILL_TTS_VOICE:-aiden}" --openai_tts_stream true --openai_tts_timeout 30)
  [ -z "${CONVFILL_TTS_API_KEY:-}" ] || speech_args+=(--openai_tts_api_key "$CONVFILL_TTS_API_KEY")
fi
"$PYTHON_BIN" -m convfill_demo.pipeline serve \
  --host 127.0.0.1 \
  --port "$S2S_PORT" \
  --stt "$HF_S2S_STT" \
  --llm_backend convfill \
  --convfill_config_path "$CONVFILL_CONFIG" \
  --convfill_mode "$CONVFILL_MODE" \
  --tts "$HF_S2S_TTS" \
  --num_pipelines 1 \
  --speculative_reopen_ms "${CONVFILL_COMPLETE_HOLD_MS:-500}" \
  --smart_turn_max_wait_ms "${CONVFILL_INCOMPLETE_HOLD_MS:-1700}" \
  --enable_live_transcription "${speech_args[@]}" &
S2S_PID=$!

echo -n "[speech] waiting for :${S2S_PORT}"
for _ in $(seq 1 240); do
  if (echo >/dev/tcp/127.0.0.1/"$S2S_PORT") >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  if ! kill -0 "$S2S_PID" 2>/dev/null; then
    echo
    echo "[error] speech-to-speech exited during startup." >&2
    wait "$S2S_PID"
  fi
  echo -n "."
  sleep 0.5
done

if ! (echo >/dev/tcp/127.0.0.1/"$S2S_PORT") >/dev/null 2>&1; then
  echo
  echo "[error] speech-to-speech did not become ready within 120 seconds." >&2
  exit 1
fi

echo "[interface] starting on http://127.0.0.1:${UI_PORT}"
"$PYTHON_BIN" -m uvicorn convfill_demo.gateway:app \
  --host "${UI_HOST:-127.0.0.1}" --port "$UI_PORT" --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" --no-access-log &
UI_PID=$!

echo
echo "ConvFill realtime voice demo is running."
echo "  open: http://127.0.0.1:${UI_PORT}"
echo "  press Ctrl+C to stop both processes"
echo

# Supervise both children: a failed pipeline must stop the public UI too.
while kill -0 "$S2S_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do
  sleep 1
done
exit 1
