#!/bin/zsh
# Colab GPU 음성 서버에 Mac 마이크·스피커로 접속한다 (모델은 Colab에서 실행).
#
# 사용법: ~/speech-to-speech/run_talk_colab.sh wss://xxxx.trycloudflare.com/v1/realtime
#   주소는 ajou-vis/server/colab_voice_server.ipynb 의 8번 셀이 출력한다.
#   헤드폰 권장 (스피커 소리가 마이크로 다시 들어가면 끼어들기로 인식될 수 있음). 종료: Ctrl+C
if [[ -z "$1" ]]; then
  echo "사용법: $0 wss://xxxx.trycloudflare.com/v1/realtime" >&2
  exit 1
fi
cd "$(dirname "$0")"
exec .venv/bin/speech-to-speech talk \
  --url "$1" \
  --block-mic-during-playback \
  2>&1 | tee -a run_talk_colab.log
