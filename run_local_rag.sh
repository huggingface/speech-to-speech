#!/bin/zsh
# AJOU-VIS 한국어 음성 파이프라인 + RAG 서버(치토) 연결 실행 스크립트
#
# 사용법 (터미널 2개):
#   터미널 1:  ~/ajou-vis/server/run_server.sh      ← RAG 서버 (포트 8100)
#   터미널 2:  ~/speech-to-speech/run_local_rag.sh  ← 이 스크립트
#
# 말투 지시: TTS_INSTRUCT="..." ./run_local_rag.sh (기본은 사용 안 함)
# 구성: whisper-large-v3-turbo(MLX) STT + 언어 자동 감지
#      + LLM 슬롯 → RAG 서버(rewrite → 학칙 검색 → 게이트 → 치토 답변, 스트리밍)
#      + Qwen3-TTS. 로그는 run_local.log 에도 기록. 종료: Ctrl+C
cd "$(dirname "$0")"
instruct_args=()
[[ -n "$TTS_INSTRUCT" ]] && instruct_args=(--qwen3_tts_instruct "$TTS_INSTRUCT")
exec .venv/bin/speech-to-speech local \
  --mac-optimal-settings \
  --stt mlx-audio-whisper \
  --language auto \
  --llm_backend chat-completions \
  --responses_api_base_url http://127.0.0.1:8100/v1 \
  --responses_api_api_key dummy \
  --model_name ajou-vis \
  --qwen3_tts_speaker Sohee \
  --qwen3_tts_coalesce_inputs False \
  "${instruct_args[@]}" \
  --stream_batch_sentences 1 \
  --local_audio_block_mic_during_playback \
  --port 8766 \
  "$@" 2>&1 | tee -a run_local.log
