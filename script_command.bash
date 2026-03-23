.venv/bin/python s2s_pipeline.py \
  --mode realtime \
  --stt parakeet-tdt \
  --llm transformers \
  --tts kokoro \
  --lm_model_name "Qwen/Qwen3.5-4B" \
  --lm_device mps \
  --lm_torch_dtype float16 \
  --enable_live_transcription

# .venv/bin/python s2s_pipeline.py \
#   --mode realtime \
#   --stt parakeet-tdt \
#   --llm open_api \
#   --tts kokoro \
#   --open_api_model_name "openai/gpt-oss-20b:groq" \
#   --open_api_base_url "https://router.huggingface.co/v1" \
#   --open_api_api_key "$HF_TOKEN" \
#   --open_api_stream \
#   --enable_live_transcription