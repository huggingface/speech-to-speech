FROM node:24.10.0-bookworm-slim AS browser
WORKDIR /ui
COPY demo/package*.json ./
RUN npm ci --omit=dev --registry=https://registry.npmjs.org

FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NLTK_DATA=/opt/nltk_data HF_HOME=/tmp/huggingface \
    PYTHON_BIN=/usr/local/bin/python UI_HOST=0.0.0.0 UI_PORT=7860 \
    FORWARDED_ALLOW_IPS=* HF_S2S_STT=openai HF_S2S_TTS=openai \
    CONVFILL_CONFIG=configs/demo_mode/convfill_hosted_config.json
RUN apt-get update && apt-get install -y --no-install-recommends git curl libsndfile1 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY examples/convfill/deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r deploy/requirements.txt
RUN python -m nltk.downloader -d /opt/nltk_data punkt_tab averaged_perceptron_tagger_eng
COPY . /app
COPY --from=browser /ui/node_modules /app/demo/node_modules
RUN pip install --no-cache-dir --no-deps . && useradd -m -u 1000 demo && chown -R demo:demo /app /opt/nltk_data
USER demo
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD curl --fail http://127.0.0.1:7860/readyz || exit 1
CMD ["bash", "examples/convfill/run.sh"]
