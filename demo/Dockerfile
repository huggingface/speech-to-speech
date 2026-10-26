FROM node:24-slim AS agents-sdk

WORKDIR /sdk
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Single runtime container: serves the static front-end AND API proxies.
# HF Spaces (sdk: docker) routes traffic to $PORT, default 7860.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=agents-sdk /sdk/node_modules/@openai/agents-realtime/dist/bundle/openai-realtime-agents.umd.js /app/vendor/openai-realtime-agents.umd.js
COPY --from=agents-sdk /sdk/node_modules/@ricky0123/vad-web/dist/bundle.min.js /sdk/node_modules/@ricky0123/vad-web/dist/bundle.min.js.LICENSE.txt /sdk/node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js /sdk/node_modules/@ricky0123/vad-web/dist/silero_vad_v6.onnx /app/vendor/local-vad/
COPY --from=agents-sdk /sdk/node_modules/onnxruntime-web/dist/ort.wasm.min.js /sdk/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs /sdk/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm /app/vendor/local-vad/

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
