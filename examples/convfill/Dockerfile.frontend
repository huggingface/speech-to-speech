FROM node:24.10.0-bookworm-slim AS browser
WORKDIR /ui
COPY demo/package*.json ./
RUN npm ci --omit=dev --registry=https://registry.npmjs.org --replace-registry-host=always

# UI, search, and session allocation only. CPU pipelines run in external workers.
FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FORWARDED_ALLOW_IPS=*
WORKDIR /app/demo
COPY demo/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY demo /app/demo
COPY --from=browser /ui/node_modules /app/demo/node_modules
RUN useradd -m -u 1000 demo && chown -R demo:demo /app
USER demo
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
