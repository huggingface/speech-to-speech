# Single container: serves the static front-end AND the /api/search proxy.
# HF Spaces (sdk: docker) routes traffic to $PORT, default 7860.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
