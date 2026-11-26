# Viva quiz coach: FastAPI server + static browser app (three.js, TalkingHead).
FROM node:22-slim AS assets
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=assets /app/node_modules ./node_modules
COPY . .
# Avaturn sample avatar from the TalkingHead repo (free for non-commercial use).
ADD --checksum=sha256:f56e7a34e249597cff08ba7ef7ce597b61b66b455a21023f186e769caca0cdda \
    https://raw.githubusercontent.com/met4citizen/TalkingHead/1563de184c19b310de4b636664d5835e2c74dbba/avatars/avaturn.glb \
    static/avatars/avaturn.glb
RUN useradd -m -u 1000 user && mkdir -p data && chown -R user data && chmod a+r static/avatars/avaturn.glb
USER user
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
