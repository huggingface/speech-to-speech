# Viva quiz coach: FastAPI server + static browser app (three.js, TalkingHead, Gaussian splats).
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
# Avatars, pinned and checksummed: Max is TalkingHead's AvatarSDK sample (free
# for non-commercial use); Ada is "Nyx", a Gaussian-splat head (MIT) from
# @myned-ai/avatar-chat-widget; Reachy is Pollen Robotics' Reachy Mini model
# (Apache-2.0, from pollen-robotics/reachy_mini_blender), with emotions from
# its Apache-2.0 emotions library baked by scripts/bake_reachy_moves.py.
ADD --checksum=sha256:193c07c351f1f94b12b499ece3f8813432dbe902afe0b5dd6957e788a9a36605 \
    https://raw.githubusercontent.com/met4citizen/TalkingHead/1563de184c19b310de4b636664d5835e2c74dbba/avatars/avatarsdk.glb \
    static/avatars/avatarsdk.glb
ADD --checksum=sha256:8a72dbf145c543a499ed490d165ffc3f7a21db5b714f3572f2217a2f26f43a0b \
    https://cdn.jsdelivr.net/npm/@myned-ai/avatar-chat-widget@1.15.0/public/asset/nyx.zip \
    static/avatars/nyx.zip
ADD --checksum=sha256:ba1619044ba34665fc65385f2eb2910f09a93f663fc09ca5fac10797ab2a485b \
    https://raw.githubusercontent.com/andimarafioti/agentic-days-keynote/a663a678fc03b69a6b920d089f4a9bdfef10d3db/assets/reachy_mini_viz.glb \
    static/avatars/reachy_mini.glb
RUN python scripts/bake_reachy_moves.py
RUN useradd -m -u 1000 user && mkdir -p data && chown -R user data && chmod a+r static/avatars/*
USER user
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
