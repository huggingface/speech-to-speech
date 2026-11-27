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
# @myned-ai/avatar-chat-widget; the robots are Pollen Robotics' Reachy Mini,
# Reachy 2 and Microduck (Apache-2.0, from reachy_mini_blender, reachy2-blender
# and microduck, as exported for the agentic-days-keynote deck), with emotions
# from Reachy Mini's Apache-2.0 emotions library baked by
# scripts/bake_reachy_moves.py.
ADD --checksum=sha256:193c07c351f1f94b12b499ece3f8813432dbe902afe0b5dd6957e788a9a36605 \
    https://raw.githubusercontent.com/met4citizen/TalkingHead/1563de184c19b310de4b636664d5835e2c74dbba/avatars/avatarsdk.glb \
    static/avatars/avatarsdk.glb
ADD --checksum=sha256:8a72dbf145c543a499ed490d165ffc3f7a21db5b714f3572f2217a2f26f43a0b \
    https://cdn.jsdelivr.net/npm/@myned-ai/avatar-chat-widget@1.15.0/public/asset/nyx.zip \
    static/avatars/nyx.zip
ADD --checksum=sha256:ba1619044ba34665fc65385f2eb2910f09a93f663fc09ca5fac10797ab2a485b \
    https://raw.githubusercontent.com/andimarafioti/agentic-days-keynote/a663a678fc03b69a6b920d089f4a9bdfef10d3db/assets/reachy_mini_viz.glb \
    static/avatars/reachy_mini.glb
ADD --checksum=sha256:98fc7a4c957b409aea8bb85f03ecb1574d02036628b2e317bad705bf94c34c91 \
    https://raw.githubusercontent.com/andimarafioti/agentic-days-keynote/a663a678fc03b69a6b920d089f4a9bdfef10d3db/assets/reachy2.glb \
    static/avatars/reachy2.glb
ADD --checksum=sha256:80fc4424ed71e430ec4919110dfdb5c714dc1c4e1138e1832a16865cf31eb8c8 \
    https://raw.githubusercontent.com/andimarafioti/agentic-days-keynote/a663a678fc03b69a6b920d089f4a9bdfef10d3db/assets/robots/duck/robot.xml \
    static/avatars/robots/duck/robot.xml
ADD --checksum=sha256:8c335491cb3bc9502a29a763d4e85669343ab5bd04e9213caee1180ded9f2367 \
    https://raw.githubusercontent.com/andimarafioti/agentic-days-keynote/a663a678fc03b69a6b920d089f4a9bdfef10d3db/assets/robots/duck/meshes.glb \
    static/avatars/robots/duck/meshes.glb
ADD --checksum=sha256:57de7bfcb0d2edde88100693d263795eb01d6e0c77d3e0aad8b5fb9ae8ebe0d9 \
    https://raw.githubusercontent.com/andimarafioti/agentic-days-keynote/a663a678fc03b69a6b920d089f4a9bdfef10d3db/assets/robots/duck/clip.json \
    static/avatars/robots/duck/clip.json
RUN python scripts/bake_reachy_moves.py
RUN useradd -m -u 1000 user && mkdir -p data && chown -R user data && chmod -R a+rX static/avatars
USER user
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
