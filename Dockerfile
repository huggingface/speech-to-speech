FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/src/app/.venv/bin:${PATH}"

WORKDIR /usr/src/app

# Install packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libportaudio2 \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages uv

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
RUN uv sync --python /usr/bin/python3 --no-install-project --no-dev

COPY . .
RUN uv sync --python /usr/bin/python3 --no-dev
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
