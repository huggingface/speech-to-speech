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
# Bootstrap patched pip with the hash published in the Python security advisory.
# Install alongside the distribution-managed pip without trying to uninstall it.
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed \
        "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" \
    && python3 -m pip install --no-cache-dir --break-system-packages "uv>=0.13.4"

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
RUN uv sync --python /usr/bin/python3 --no-install-project --no-dev

COPY . .
RUN uv sync --python /usr/bin/python3 --no-dev
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
