# Certificate security updates

The [Python security advisory](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)
requires `certifi>=2026.10.19`, `pip>=26.2.2`, and `uv>=0.13.4` to remove
trust in the affected emSign CA. The package and both demo requirements enforce
the certifi minimum; repository uv commands enforce the uv minimum.

## Existing environments

Repeat these steps in each environment you use. Updating one environment does
not update other environments or the pip bundled with a Python installation.
For Python installations managed by Homebrew or Conda, use that package manager
to update Python and CA certificates as well. Conda needs version 26.9.4 or newer.

Bootstrap pip using the advisory's verified wheel and SHA-256 hash:

```bash
python -m pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
python -m pip install --upgrade "certifi>=2026.10.19"
```

If uv is installed through pip, update it using the patched pip:

```bash
python -m pip install --upgrade "uv>=0.13.4"
```

For standalone uv installations, follow the advisory's secure update guidance.
Using system certificates to update uv requires an already updated system trust
store. Do not disable certificate verification.

Refresh any local lockfile and synchronize the project using patched uv:

```bash
uv lock --upgrade-package certifi
uv sync --group dev
```

Restart running services after updating their environments. New environments may
still inherit old pip from ensurepip; update the underlying Python installation
or repeat the verified pip bootstrap after creating them.

## Container images

Rebuild existing images with fresh base images and Docker layer caching disabled
so CA certificate packages and Python dependencies are downloaded again:

```bash
docker build --pull --no-cache -t speech-to-speech .
docker build --pull --no-cache -f Dockerfile.arm64 -t speech-to-speech-arm64 .
docker build --pull --no-cache -t s2s-demo demo/
```

Apply the same flags to evaluation image builds. Redeploy the rebuilt images;
changes to the Dockerfiles do not update running containers automatically.
