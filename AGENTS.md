# Repository Instructions

- Never include `codex` in branch names or pull request titles.
- Keep release pull requests focused on version metadata and release documentation.
- Do not commit local build artifacts such as `dist/`, `build/`, or generated wheel/sdist files.

## Publishing to PyPI

PyPI publishing is handled by GitHub Actions in `.github/workflows/publish.yml`. The workflow runs on pushed tags that match `v*`, builds the package with `uv build`, checks the artifacts with `twine check --strict`, and publishes through the configured `pypi` environment.

To prepare a release:

1. Confirm the intended version is not already published on PyPI.
2. Bump `version` in `pyproject.toml`.
3. Bump `__version__` in `src/speech_to_speech/__init__.py`.
4. Open and merge a pull request with only the release preparation changes.

To publish after the release PR is merged:

1. Update `main` locally: `git checkout main && git pull origin main`.
2. Create an annotated tag for the version: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
3. Push the tag: `git push origin vX.Y.Z`.
4. Watch the `Publish` GitHub Actions workflow complete successfully.
5. Verify the new version appears at `https://pypi.org/project/speech-to-speech/`.

Only upload manually if the GitHub Actions workflow is unavailable and the maintainers have explicitly chosen that fallback.
