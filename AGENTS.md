# Repository Instructions

- Never include `codex` in branch names or pull request titles.
- Never merge a pull request into `main` unless the user explicitly asks the agent to perform the merge and then confirms when asked immediately before the merge. Requests to review, fix, prepare, finish, or get a pull request ready do not count as merge authorization. Once the pull request is ready, report its status and ask for confirmation; merge only after a clear follow-up confirmation.
- For existing/open pull requests, do not amend, squash, rebase-rewrite, or force-push follow-up changes. Make new commits and push normally unless explicitly asked to rewrite history.
- Keep release pull requests focused on version metadata and release documentation.
- Do not commit local build artifacts such as `dist/`, `build/`, or generated wheel/sdist files.

## Evals

- Changes to conversation history, prompts, tool handling, chat templates or the LLM backend: run the tool-use conversation eval (`python -m speech_to_speech.evals.tool_conversations`) against `main` and against the change, then `compare` the reports. Setup, text-only runs and how to read results are in `docs/vibe-checks.md#tool-use-conversations`.
- Changes to the audio path (VAD, STT, TTS, latency): Big Bench Audio, in the same document.

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
