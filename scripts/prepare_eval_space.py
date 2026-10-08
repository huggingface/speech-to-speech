"""Upload committed evaluation source to a private Docker Space for image builds.

    python scripts/prepare_eval_space.py --space-name s2s-big-bench-audio-dev

The Space builds and hosts the container image for `hf jobs run`; it does not
run evaluations or need a long-running app. A storage bucket can hold files but
does not provide the Docker build and image-registry workflow.

Commit source changes before uploading. Run the built image using `hf jobs run`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination: Path) -> str:
    paths = ["src", "docker", "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Commit evaluation source changes before uploading the image.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=ROOT).decode().split("\0")
    for name in filter(None, files):
        target = destination / ("Dockerfile" if name == "docker/Dockerfile.eval" else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (destination / "source-revision.txt").write_text(revision + "\n")
    readme = destination / "README.md"
    readme.write_text(
        "---\ntitle: S2S Big Bench Audio Jobs image\nsdk: docker\nlicense: apache-2.0\n---\n\n"
        "Private development image for Hugging Face Jobs. The Space builds the image; "
        "run `vibe-check` in a GPU Job to evaluate the engine.\n\n"
        "The source revision is recorded in `source-revision.txt`.\n\n" + readme.read_text()
    )
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space-name", default="s2s-big-bench-audio-dev", help="Unqualified repository name in your personal profile."
    )
    args = parser.parse_args()
    if "/" in args.space_name:
        parser.error("Use a Space name without a namespace; the authenticated personal profile is used.")
    api = HfApi()
    owner = api.whoami()["name"]
    repo = f"{owner}/{args.space_name}"
    with tempfile.TemporaryDirectory(prefix="s2s-eval-space-") as folder:
        revision = prepare(Path(folder))
        api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
        if not api.repo_info(repo, repo_type="space").private:
            raise RuntimeError(f"Refusing to upload development source to public Space {repo}")
        api.upload_folder(
            repo_id=repo,
            repo_type="space",
            folder_path=folder,
            # Synchronize only paths managed by this uploader. Files uploaded in
            # this commit are automatically excluded from the deletion list.
            delete_patterns=[
                "src/*",
                "docker/*",
                "Dockerfile",
                "pyproject.toml",
                "README.md",
                "LICENSE",
                "MANIFEST.in",
                "source-revision.txt",
            ],
            commit_message=f"Build S2S evaluation {revision[:12]}",
        )
    print(f"https://huggingface.co/spaces/{repo}")
    print(
        f"hf jobs run --namespace {owner} --flavor l4x1 --timeout 45m --secrets HF_TOKEN "
        f"-e S2S_LIMIT=4 -e S2S_PUSH_TO_HUB={owner}/s2s-big-bench-audio-results "
        f"hf.co/spaces/{repo} vibe-check"
    )


if __name__ == "__main__":
    main()
