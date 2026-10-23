"""Upload this example to an explicitly selected private Hugging Face Space.

HF_TOKEN authenticates the Hub upload only. Inference keys are supplied through
separate environment variables and stored as server-side Space secrets.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Private destination Space, in owner/name form")
    parser.add_argument("--frontend-only", action="store_true", help="Use an external managed pipeline pool")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.frontend_only:
        urls = {}
        load_balancer = os.environ["LOAD_BALANCER_URL"].strip()
        if not load_balancer.startswith("https://"):
            raise ValueError("Hosted session allocation must use HTTPS")
    else:
        urls = {stage: os.environ["CONVFILL_" + stage + "_URL"] for stage in ["REASONER", "TALKER", "STT", "TTS"]}
        if any(not value.startswith("https://") for value in urls.values()):
            raise ValueError("Hosted inference URLs must use HTTPS")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    if not api.space_info(args.repo).private:
        raise ValueError("Refusing to upload private integration source to a public Space")
    for stage, url in urls.items():
        api.add_space_variable(args.repo, "CONVFILL_" + stage + "_URL", url)
        key = os.environ.get("CONVFILL_" + stage + "_API_KEY")
        if key:
            api.add_space_secret(args.repo, "CONVFILL_" + stage + "_API_KEY", key)
    if args.frontend_only:
        api.add_space_secret(args.repo, "LOAD_BALANCER_URL", load_balancer)
        if os.environ.get("LB_HF_TOKEN"):
            api.add_space_secret(args.repo, "LB_HF_TOKEN", os.environ["LB_HF_TOKEN"])
    else:
        api.add_space_variable(
            args.repo,
            "CONVFILL_REASONER_MODEL",
            os.environ.get("CONVFILL_REASONER_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4"),
        )
        api.add_space_variable(
            args.repo, "CONVFILL_STT_MODEL", os.environ.get("CONVFILL_STT_MODEL", "Qwen/Qwen3-ASR-1.7B")
        )
        api.add_space_variable(args.repo, "CONVFILL_LOG_CONTENT", "0")
    for name in ("CONVFILL_SEARCH_URL", "STARTUP_GREETING"):
        if name in os.environ:
            api.add_space_variable(args.repo, name, os.environ[name])
    for name in ("SERPER_API_KEY", "CONVFILL_SEARCH_API_KEY"):
        if os.environ.get(name):
            api.add_space_secret(args.repo, name, os.environ[name])
    with tempfile.TemporaryDirectory(prefix="convfill-space-") as temporary:
        target = Path(temporary)
        shutil.copy2(
            root / ("examples/convfill/Dockerfile.frontend" if args.frontend_only else "examples/convfill/Dockerfile"),
            target / "Dockerfile",
        )
        for name in ["pyproject.toml", "LICENSE"]:
            shutil.copy2(root / name, target / name)
        for name in ["demo"] if args.frontend_only else ["src", "demo", "examples/convfill"]:
            shutil.copytree(
                root / name,
                target / name,
                ignore=shutil.ignore_patterns("__pycache__", "node_modules", ".venv", "*.pyc", ".env", "*.log"),
            )
        oauth = "hf_oauth: true\n" if args.frontend_only else ""
        (target / "README.md").write_text(
            "---\ntitle: ConvFill Gemma Voice Demo\nemoji: 🎙️\ncolorFrom: blue\ncolorTo: green\nsdk: docker\napp_port: 7860\n"
            + oauth
            + "pinned: false\n---\n\nPrivate ConvFill voice demo. See operator deployment settings for the pipeline placement.\n"
        )
        commit = api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=target,
            commit_message="Deploy hosted Gemma and ConvFill speech demo",
        )
    print("Space:", args.repo, "commit:", commit.oid)


if __name__ == "__main__":
    main()
