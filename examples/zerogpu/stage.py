"""Create an explicit deployment snapshot without uploading local secrets."""

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def stage(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}; choose a new staging directory.")
    destination.mkdir(parents=True)
    for source in HERE.iterdir():
        if source.suffix == ".py" or source.name in {"requirements.txt", "README.md", "packages.txt"}:
            shutil.copy2(source, destination / source.name)
    shutil.copytree(
        ROOT / "src/speech_to_speech",
        destination / "speech_to_speech",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    frontend = destination / "frontend"
    frontend.mkdir()
    for name in ["index.html", "style.css", "main.js", "s2s-realtime-client.js", "turn-latency.js"]:
        shutil.copy2(ROOT / "demo" / name, frontend / name)
    # Deployment-specific model credits accompany the readiness gate.
    index = frontend / "index.html"
    html = index.read_text()
    for old, new in [
        ("nvidia/parakeet-tdt-1.1b", "nvidia/parakeet-tdt-0.6b-v3"),
        ("google/gemma-4-31B-it", "Qwen/Qwen3.5-4B"),
        ("https://huggingface.co/inference-endpoints", "https://huggingface.co/docs/hub/spaces-zerogpu"),
        ("Inference Endpoints", "ZeroGPU"),
        ('<span class="pipe-tag">VLM</span>', '<span class="pipe-tag">LLM</span>'),
    ]:
        html = html.replace(old, new)
    html = re.sub(r'<a class="cerebras-credit".*?</a>', "", html, flags=re.DOTALL)
    html = re.sub(r'<span class="pipe-note">· via\s*</span>', "", html)
    html = html.replace('<span class="sep" aria-hidden="true">·</span>', "")
    index.write_text(html)
    for name in ["ui", "ws", "worklets"]:
        shutil.copytree(ROOT / "demo" / name, frontend / name)
    vendor = frontend / "vendor"
    local_vad = vendor / "local-vad"
    local_vad.mkdir(parents=True)
    node_modules = ROOT / "demo/node_modules"
    shutil.copy2(node_modules / "@openai/agents-realtime/dist/bundle/openai-realtime-agents.umd.js", vendor)
    for package, names in [
        (
            "@ricky0123/vad-web",
            ["bundle.min.js", "bundle.min.js.LICENSE.txt", "vad.worklet.bundle.min.js", "silero_vad_v6.onnx"],
        ),
        ("onnxruntime-web", ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]),
    ]:
        for name in names:
            shutil.copy2(node_modules / package / "dist" / name, local_vad / name)
    print(destination)


if __name__ == "__main__":
    stage(sys.argv[1])
