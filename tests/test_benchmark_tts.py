from argparse import Namespace

import pytest

from benchmark_tts import (
    build_benchmark_targets,
    normalize_qwen3_mlx_quantizations,
)


def test_normalize_qwen3_mlx_quantizations_dedupes_and_maps_default():
    assert normalize_qwen3_mlx_quantizations(["default", "6bit", "6bit", "8bit"]) == [
        "bf16",
        "6bit",
        "8bit",
    ]


def test_normalize_qwen3_mlx_quantizations_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unsupported qwen3 MLX quantization"):
        normalize_qwen3_mlx_quantizations(["5bit"])


def test_build_benchmark_targets_expands_qwen3_quantizations():
    args = Namespace(
        handlers=["qwen3", "kokoro"],
        qwen3_mlx_quantizations=["bf16", "4bit", "8bit"],
    )

    assert build_benchmark_targets(args) == [
        ("qwen3[bf16]", "qwen3", {"mlx_quantization": "bf16"}),
        ("qwen3[4bit]", "qwen3", {"mlx_quantization": "4bit"}),
        ("qwen3[8bit]", "qwen3", {"mlx_quantization": "8bit"}),
        ("kokoro", "kokoro", {}),
    ]


def test_build_benchmark_targets_leaves_qwen3_unexpanded_without_quantizations():
    args = Namespace(
        handlers=["qwen3"],
        qwen3_mlx_quantizations=None,
    )

    assert build_benchmark_targets(args) == [("qwen3", "qwen3", {})]
