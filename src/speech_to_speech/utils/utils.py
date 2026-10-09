from __future__ import annotations

import importlib.util
import logging
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams


def response_wants_audio(response: RealtimeResponseCreateParams | None) -> bool:
    """Whether a response should produce audio (and audio events) vs. text only.

    Mirrors the OpenAI realtime semantics for ``output_modalities``: an absent
    value (``None``) or an empty list, or an explicit ``"audio"`` entry means
    audio; a non-empty list without ``"audio"`` (e.g. ``["text"]``) means text
    only.
    """
    if response is None:
        return True
    mods = response.output_modalities
    return not mods or "audio" in mods


def is_out_of_band(response: RealtimeResponseCreateParams | None) -> bool:
    """Whether a response is *out-of-band* (``conversation="none"``).

    Out-of-band responses are generated against a temporary context and are
    never threaded into the default conversation: their ``input`` (if any)
    seeds a throwaway chat, their assistant output is not committed back, and
    their ``conversation_id`` is reported as ``null``. Any other ``conversation``
    value (``"auto"``, ``None``, or an arbitrary id) is treated as in-band.
    """
    return response is not None and response.conversation == "none"


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def is_npu_available() -> bool:
    """Whether an Ascend NPU is available through a ``torch_npu`` build.

    ``torch.npu`` only exists once ``torch_npu`` has been imported, so the import
    is attempted lazily here and stays invisible to CUDA/MPS/CPU-only installs.
    """
    import torch

    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
    npu = getattr(torch, "npu", None)
    return npu is not None and bool(npu.is_available())


# Device types a plain PyTorch model can run on, in ``auto`` preference order.
TORCH_DEVICES = ("cuda", "npu", "xpu", "mps", "cpu")


logger = logging.getLogger(__name__)

# ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` namespace, so
# ``rocm`` and ``hip`` are accepted as spellings of ``cuda`` on the command line.
_DEVICE_ALIASES = {"rocm": "cuda", "hip": "cuda"}


def is_rocm() -> bool:
    """Whether the installed PyTorch is a ROCm (HIP) build."""
    import torch

    return bool(getattr(torch.version, "hip", None))


def normalize_device(device: str) -> str:
    """Map ``rocm``/``hip`` (with an optional ``:index``) to the ``cuda`` device PyTorch uses on ROCm."""
    device_type, sep, index = device.partition(":")
    alias = _DEVICE_ALIASES.get(device_type.lower())
    return f"{alias}{sep}{index}" if alias else device


def resolve_attn_implementation(requested: str, device: str) -> str:
    """Fall back from ``flash_attention_2`` to ``sdpa`` when the flash-attn package is not installed.

    flash-attn ships CUDA wheels only; ROCm needs its own build, and CPU never has it.
    PyTorch SDPA dispatches to its own fused kernels on CUDA and ROCm, so it is the safe default.
    """
    if requested != "flash_attention_2":
        return requested
    if device.split(":", 1)[0] == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        return requested
    logger.warning("flash_attention_2 is unavailable on %s%s; using sdpa.", device, " (ROCm)" if is_rocm() else "")
    return "sdpa"


def _is_device_available(device_type: str) -> bool:
    # torch is imported here rather than at module level: this module is also
    # used by lightweight paths (CLI, Realtime API, LLM helpers) that never
    # touch a device, and importing torch costs seconds.
    import torch

    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "npu":
        return is_npu_available()
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "mps":
        return torch.backends.mps.is_available()
    return device_type == "cpu"


def validate_device(device: str, supported: Sequence[str], component: str) -> None:
    """Raise unless ``device`` is ``auto`` or one of the ``supported`` device types (``cuda:1`` counts as ``cuda``)."""
    device = normalize_device(device)
    if device != "auto" and device.split(":", 1)[0] not in supported:
        raise ValueError(f"{component} supports device 'auto' or one of: {', '.join(supported)}; got {device!r}.")


def resolve_device(device: str, supported: Sequence[str], component: str) -> str:
    """Turn ``auto`` into the first available of ``supported``; keep a supported explicit choice as is."""
    device = normalize_device(device)
    validate_device(device, supported, component)
    if device != "auto":
        return device
    for device_type in supported:
        if _is_device_available(device_type):
            return device_type
    raise ValueError(f"{component} found none of its supported devices available: {', '.join(supported)}.")


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def int2float(sound: np.ndarray) -> np.ndarray:
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound
