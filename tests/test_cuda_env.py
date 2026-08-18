"""Unit tests for CUDA/cuDNN environment compatibility helpers."""


from speech_to_speech.utils.cuda_env import (
    clean_ld_library_path,
    has_conflicting_cuda_paths,
    is_sanitization_enabled,
    sanitize_cuda_environment,
)


def test_has_conflicting_cuda_paths():
    assert has_conflicting_cuda_paths("/usr/local/cuda/lib64") is True
    assert has_conflicting_cuda_paths("/opt/cuda/lib64:/usr/lib") is True
    assert has_conflicting_cuda_paths("/home/user/cuda/lib64") is True
    assert has_conflicting_cuda_paths("/usr/local/lib/libcudnn.so") is True
    assert has_conflicting_cuda_paths("/usr/lib/x86_64-linux-gnu") is False
    assert has_conflicting_cuda_paths("/usr/local/lib:/usr/lib") is False
    assert has_conflicting_cuda_paths("") is False


def test_clean_ld_library_path():
    dirty = "/usr/local/cuda/lib64:/usr/local/lib:/opt/cuda/lib:/usr/lib"
    cleaned = clean_ld_library_path(dirty)
    assert cleaned == "/usr/local/lib:/usr/lib"


def test_clean_ld_library_path_all_cuda():
    dirty = "/usr/local/cuda/lib64:/opt/cuda/lib"
    cleaned = clean_ld_library_path(dirty)
    assert cleaned == ""


def test_clean_ld_library_path_empty():
    assert clean_ld_library_path("") == ""


def test_is_sanitization_disabled_under_pytest():
    # During pytest execution, sanitization must be disabled to avoid re-exec
    assert is_sanitization_enabled() is False


def test_is_sanitization_disabled_by_env_flag(monkeypatch):
    monkeypatch.setenv("S2S_SANITIZE_CUDA_ENV", "0")
    assert is_sanitization_enabled() is False


def test_sanitize_cuda_environment_is_noop_in_tests():
    # Calling sanitize_cuda_environment during test suite should safely return False
    result = sanitize_cuda_environment()
    assert result is False
