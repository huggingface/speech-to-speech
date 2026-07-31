import pytest

from speech_to_speech.STT.qwen3_asr_handler import _language_to_code, _version_tuple


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.13.0", (5, 13, 0)),
        ("5.14.0.dev0", (5, 14, 0)),
        ("5.13.0rc1", (5, 13, 0)),
    ],
)
def test_version_tuple_handles_transformers_versions(raw, expected):
    assert _version_tuple(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("English", "en"),
        ("Chinese", "zh"),
        ("Filipino", "fil"),
        ("Cantonese", "yue"),
        ("en", "en"),
        (None, None),
    ],
)
def test_language_to_code_normalizes_qwen3_asr_language_names(raw, expected):
    assert _language_to_code(raw) == expected
