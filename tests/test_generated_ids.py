import re

import pytest

from speech_to_speech.utils.utils import _generate_id


def test_generated_call_ids_fit_livekit_limit():
    call_ids = [_generate_id("call") for _ in range(2)]
    assert all(re.fullmatch(r"call_[0-9a-f]{27}", call_id) for call_id in call_ids)
    assert call_ids[0] != call_ids[1]


@pytest.mark.parametrize("prefix", ["event", "session", "conv", "resp", "item", "msg", "fc"])
def test_other_generated_id_formats_are_unchanged(prefix):
    assert re.fullmatch(rf"{prefix}_[0-9a-f]{{32}}", _generate_id(prefix))
