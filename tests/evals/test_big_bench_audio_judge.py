"""An explicit judge verdict overrides extraction; an unavailable judge does not."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import openai
import pytest

from speech_to_speech.evals.big_bench_audio.judge import JudgeConfig, judge_results
from speech_to_speech.evals.big_bench_audio.report import ItemResult, aggregate


@pytest.mark.parametrize(
    "reply, completed, extracted, correct",
    [
        ("UNPARSED", True, None, None),
        ("No", True, "no", False),
        ("Yes", True, "yes", True),
        ("", False, "yes", True),
        (TimeoutError("Judge unavailable"), False, "yes", True),
    ],
)
async def test_judge_verdict_and_outage_survive_report_round_trip(monkeypatch, reply, completed, extracted, correct):
    result = ItemResult(
        id=1,
        category="navigate",
        official_answer="Yes",
        reply="It might be Yes, but I cannot determine an answer.",
        extracted="yes",
        correct=True,
    )
    create = AsyncMock()
    if isinstance(reply, Exception):
        create.side_effect = reply
    else:
        create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    await judge_results([result], JudgeConfig(model="test-extractor"))

    create.assert_awaited_once()
    for item in (result, ItemResult.from_dict(result.to_dict())):
        assert item.judge_completed is completed
        assert item.final_extracted == extracted
        assert item.final_correct is correct
        assert item.extracted == "yes" and item.correct is True
        totals = aggregate([item])["totals"]
        assert totals["parsed"] == int(extracted is not None)
        assert totals["correct"] == int(correct is True)
