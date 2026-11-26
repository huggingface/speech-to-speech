"""Research helpers that need no network: link parsing, source digests, picking distinct questions.

Run from examples/avatar-quiz-coach:  python -m pytest tests
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coach import research, sources  # noqa: E402


def test_links_in_a_topic_become_targets():
    got = sources.find_targets(
        "quiz me on https://github.com/huggingface/speech-to-speech/tree/main/src/speech_to_speech/TTS, "
        "https://huggingface.co/papers/2305.18290 and https://arxiv.org/pdf/1706.03762v7. Also https://example.com/post"
    )
    assert [t.kind for t in got] == ["github", "arxiv", "arxiv", "web"]
    gh = got[0]
    assert (gh.owner, gh.repo, gh.ref, gh.path) == (
        "huggingface",
        "speech-to-speech",
        "main",
        "src/speech_to_speech/TTS",
    )
    assert gh.url == "https://github.com/huggingface/speech-to-speech"
    assert [got[1].arxiv_id, got[2].arxiv_id] == ["2305.18290", "1706.03762"]
    assert sources.find_targets("the French Revolution") == []


def test_python_outline_keeps_structure_docs_and_reasons():
    code = '''"""Turn-taking for the voice pipeline."""
import os

class SmartTurn(Base):
    """Decides whether the user finished speaking."""

    def __init__(self, threshold: float = 0.5):
        # Below this probability we keep listening instead of answering the user.
        self.threshold = threshold

    def _private(self):
        pass

def decide(audio, *, max_wait_ms: int = 2000) -> bool:
    """Return True when the turn is complete."""
    x = 1  # short
    return True
'''
    out = sources.python_outline(code)
    assert '"""Turn-taking for the voice pipeline."""' in out
    assert "class SmartTurn(Base):" in out
    assert "def __init__(self, threshold: float=0.5)" in out
    assert "# Below this probability we keep listening" in out
    assert "def decide(audio, *, max_wait_ms: int=2000) -> bool" in out
    assert "_private" not in out and "x = 1" not in out


def test_docs_rank_design_docs_over_changelogs():
    ranked = sources.rank_docs(
        [
            ("README.md", 50000),
            ("CHANGELOG.md", 9000),
            ("docs/releases/v1.0.0.md", 4000),
            ("docs/response-latency.md", 19000),
            ("src/pkg/TTS/README.md", 20000),
            ("tests/fixtures/notes.md", 5000),
            ("docs/faq.md", 3000),
        ]
    )
    assert ranked[:2] == ["docs/response-latency.md", "src/pkg/TTS/README.md"]
    assert not {"README.md", "CHANGELOG.md", "docs/releases/v1.0.0.md", "tests/fixtures/notes.md"} & set(ranked)


def test_code_ranking_prefers_central_files_named_in_docs():
    ranked = sources.rank_code(
        [
            ("src/pkg/pipeline.py", 30000),
            ("src/pkg/vad_handler.py", 8000),
            ("src/pkg/utils.py", 9000),
            ("tests/test_pipeline.py", 40000),
            ("examples/demo.py", 30000),
            ("setup.py", 3000),
        ],
        mentions="the vad_handler finds speech boundaries",
        package_dirs={"src/pkg"},
    )
    assert set(ranked[:2]) == {"src/pkg/pipeline.py", "src/pkg/vad_handler.py"}
    assert "tests/test_pipeline.py" not in ranked and "examples/demo.py" not in ranked and "setup.py" not in ranked


def test_references_are_cut_from_papers():
    body = "Abstract\n" + "method " * 400 + "\nReferences\n[1] Someone. 2017.\nAppendix A\nproofs"
    out = sources.cut_references(body)
    assert out.endswith("method ") and "Someone" not in out


def q(concept, question, answer):
    return {"concept": concept, "question": question, "answer": answer}


def test_repeated_questions_are_dropped_in_favour_of_spares():
    questions = [
        q("pipeline stages", "What are the four stages of the pipeline?", "VAD, speech to text, language model, TTS."),
        q("stage threads", "How are the pipeline stages connected at runtime?", "Threads connected by queues."),
        q(
            "pipeline stages",
            "Which stages make up the speech pipeline?",
            "Voice detection, transcription, LLM, synthesis.",
        ),
        q(
            "speech recognition",
            "What does the speech recognition stage of the pipeline do?",
            "It transcribes the user's speech into text.",
        ),
        q(
            "transcription",
            "What does the transcription stage do in the pipeline?",
            "Transcribes the speech into text.",
        ),
        q("interruptions", "What happens to history when the user interrupts?", "It is cut to what the user heard."),
    ]
    picked = research.pick_distinct(questions, 4)
    assert [p["concept"] for p in picked] == ["pipeline stages", "stage threads", "speech recognition", "interruptions"]


def test_writer_prompt_asks_for_spares_and_uses_the_material_guide():
    srcs = [{"title": "repo README", "url": "https://github.com/a/b", "text": "x" * 50000}]
    prompt = research._writer_prompt("a/b", "pro", 8, srcs, kind="codebase")
    assert "Write exactly 11 questions (the best 8 will be used)" in prompt
    assert "software project" in prompt and "x" * 50000 in prompt
    general = research._writer_prompt("black holes", "pro", 8, srcs)
    assert "software project" not in general and "x" * 26001 not in general


def test_cache_key_changes_with_research_version():
    assert research.pack_id_for("Black  Holes", "pro", 8) == research.pack_id_for("black holes", "pro", 8)
    current = research.pack_id_for("black holes", "pro", 8)
    old = research.RESEARCH_VERSION
    try:
        research.RESEARCH_VERSION = old + 1
        assert research.pack_id_for("black holes", "pro", 8) != current
    finally:
        research.RESEARCH_VERSION = old
