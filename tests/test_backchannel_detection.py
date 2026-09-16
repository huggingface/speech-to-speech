from speech_to_speech.LLM.utils import is_backchannel_turn


def test_english_backchannels() -> None:
    for text in [
        "mm-hmm",
        "uh-huh",
        "yeah",
        "yep",
        "alright",
        "go ahead",
        "ok",
        "okay",
        "I see",
        "sure",
        "Right.",
        "Great.",
    ]:
        assert is_backchannel_turn(text, duration_s=0.5), f"Expected {text} to be a backchannel"


def test_multilingual_backchannels() -> None:
    # Spanish
    assert is_backchannel_turn("ajá", duration_s=0.4)
    assert is_backchannel_turn("claro", duration_s=0.4)
    assert is_backchannel_turn("vale", duration_s=0.3)

    # French
    assert is_backchannel_turn("ouais", duration_s=0.4)
    assert is_backchannel_turn("d'accord", duration_s=0.5)

    # German
    assert is_backchannel_turn("genau", duration_s=0.4)
    assert is_backchannel_turn("ja", duration_s=0.3)

    # Chinese & Japanese
    assert is_backchannel_turn("嗯", duration_s=0.3)
    assert is_backchannel_turn("好的", duration_s=0.4)
    assert is_backchannel_turn("うん", duration_s=0.3)
    assert is_backchannel_turn("はい", duration_s=0.3)


def test_questions_are_not_backchannels() -> None:
    assert not is_backchannel_turn("What?", duration_s=0.4)
    assert not is_backchannel_turn("Really?", duration_s=0.4)
    assert not is_backchannel_turn("Why?", duration_s=0.3)
    assert not is_backchannel_turn("什么？", duration_s=0.3)
    assert not is_backchannel_turn("¿Cómo?", duration_s=0.4)


def test_commands_and_exclamations_are_not_backchannels() -> None:
    for word in ["Stop!", "Wait!", "No!", "Halt!", "Quiet!", "Cancel!", "Arrête!", "¡Para!", "रुको!", "توقف!"]:
        assert not is_backchannel_turn(word, duration_s=0.4)


def test_long_duration_is_not_backchannel() -> None:
    assert not is_backchannel_turn("mm-hmm", duration_s=2.5)


def test_multi_word_utterances_are_not_backchannels() -> None:
    assert not is_backchannel_turn("I disagree with that point", duration_s=0.8)
    assert not is_backchannel_turn("Stop talking right now", duration_s=0.7)
    assert not is_backchannel_turn("Can you repeat that", duration_s=0.9)
    assert not is_backchannel_turn("Can I build a boat", duration_s=1.2)
    assert not is_backchannel_turn("Tell me all about boats again", duration_s=1.5)
