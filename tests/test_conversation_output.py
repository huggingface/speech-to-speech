import logging

import pytest

from speech_to_speech.pipeline.console import console, conversation_output


def test_conversation_output_hides_backend_text_and_routine_logs_and_restores_after_failure(capsys, caplog):
    previous_quiet = console.quiet
    previous_disable = logging.root.manager.disable
    backend_logger = logging.getLogger("speech_to_speech.test_backend")
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            with conversation_output(True):
                console.print("USER: duplicate backend transcript")
                backend_logger.info("routine timing")
                backend_logger.warning("microphone warning")
                backend_logger.error("backend error")
                print("USER: conversation transcript")
                raise RuntimeError("startup failed")

    assert capsys.readouterr().out == "USER: conversation transcript\n"
    assert "routine timing" not in caplog.text
    assert "microphone warning" in caplog.text
    assert "backend error" in caplog.text
    assert console.quiet == previous_quiet
    assert logging.root.manager.disable == previous_disable


def test_nested_output_context_keeps_existing_logging_policy():
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.ERROR)
        with conversation_output(True):
            with conversation_output(False):
                assert console.quiet
                assert logging.root.manager.disable == logging.ERROR
            with conversation_output(True):
                assert console.quiet
            assert console.quiet
        assert logging.root.manager.disable == logging.ERROR
    finally:
        logging.disable(previous_disable)


@pytest.mark.parametrize("flags,quiet", [([], True), (["--verbose"], False), (["--local_audio_print_json"], False)])
def test_local_display_policy_is_applied_before_pipeline_startup(monkeypatch, caplog, flags, quiet):
    from speech_to_speech import s2s_pipeline

    observed = []

    def check_startup(command, args):
        observed.append(command)
        assert console.quiet is quiet
        assert logging.getLogger("speech_to_speech.test_backend").isEnabledFor(logging.INFO) is not quiet

    monkeypatch.setattr(s2s_pipeline, "_run_parsed_pipeline_command", check_startup)
    with caplog.at_level(logging.INFO):
        s2s_pipeline.run_pipeline_command("local", flags)
    assert observed == ["local"]
