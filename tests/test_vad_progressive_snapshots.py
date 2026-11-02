import torch

from tests.test_speculative_turns import _StaticVADIterator, _vad_handler_for_iterator

SR = 16000


def test_vad_emits_internal_snapshots_without_closing_turn_or_enabling_live_captions():
    iterator = _StaticVADIterator(
        triggered=True, vad_output=None, buffer_chunks=[torch.ones(SR // 2)], active_speech_samples=2 * SR
    )
    vad = _vad_handler_for_iterator(iterator)
    vad.transcription_window_samples = SR
    vad._last_window_snapshot_samples = vad._window_check_after_samples = 0
    vad.speculative_turns.start_turn()
    recounts = []
    count_buffer = vad._speech_buffer_duration_ms

    def counted_buffer_duration_ms():
        recounts.append(None)
        return count_buffer()

    vad._speech_buffer_duration_ms = counted_buffer_duration_ms
    assert list(vad._process_realtime(None)) == []
    # A window needs another half second of received audio; do not recount the buffer per chunk.
    for _ in range(3):
        vad._total_samples += SR // 8
        assert list(vad._process_realtime(None)) == []
    assert len(recounts) == 1
    vad._total_samples += SR // 8
    iterator.buffer = iterator._speech_chunks = [torch.ones(SR)]
    (snapshot,) = list(vad._process_realtime(None))
    assert len(recounts) == 2
    assert snapshot.mode == "progressive"
    assert not vad.enable_realtime_transcription
    assert vad.text_output_queue.empty()
    assert list(vad._process_realtime(None)) == []
    # A reopening keeps the cumulative cursor; a new logical turn resets it.
    vad.speculative_turns.segment_finalized(1000)
    vad._speech_started_emitted = False
    vad._ensure_turn_for_speech_start(9000)
    assert vad._last_window_snapshot_samples == vad._window_check_after_samples == 0
