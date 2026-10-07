"""Apply model history proposals at the service's accepted-output boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING

from speech_to_speech.LLM.chat import Chat, CompactionResult
from speech_to_speech.pipeline.events import ResponseGenerationDoneEvent
from speech_to_speech.pipeline.history import ResponseHistory

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService
    from speech_to_speech.pipeline.events import PipelineEvent


@dataclass
class _ResponseHistoryState:
    proposal: ResponseHistory | None = None
    accepted: bool = False
    failed: bool = False
    applied_ids: set[str] = field(default_factory=set)
    last_item_id: str | None = None
    cleaned: bool = False


@dataclass(frozen=True)
class _CompactionReady:
    conn_id: str
    chat_id: int
    user_summary: str
    assistant_summary: str
    marker_ids: frozenset[str]
    generation: int


class HistoryCommitError(RuntimeError):
    """The service could not apply a proposed history transaction."""


class HistoryWriter:
    """Service-owned staging, commit, rollback and background-result application."""

    def __init__(self, service: RealtimeService) -> None:
        self._service = service
        self._responses: dict[tuple[str, str], _ResponseHistoryState] = {}
        self._compactions: Queue[_CompactionReady] = Queue()

    def stage(self, conn_id: str, event: PipelineEvent) -> None:
        key = getattr(event, "response_key", None)
        if key is None or key in self._service._state(conn_id).closed_response_keys:
            return
        if isinstance(event, ResponseGenerationDoneEvent) and not event.succeeded:
            self.reject(conn_id, key)
            return
        proposal = getattr(event, "history", None)
        if proposal is None:
            return
        chat = self._service._state(conn_id).runtime_config.chat
        if proposal.chat_id != id(chat):
            return  # Late output from a released session cannot write a new chat.
        state = self._responses.setdefault((conn_id, key), _ResponseHistoryState())
        if state.failed:
            return
        if (
            state.proposal is None
            or proposal.version > state.proposal.version
            or (proposal.version == state.proposal.version and proposal.complete)
        ):
            state.proposal = proposal
        prefetch = self._service._state(conn_id).tool_followup_prefetch_request
        if prefetch is not None and prefetch.response_key == key and prefetch.prefetch_transaction is not None:
            # The service owns the deferred cleanup; model workers only emit data.
            def claim_history() -> None:
                tracker = self._service.speculative_turns
                if tracker is not None:
                    from speech_to_speech.pipeline.speculative_turns import TurnGateAction

                    if (
                        tracker.gate(prefetch.turn_id, prefetch.turn_revision, commit=True).action
                        is not TurnGateAction.ACCEPT
                    ):
                        raise RuntimeError("Prefetched history no longer belongs to an accepted turn")
                self.accept(conn_id, key)

            prefetch.prefetch_transaction.complete(claim_history)
        if state.accepted:
            self._apply(conn_id, key, state)

    def accept(self, conn_id: str, key: str | None) -> None:
        """Called after the output gate commits this turn, before exposing output."""
        if key is None or key in self._service._state(conn_id).closed_response_keys:
            return
        state = self._responses.setdefault((conn_id, key), _ResponseHistoryState())
        state.accepted = True
        if not state.failed:
            self._apply(conn_id, key, state)

    def _apply(self, conn_id: str, key: str, state: _ResponseHistoryState) -> None:
        try:
            self._apply_proposal(conn_id, key, state)
        except Exception as exc:
            self.reject(conn_id, key)
            raise HistoryCommitError(f"Language model history commit failed: {exc}") from exc

    def _apply_proposal(self, conn_id: str, key: str, state: _ResponseHistoryState) -> None:
        proposal = state.proposal
        if proposal is None:
            return
        chat = self._service._state(conn_id).runtime_config.chat
        if proposal.chat_id != id(chat):
            return
        items = [item for item in proposal.decode_items() if item.id not in state.applied_ids]
        recorded = chat.add_provisional_generation_items(
            key,
            items,
            after_item_id=state.last_item_id or proposal.after_item_id,
            committed_item_ids={proposal.input_item_id} if proposal.complete and proposal.input_item_id else None,
        )
        if recorded is None:
            state.failed = True
            return
        for item in recorded:
            if item.id is not None:
                state.applied_ids.add(item.id)
                state.last_item_id = item.id
        if not proposal.complete or state.cleaned:
            return
        snapshot = chat.snapshot_history_cleanup()
        try:
            if proposal.item_order is not None:
                chat.order_response_items(list(proposal.item_order))
            chat.strip_images(set(proposal.consumed_image_ids))
            if proposal.audio_history_turns is not None:
                chat.compact_audio_history(proposal.audio_history_turns)
            self._route_compaction(conn_id, chat)
            chat.trim_if_needed(proposal.compactor)
        except Exception:
            chat.restore_history_cleanup(snapshot)
            self.reject(conn_id, key)
            raise
        state.cleaned = True

    def reject(self, conn_id: str, key: str) -> None:
        state = self._responses.setdefault((conn_id, key), _ResponseHistoryState())
        state.failed = True
        state.proposal = None
        self._service._state(conn_id).runtime_config.chat.rollback_provisional_generation(key)

    def close(self, conn_id: str, key: str | None) -> None:
        if key is None:
            return
        self._responses.pop((conn_id, key), None)
        # A completed response has already finalized these items. For every
        # other exit, including a never-opened stale response, undo its writes.
        self._service._state(conn_id).runtime_config.chat.rollback_provisional_generation(key)

    def close_session(self, conn_id: str) -> None:
        for owner, key in tuple(self._responses):
            if owner == conn_id:
                self._responses.pop((owner, key), None)

    def _route_compaction(self, conn_id: str, chat: Chat) -> None:
        def publish(result: CompactionResult, marker_ids: set[str], generation: int) -> None:
            self._compactions.put(
                _CompactionReady(
                    conn_id, id(chat), result.user_summary, result.assistant_summary, frozenset(marker_ids), generation
                )
            )

        chat.compaction_result_callback = publish

    def drain_compactions(self) -> None:
        """Apply computed summaries on the same service thread as other writes."""
        while True:
            try:
                result = self._compactions.get_nowait()
            except Empty:
                return
            state = self._service._conns.get(result.conn_id)
            if state is None or id(state.runtime_config.chat) != result.chat_id:
                continue
            state.runtime_config.chat.finish_compaction(
                CompactionResult(user_summary=result.user_summary, assistant_summary=result.assistant_summary),
                set(result.marker_ids),
                result.generation,
            )
