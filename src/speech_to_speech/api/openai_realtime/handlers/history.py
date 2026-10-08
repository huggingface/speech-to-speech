from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.pipeline.events import ResponseGenerationDoneEvent
from speech_to_speech.pipeline.speculative_turns import TurnGateAction

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService
    from speech_to_speech.pipeline.history import ResponseHistory
    from speech_to_speech.pipeline.messages import GenerateResponseRequest


class HistoryCommitError(RuntimeError):
    """The service could not write history for output it accepted."""


@dataclass
class _ResponseHistoryState:
    proposal: ResponseHistory | None = None
    accepted: bool = False
    rejected: bool = False
    written: int = 0
    last_item_id: str | None = None
    done: bool = False


class HistoryHandler(RealtimeBaseHandler):
    """The only writer of model output to a session's chat.

    Models attach cumulative :class:`ResponseHistory` proposals to their
    output. ``hold`` keeps the newest proposal per response. ``accept`` runs
    after the turn gate commits a response and writes what it holds; later
    proposals for that response are written as they arrive. ``close`` undoes
    whatever a response wrote unless ``finalize`` made it permanent first.
    """

    def __init__(self, service: RealtimeService) -> None:
        super().__init__(service)
        self._responses: dict[tuple[str, str], _ResponseHistoryState] = {}

    def hold(self, conn_id: str, event: object) -> None:
        """Keep the proposal carried by *event* without writing it."""
        self._keep(conn_id, event)

    def stage(self, conn_id: str, event: object) -> None:
        """Keep the proposal carried by *event*; write it if already accepted."""
        kept = self._keep(conn_id, event)
        if kept is not None and kept[1].accepted:
            self._write(conn_id, *kept)

    def accept(self, conn_id: str, key: str | None) -> None:
        """Write a response's history. Call after the gate commits its turn."""
        if key is None or key in self._state(conn_id).closed_response_keys:
            return
        state = self._responses.setdefault((conn_id, key), _ResponseHistoryState())
        state.accepted = True
        self._write(conn_id, key, state)

    def reject(self, conn_id: str, key: str) -> None:
        """Drop a failed response's proposal and undo what it wrote."""
        st = self._state(conn_id)
        if key in st.closed_response_keys:
            return
        state = self._responses.setdefault((conn_id, key), _ResponseHistoryState())
        state.rejected = True
        state.proposal = None
        st.runtime_config.chat.rollback_provisional_generation(key)

    def finalize(self, conn_id: str, key: str | None) -> None:
        """Make a completed response's history permanent."""
        self._state(conn_id).runtime_config.chat.finalize_provisional_generation(key)

    def close(self, conn_id: str, key: str | None) -> None:
        """Forget a closed response and undo any history it did not finalize."""
        if key is None:
            return
        self._responses.pop((conn_id, key), None)
        self._state(conn_id).runtime_config.chat.rollback_provisional_generation(key)

    def close_session(self, conn_id: str) -> None:
        for owner, key in tuple(self._responses):
            if owner == conn_id:
                del self._responses[(owner, key)]

    def _keep(self, conn_id: str, event: object) -> tuple[str, _ResponseHistoryState] | None:
        key = getattr(event, "response_key", None)
        st = self._state(conn_id)
        if key is None or key in st.closed_response_keys:
            return None
        if isinstance(event, ResponseGenerationDoneEvent) and not event.succeeded:
            self.reject(conn_id, key)
            return None
        proposal: ResponseHistory | None = getattr(event, "history", None)
        if proposal is None or proposal.chat is not st.runtime_config.chat:
            return None  # Late output from a released session must not write a new chat.
        state = self._responses.setdefault((conn_id, key), _ResponseHistoryState())
        if state.rejected:
            return None
        held = state.proposal
        if held is None or (len(proposal.items), proposal.complete) > (len(held.items), held.complete):
            state.proposal = proposal
        prefetch = st.tool_followup_prefetch_request
        if (
            proposal.complete
            and prefetch is not None
            and prefetch.response_key == key
            and prefetch.prefetch_transaction is not None
        ):
            # Only completion releases the provider abort callback. A tool
            # prefix must stay abortable while the hidden stream runs.
            prefetch.prefetch_transaction.complete(partial(self._claim_prefetch, conn_id, prefetch))
        return key, state

    def _claim_prefetch(self, conn_id: str, request: GenerateResponseRequest) -> None:
        tracker = self._service.speculative_turns
        if tracker is not None:
            decision = tracker.gate(request.turn_id, request.turn_revision, commit=True)
            if decision.action is not TurnGateAction.ACCEPT:
                raise HistoryCommitError("Prefetched history no longer belongs to an accepted turn")
        self.accept(conn_id, request.response_key)

    def _write(self, conn_id: str, key: str, state: _ResponseHistoryState) -> None:
        proposal = state.proposal
        if proposal is None or state.rejected or state.done:
            return
        try:
            # A complete proposal is written even with no new items: it
            # commits the input item that earlier tool prefixes left reversible.
            if state.written < len(proposal.items) or proposal.complete:
                written = proposal.write(key, state.written, state.last_item_id)
                state.written = len(proposal.items)
                if written:
                    state.last_item_id = written[-1].id
            if proposal.complete:
                proposal.clean_up()
                state.done = True
        except Exception as exc:
            self.reject(conn_id, key)
            raise HistoryCommitError(f"Language model history commit failed: {exc}") from exc
