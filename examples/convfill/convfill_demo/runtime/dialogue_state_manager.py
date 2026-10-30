import threading


class DialogueStateManager:
    def __init__(self, num_history_turns=0):
        self.transcript = []
        self.user_turns = []
        # This should be a list of lists, where each inner list is the assistant's thoughts for a turn.
        self.thoughts = []
        self.responses = []
        self.num_history_turns = num_history_turns
        self.lock = threading.Lock()

    # Guarded from multiple adds
    def update_user_state(self, user_turn):
        if len(self.user_turns) == len(self.responses):
            self.transcript.append(f"User: {user_turn}")
            self.user_turns.append(user_turn)

    def update_response_state(self, assistant_response):
        self.transcript.append(f"You: {assistant_response}")
        self.responses.append(assistant_response)
        assert len(self.user_turns) == len(self.responses)

    # Must happen from the turn state manager in the correct format
    # assistant_thoughts should be a list of strings
    def update_thoughts_state(self, assistant_thoughts):
        non_sil_thoughts = [thought for thought in assistant_thoughts if thought != "<sil>"]
        self.thoughts.append(non_sil_thoughts)
        assert len(self.thoughts) == len(self.user_turns)

    # Attaches a response to the pending user turn in one step, so an
    # interrupted turn lands with the same list invariants as a completed one.
    def commit_turn(self, user_turn, response, thoughts=None):
        with self.lock:
            assert self.user_turns, "commit_turn called with no pending user turn"
            assert self.user_turns[-1] == user_turn, "commit_turn user turn does not match the pending turn"
            assert len(self.responses) == len(self.user_turns) - 1, "commit_turn called on an already answered turn"
            self.update_response_state(response)
            self.update_thoughts_state(list(thoughts or []))
            return True

    def get_transcript(self) -> str:
        """Return the conversation the Reasoner should continue.

        The Reasoner must see what the user actually heard from the Talker,
        not the Reasoner's private knowledge sentences from the prior turn.
        Otherwise follow-up references can be grounded in text that was
        never spoken, while the real assistant response is absent.
        """
        assert len(self.thoughts) == len(self.responses)
        lines = []
        for i, user_turn in enumerate(self.user_turns):
            lines.append(f"User: {user_turn}")
            if i < len(self.responses):
                lines.append(f"You: {self.responses[i]}")
        return "\n".join(lines)

    def get_history(self):
        # Get the history without the latest user turn and response, since those are not confirmed until the next turn
        confirmed_history_len = min(len(self.user_turns), len(self.responses))
        confirmed_user_history = self.user_turns[0:confirmed_history_len]
        confirmed_response_history = self.responses[0:confirmed_history_len]
        assert len(confirmed_user_history) == len(confirmed_response_history)

        history = {"user_turns_history": [], "responses_history": []}
        num_user_turns = len(confirmed_user_history)
        start_idx = max(0, num_user_turns - self.num_history_turns)
        assert len(confirmed_user_history) == len(confirmed_response_history)
        for i in range(start_idx, num_user_turns):
            history["user_turns_history"].append(f"{confirmed_user_history[i]}")
            history["responses_history"].append(f"{confirmed_response_history[i]}")
        return history

    def reset(self):
        with self.lock:
            self.transcript = []
            self.user_turns = []
            self.responses = []
            self.thoughts = []

    def rollback_turn(self, user_turn):
        """Remove the newest matching turn, whether pending or completed.

        Realtime transports can cancel a turn after STT has committed the user
        text but before its audio response completes. ConvFill must not retain
        that partial turn in either the Talker or Reasoner history.
        """
        with self.lock:
            if not self.user_turns or self.user_turns[-1] != user_turn:
                return False

            self.user_turns.pop()
            if len(self.responses) > len(self.user_turns):
                self.responses.pop()
            if len(self.thoughts) > len(self.user_turns):
                self.thoughts.pop()

            # ``transcript`` is retained for compatibility with older callers;
            # rebuild it from the canonical lists after the rollback.
            self.transcript = []
            for idx, response in enumerate(self.responses):
                self.transcript.append(f"User: {self.user_turns[idx]}")
                self.transcript.append(f"You: {response}")
            return True
