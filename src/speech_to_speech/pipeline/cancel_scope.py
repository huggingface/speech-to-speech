class CancelScope:
    """Unified cancellation signal for the speech-to-speech pipeline.

    Uses a generation counter so pipeline threads (LLM, TTS) can detect
    cancellation without brief-pulse timing games, and an internal
    ``discarding`` flag so the async send loop can drop stale output.

    Thread safety: one writer (the asyncio router thread) and multiple
    readers (pipeline handler threads).  Python's GIL makes int/bool
    reads and writes atomic at the bytecode level, so no lock is needed.
    """

    def __init__(self) -> None:
        self._gen: int = 0
        self._discarding: bool = False

    @property
    def generation(self) -> int:
        """Current generation number.  Pipeline threads capture this at
        the start of each response and compare with ``is_stale``."""
        return self._gen

    def cancel(self) -> None:
        """Cancel the current response.

        Increments the generation (so pipeline threads see their captured
        generation as stale) and enables the send-loop discard guard.
        """
        # prevent overflow... after 4 billion generations, we'll wrap around xD...
        self._gen = (self._gen + 1) & 0xFFFFFFFF
        self._discarding = True

    def response_done(self) -> None:
        """Pipeline acknowledged completion.  Clears the discard guard."""
        self._discarding = False

    def new_response(self) -> None:
        """An explicit ``response.create`` starts a new response.
        Clears the discard guard."""
        self._discarding = False

    def is_stale(self, gen: int) -> bool:
        """Return True if *gen* has been superseded by a ``cancel`` call."""
        return gen != self._gen

    @property
    def discarding(self) -> bool:
        """Whether the send loop should silently drop stale output."""
        return self._discarding

    def reset(self) -> None:
        """Clear discard state (e.g. on new session connect)."""
        self._discarding = False
