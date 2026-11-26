from collections.abc import Iterator

import numpy as np


class PCMBlockBuffer:
    """Pack mono int16 samples into blocks without adding or removing samples.

    Use one buffer per synthesis call. Flush on completion; discard on cancellation.
    Returned blocks own their storage, including the short final block.
    """

    def __init__(self, blocksize: int) -> None:
        if blocksize <= 0:
            raise ValueError("blocksize must be positive")
        self.blocksize = blocksize
        self._pending = np.empty(0, dtype=np.int16)

    def push(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        if self._pending.size:
            samples = np.concatenate((self._pending, samples))
        end = (samples.size // self.blocksize) * self.blocksize
        self._pending = samples[end:].copy()
        for offset in range(0, end, self.blocksize):
            yield samples[offset : offset + self.blocksize].copy()

    def flush(self) -> Iterator[np.ndarray]:
        tail = self._pending
        self._pending = np.empty(0, dtype=np.int16)
        if tail.size:
            yield tail
