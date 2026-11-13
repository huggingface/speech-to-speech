"""Optional shared ZeroGPU inference for native speech-to-speech handlers.

The runtime is deliberately lazy: import ``spaces`` first in the deployment,
then explicitly call ``runtime.initialize()`` before ZeroGPU startup.
"""

from . import runtime

__all__ = ["runtime"]
