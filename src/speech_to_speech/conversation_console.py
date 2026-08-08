"""Console controls for optional conversation text output."""

from rich.console import Console

console = Console()


def configure_conversation_text_output(*, enabled: bool) -> None:
    """Enable or suppress user and assistant text written by pipeline handlers."""

    console.quiet = not enabled
