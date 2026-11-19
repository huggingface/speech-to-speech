from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from speech_to_speech.backend_registry import (
    LLM_BACKENDS,
    BackendSpec,
    HandlerContext,
)

from .handler import ConvFillRealtimeHandler


@dataclass
class ConvFillHandlerArguments:
    convfill_config_path: str = field(
        default="configs/demo_mode/convfill_hosted_config.json",
        metadata={"help": "Path to the ConvFill overall demo configuration."},
    )
    convfill_mode: str = field(
        default="normal",
        metadata={"help": "ConvFill task mode (this example supports normal)."},
    )


def _create_handler(context: HandlerContext, config: Mapping[str, Any]) -> ConvFillRealtimeHandler:
    return ConvFillRealtimeHandler(
        context.stop_event,
        queue_in=context.queue_in,
        queue_out=context.queue_out,
        setup_kwargs={
            **dict(config),
            "cancel_scope": context.cancel_scope,
            "speculative_turns": context.speculative_turns,
        },
    )


def register_convfill_backend() -> None:
    """Register the in-repo adapter before HF parses its backend choices."""
    existing = LLM_BACKENDS.get("convfill")
    if existing is not None:
        if existing.create_handler is not _create_handler:
            raise RuntimeError("A different Hugging Face LLM backend named 'convfill' is already registered.")
        return
    LLM_BACKENDS["convfill"] = BackendSpec(
        name="convfill",
        kind="llm",
        config_type=ConvFillHandlerArguments,
        create_handler=_create_handler,
        config_prefix="convfill",
    )
