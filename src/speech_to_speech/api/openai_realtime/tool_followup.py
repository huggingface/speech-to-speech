"""Metadata shared by the packaged tool client and Realtime server."""

import json
from typing import Any

TOOL_INPUT_METADATA_KEY = "s2s_tool_input_call_ids"
TOOL_FOLLOWUP_METADATA_KEY = "s2s_tool_followup_call_ids"
TOOL_FOLLOWUP_COVERED = "tool_followup_already_answered"
TOOL_FOLLOWUP_WAIT = "tool_followup_user_turn_open"


def tool_call_ids(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    try:
        decoded = json.loads(value)
    except ValueError:
        return set()
    if not isinstance(decoded, list) or not all(isinstance(item, str) and item for item in decoded):
        return set()
    return set(decoded)
