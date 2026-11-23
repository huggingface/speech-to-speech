"""Scripted tool-use conversations for a desk-robot voice assistant.

A scenario set is a list of conversations; a conversation is a list of user turns, each with
the tool it should lead to (``None`` when no tool is needed). The client runs the tools itself:
most return at once, ``research_report`` is long-running and returns two user turns later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

INSTRUCTIONS = (
    "You are Reachy, a friendly desk robot with a camera, a head you can move and antennas. "
    "You help the person in front of you with small everyday things and keep them company."
)

LONG_RUNNING_TOOLS = frozenset({"research_report"})
LATE_RESULT_AFTER_TURNS = 2

_STRING = {"type": "string"}


def _tool(name: str, description: str, properties: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    properties = properties or {}
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties)},
    }


TOOLS = [
    _tool("take_picture", "Take a picture with your camera and get a description of what is in front of you."),
    _tool("get_weather", "Get the current weather for a city.", {"city": _STRING}),
    _tool("get_time", "Get the current local time in a city.", {"city": _STRING}),
    _tool("set_timer", "Start a countdown timer.", {"minutes": {"type": "integer"}, "label": _STRING}),
    _tool("check_timer", "Check how much time is left on a timer.", {"label": _STRING}),
    _tool("search_web", "Search the web and return short snippets.", {"query": _STRING}),
    _tool("remember", "Save a fact about the user for later conversations.", {"fact": _STRING}),
    _tool("dance", "Perform a short dance with your head and antennas.", {"style": _STRING}),
    _tool("play_emotion", "Express an emotion with your head and antennas.", {"emotion": _STRING}),
    _tool("move_head", "Turn your head in a direction.", {"direction": _STRING}),
    _tool(
        "research_report", "Write an in-depth research report on a topic. Takes several minutes.", {"topic": _STRING}
    ),
]

EXPRESSION_TOOLS = frozenset({"dance", "play_emotion", "move_head"})

RESULTS: dict[str, Any] = {
    "take_picture": {"description": "A desk with a laptop, a green plant and a red notebook."},
    "get_weather": {"temp_c": 17, "sky": "cloudy"},
    "get_time": {"time": "14:05"},
    "set_timer": {"status": "started"},
    "check_timer": {"remaining_minutes": 12},
    "search_web": {"snippets": ["The first result answers it: 1969."]},
    "remember": {"status": "saved"},
    "dance": {"status": "done"},
    "play_emotion": {"status": "done"},
    "move_head": {"status": "done"},
    "research_report": {
        "report": "Solid-state batteries promise higher energy density; mass production is expected around 2030."
    },
}


@dataclass(frozen=True)
class Turn:
    ask: str
    expected_tool: Optional[str]


Conversation = tuple[Turn, ...]


def _conversation(*turns: tuple[str, Optional[str]]) -> Conversation:
    return tuple(Turn(ask, tool) for ask, tool in turns)


CONVERSATIONS: tuple[Conversation, ...] = (
    _conversation(
        ("What's the weather in Lisbon?", "get_weather"),
        ("Can you dance for me?", "dance"),
        ("Set a 10 minute timer for the eggs.", "set_timer"),
        ("What's the capital of Peru?", None),
        ("Show me your happiest face!", "play_emotion"),
        ("What time is it in Tokyo?", "get_time"),
        ("Can you see what's on my desk?", "take_picture"),
        ("How much time is left on the eggs?", "check_timer"),
        ("And the weather in Berlin?", "get_weather"),
        ("Thanks, that's all.", None),
    ),
    _conversation(
        ("Can you put together a research report on solid-state batteries?", "research_report"),
        ("Meanwhile, what's the weather in Rome?", "get_weather"),
        ("Can you dance a little?", "dance"),
        ("What do you see in front of you?", "take_picture"),
        ("Is the report ready yet?", None),
        ("Set a 3 minute timer for the tea.", "set_timer"),
        ("Look to your right.", "move_head"),
        ("What time is it in New York?", "get_time"),
    ),
    _conversation(
        ("Take a look, what am I holding?", "take_picture"),
        ("Look up when the first moon landing was.", "search_web"),
        ("Remember that I'm allergic to peanuts.", "remember"),
        ("Tell me a joke.", None),
        ("Show me how excited you are!", "play_emotion"),
        ("What time is it in Sydney?", "get_time"),
        ("Is it raining in London?", "get_weather"),
        ("Do a happy dance!", "dance"),
    ),
)

# Opening requests only: the first call of a conversation, with no history to learn from.
FIRST_TURNS: tuple[Conversation, ...] = tuple(
    _conversation(turn)
    for turn in (
        ("Can you put together a research report on solid-state batteries?", "research_report"),
        ("I need an in-depth report on deep-sea mining.", "research_report"),
        ("Please write me a detailed report about high-speed rail in Europe.", "research_report"),
        ("Could you research fusion startups and write it up for me?", "research_report"),
        ("What's the weather in Lisbon?", "get_weather"),
        ("Is it raining in Oslo right now?", "get_weather"),
        ("Do I need a jacket in Quebec today?", "get_weather"),
        ("What time is it in Tokyo?", "get_time"),
        ("What's the time in New York right now?", "get_time"),
        ("Set a 10 minute timer for the eggs.", "set_timer"),
        ("Can you start a five minute timer for the tea?", "set_timer"),
        ("Take a look, what am I holding?", "take_picture"),
        ("Can you see what's on my desk?", "take_picture"),
        ("What color is my shirt?", "take_picture"),
        ("Look up when the first moon landing was.", "search_web"),
        ("Who won the last Tour de France?", "search_web"),
        ("Remember that I'm allergic to peanuts.", "remember"),
        ("Please remember my sister is called Ana.", "remember"),
        ("Can you dance for me?", "dance"),
        ("Do a happy dance!", "dance"),
        ("Show me your happiest face!", "play_emotion"),
        ("Show me how excited you are!", "play_emotion"),
        ("Look to your right.", "move_head"),
        ("Turn your head to the left, please.", "move_head"),
    )
)

SCENARIO_SETS: dict[str, tuple[Conversation, ...]] = {
    "conversations": CONVERSATIONS,
    "first-turns": FIRST_TURNS,
}
