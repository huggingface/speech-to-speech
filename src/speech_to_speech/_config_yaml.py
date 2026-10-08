"""Bounded YAML parsing. Imported only by the explicit configuration loader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from speech_to_speech.config import ConfigDocument, ConfigurationError

_MAX_BYTES = 1024 * 1024
_MAX_DEPTH = 32


def read_document(path: Path) -> ConfigDocument:
    try:
        import yaml
    except ImportError:
        raise ImportError('YAML configuration requires `pip install "speech-to-speech[config]"`.') from None

    def fail(message: str, mark: Any = None) -> None:
        location = f"{path}"
        if mark is not None:
            location += f":{mark.line + 1}:{mark.column + 1}"
        raise ConfigurationError(f"{location}: {message}") from None

    io_failure = False
    try:
        if path.stat().st_size > _MAX_BYTES:
            fail("Configuration exceeds the 1 MiB limit; use a smaller file.")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            fail("Configuration exceeds the 1 MiB limit; use a smaller file.")
        source = raw.decode("utf-8")
    except (OSError, UnicodeError):
        io_failure = True
    if io_failure:
        fail("Cannot read a UTF-8 configuration file; check the path and encoding.")

    class Loader(yaml.SafeLoader):
        yaml_implicit_resolvers: dict[Any, Any] = {}

    Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), list("tf"))
    Loader.add_implicit_resolver("tag:yaml.org,2002:null", re.compile(r"^(?:null|)$"), ["n", ""])
    Loader.add_implicit_resolver("tag:yaml.org,2002:int", re.compile(r"^-?(?:0|[1-9][0-9]*)$"), list("-0123456789"))
    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)$"),
        list("-0123456789"),
    )
    # Reject YAML non-finite spellings rather than treating them as harmless strings.
    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[+-]?\.(?:inf|Inf|INF|nan|NaN|NAN)$"),
        list("+-."),
    )
    locations: dict[tuple[str | int, ...], tuple[int, int]] = {}
    parse_failure: tuple[str, Any] | None = None
    try:
        depth = 0
        for event in yaml.parse(source, Loader=Loader):
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) is not None:
                fail("Anchors and aliases are unsupported; repeat the value explicitly.", event.start_mark)
            if getattr(event, "tag", None) is not None:
                fail("Explicit YAML tags are unsupported; use plain JSON-compatible values.", event.start_mark)
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                if depth > _MAX_DEPTH:
                    fail("Container nesting exceeds 32 levels; flatten the configuration.", event.start_mark)
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
        loader = Loader(source)
        try:
            root = loader.get_single_node()
            if not isinstance(root, yaml.MappingNode):
                fail("Use one nonempty mapping document as the configuration root.")

            def construct(node: Any, key_path: tuple[str | int, ...]) -> Any:
                locations[key_path] = (node.start_mark.line + 1, node.start_mark.column + 1)
                if isinstance(node, yaml.MappingNode):
                    result: dict[str, Any] = {}
                    for key_node, value_node in node.value:
                        key = loader.construct_object(key_node)
                        if not isinstance(key, str):
                            fail("Mapping keys must be strings.", key_node.start_mark)
                        if key == "<<":
                            fail("YAML merge keys are unsupported; provide settings explicitly.", key_node.start_mark)
                        if key in result:
                            fail("Duplicate mapping key; keep one definition per key.", key_node.start_mark)
                        result[key] = construct(value_node, (*key_path, key))
                    return result
                if isinstance(node, yaml.SequenceNode):
                    return [construct(item, (*key_path, index)) for index, item in enumerate(node.value)]
                value = loader.construct_object(node)
                if isinstance(value, float):
                    import math

                    if not math.isfinite(value):
                        fail("Numeric values must be finite.", node.start_mark)
                return value

            data = construct(root, ())
        finally:
            loader.dispose()
    except ConfigurationError:
        raise
    except ValueError:
        parse_failure = ("Invalid numeric value; use bounded decimal numbers.", None)
    except yaml.YAMLError as exc:
        parse_failure = ("Invalid YAML; correct the syntax and use one document.", getattr(exc, "problem_mark", None))
    # Raise outside the parser exception handler to remove sensitive exception context.
    if parse_failure is not None:
        fail(*parse_failure)
    return ConfigDocument(path, data, locations)
