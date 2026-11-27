import inspect
import keyword
from typing import Any, Literal, Union

JSON_TYPE_TO_PYTHON_TYPE = {
    "string": str,
    "number": float,
    "boolean": bool,
    "integer": int,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _dedupe_types(types: list[Any]) -> list[Any]:
    seen = []
    for t in types:
        if t not in seen:
            seen.append(t)
    return seen


def _annotation_from_spec(spec: dict[str, Any]) -> Any:
    if not spec or not isinstance(spec, dict):
        return Any

    # const → Literal[value]
    if "const" in spec:
        return Literal[spec["const"]]

    # enum → Literal[val1, val2, ...]
    if "enum" in spec:
        values = spec["enum"]
        if not values:
            return Any
        return Literal[tuple(values)]

    # anyOf / oneOf → Union[Type1, Type2, ...]
    for key in ("anyOf", "oneOf"):
        if key in spec:
            variants = [_annotation_from_spec(s) for s in spec[key]]
            unique = _dedupe_types(variants)
            if len(unique) == 0:
                return Any
            if len(unique) == 1:
                return unique[0]
            return Union[tuple(unique)]

    # allOf → merge sub-schemas then resolve
    if "allOf" in spec:
        merged = {}
        for sub in spec["allOf"]:
            merged.update(sub)
        return _annotation_from_spec(merged)

    json_type = spec.get("type")

    if json_type is None:
        return Any

    # type as list, e.g. ["string", "null"] → Union[str, None] (Optional)
    if isinstance(json_type, list):
        types = [JSON_TYPE_TO_PYTHON_TYPE.get(t, Any) for t in json_type]
        unique = _dedupe_types(types)
        if len(unique) == 0:
            return Any
        if len(unique) == 1:
            return unique[0]
        return Union[tuple(unique)]

    # array with items → list[ItemType]
    if json_type == "array" and "items" in spec:
        item_type = _annotation_from_spec(spec["items"])
        return list[item_type]  # type: ignore[valid-type]

    return JSON_TYPE_TO_PYTHON_TYPE.get(json_type, Any)


def _needs_alias(name: str) -> bool:
    return not name.isidentifier() or keyword.iskeyword(name)


def _identifier_base(name: str) -> str:
    """Turn *name* into a valid Python identifier without collision handling."""
    base = "".join(char if f"a{char}".isidentifier() else "_" for char in name)
    if not base.isidentifier():
        base = f"_{base}"
    if keyword.iskeyword(base):
        base = f"{base}_"
    return base


def parameter_names_from_schema(schema: object | None) -> dict[str, str]:
    """Map each schema property name to the Python parameter name the prompt shows.

    Names that are already valid identifiers are kept. Other names get an alias,
    assigned in schema ``properties`` order and never equal to another property's
    name or alias.
    """
    if not schema or not isinstance(schema, dict):
        return {}
    props = schema.get("properties", {})
    used = {name for name in props if not _needs_alias(name)}
    names: dict[str, str] = {}
    for name in props:
        if not _needs_alias(name):
            names[name] = name
            continue
        base = _identifier_base(name)
        alias = base
        suffix = 1
        while alias in used:
            suffix += 1
            alias = f"{base}_{suffix}"
        used.add(alias)
        names[name] = alias
    return names


def signature_from_schema(schema: object | None) -> inspect.Signature:
    if not schema or not isinstance(schema, dict):
        return inspect.Signature()

    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = []

    # JSON Schema does not require properties to list required fields first.
    # Python signatures do: parameters without defaults must precede parameters
    # with defaults. Keep the schema's order within both groups while moving
    # required fields without schema defaults ahead of optional/defaulted ones.
    property_items = list(props.items())
    parameter_names = parameter_names_from_schema(schema)
    required_without_default = [
        (name, spec)
        for name, spec in property_items
        if name in required and not (isinstance(spec, dict) and "default" in spec)
    ]
    defaulted_or_optional = [
        (name, spec)
        for name, spec in property_items
        if not (name in required and not (isinstance(spec, dict) and "default" in spec))
    ]

    for name, spec in [*required_without_default, *defaulted_or_optional]:
        annotation = _annotation_from_spec(spec)

        has_schema_default = "default" in spec if isinstance(spec, dict) else False

        if name in required and not has_schema_default:
            default = inspect.Parameter.empty
        elif has_schema_default:
            default = spec["default"]
        else:
            default = None

        params.append(
            inspect.Parameter(
                parameter_names[name],
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )

    return inspect.Signature(params)
