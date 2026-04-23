import inspect
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


def _dedupe_types(types: list) -> list:
    seen = []
    for t in types:
        if t not in seen:
            seen.append(t)
    return seen


def _annotation_from_spec(spec: dict):
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


def signature_from_schema(schema: object | None) -> inspect.Signature:
    if not schema or not isinstance(schema, dict):
        return inspect.Signature()

    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = []

    for name, spec in props.items():
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
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )

    return inspect.Signature(params)