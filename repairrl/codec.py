"""JSON-only transport for HumanEval function arguments/results; never pickle."""
import math


def encode(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"type": "float", "value": repr(value)}
    if type(value) in (list, tuple, set):
        return {"type": type(value).__name__, "items": [encode(v) for v in value]}
    if type(value) is dict:
        return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
    raise TypeError("Unsupported function transport value")


def decode(value, depth=0):
    if depth > 100:
        raise ValueError("Transport depth exceeded")
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("Invalid transport envelope")
    kind = value.get("type")
    if kind == "float" and value.get("value") in ("inf", "-inf", "nan"):
        return float(value["value"])
    items = value["items"]
    if kind == "dict":
        return {decode(k, depth + 1): decode(v, depth + 1) for k, v in items}
    constructors = {"list": list, "tuple": tuple, "set": set}
    if kind not in constructors:
        raise ValueError("Invalid transport type")
    return constructors[kind](decode(v, depth + 1) for v in items)
