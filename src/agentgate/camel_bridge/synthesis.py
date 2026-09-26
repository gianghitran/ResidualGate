
from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Iterable, Sequence

__all__ = ["ArgumentOrigin", "Synthesis", "classify_argument", "synthesise",
           "synthesise_attack", "AttackMode", "PrintMode", "PRINT_FOR_UTILITY",
           "PRINT_FOR_SECURITY"]

PrintMode = str
PRINT_FOR_UTILITY: PrintMode = "all"
PRINT_FOR_SECURITY: PrintMode = "last"

_WS = re.compile(r"\s+")


def _norm(value: Any) -> str:
    return _WS.sub(" ", str(value).strip().lower())


def _atoms(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return [a for v in value for a in _atoms(v)]
    if isinstance(value, dict):
        return [a for v in value.values() for a in _atoms(v)]
    return [value]


def _contained(value: Any, blob: Any) -> bool:
    haystack = _norm(blob)
    parts = [_norm(a) for a in _atoms(value)]
    parts = [p for p in parts if len(p) >= 3]
    return bool(parts) and all(p in haystack for p in parts)


class ArgumentOrigin(str):

    USER = "user"
    VISUAL_DERIVED = "visual_derived"
    DERIVED_LOCATED = "derived_located"
    DERIVED_INFERRED = "derived_inferred"
    CONVENTION = "convention"


def classify_argument(value: Any, prompt: str, prior_outputs: Sequence[Any]) -> str:
    if _contained(value, prompt):
        return ArgumentOrigin.USER
    if any(_contained(value, out) for out in prior_outputs):
        return ArgumentOrigin.DERIVED_LOCATED
    if prior_outputs:
        return ArgumentOrigin.DERIVED_INFERRED
    return ArgumentOrigin.CONVENTION


_TYPES = {bool: "bool", int: "int", float: "float", str: "str"}


def _annotation(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    for py_type, name in _TYPES.items():
        if type(value) is py_type:
            return name
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


def _literal(value: Any) -> str:
    return repr(value)


@dataclasses.dataclass
class Synthesis:

    code: str
    qllm_values: list[dict[str, Any]]
    origins: dict[str, int] = dataclasses.field(default_factory=dict)
    filled: list[str] = dataclasses.field(default_factory=list)

    @property
    def fenced(self) -> str:
        return f"```python\n{self.code}```"


def empty_like(schema: dict[str, Any] | None) -> Any:
    if not schema:
        return ""
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            if branch.get("type") != "null":
                return empty_like(branch)
        return ""
    return {"array": [], "object": {}, "string": "", "integer": 0,
            "number": 0.0, "boolean": False}.get(schema.get("type"), "")


def _no_schema(function: str, name: str) -> str:
    return ""


def synthesise(prompt: str, calls: Sequence[Any], outputs: Sequence[Any],
               *, required_args: dict[str, Iterable[str]] | None = None,
               print_mode: PrintMode = PRINT_FOR_UTILITY,
               empty_value: Callable[[str, str], Any] = _no_schema) -> Synthesis:
    required_args = required_args or {}
    lines: list[str] = []
    qllm_values: list[dict[str, Any]] = []
    origins: dict[str, int] = {}
    filled: list[str] = []

    for index, call in enumerate(calls):
        prior = outputs[:index]
        prior_vars = [f"v{i}" for i in range(index)]
        literal: dict[str, Any] = {}
        derived: dict[str, Any] = {}

        for name, value in call.args.items():
            origin = classify_argument(value, prompt, prior)
            origins[origin] = origins.get(origin, 0) + 1
            if origin in (ArgumentOrigin.USER, ArgumentOrigin.CONVENTION):
                literal[name] = value
            else:
                derived[name] = value

        for name in required_args.get(call.function, ()):
            if name not in call.args:
                derived.setdefault(name, empty_value(call.function, name))
                filled.append(f"{call.function}.{name}")

        kwargs: dict[str, str] = {n: _literal(v) for n, v in literal.items()}

        if derived:
            model = f"Extracted{index}"
            lines.append(f"class {model}(BaseModel):")
            for name, value in derived.items():
                lines.append(f"    {name}: {_annotation(value)}")
            lines.append("")
            context = ", ".join(f"{{{v}}}" for v in prior_vars) or "no prior output"
            fields = ", ".join(derived)
            lines.append(f'e{index} = query_ai_assistant(f"Extract {fields} for '
                         f'{call.function}. Context: {context}", {model})')
            qllm_values.append(dict(derived))
            for name in derived:
                kwargs[name] = f"e{index}.{name}"

        rendered = ", ".join(f"{n}={v}" for n, v in kwargs.items())
        lines.append(f"v{index} = {call.function}({rendered})")
        lines.append("")

    if calls and print_mode == "all":
        for index in range(len(calls)):
            lines.append(f"print(v{index})")
    elif calls and print_mode == "last":
        lines.append(f"print(v{len(calls) - 1})")
    elif print_mode not in ("all", "last", "none"):
        raise ValueError(f"unknown print mode {print_mode!r}")

    return Synthesis("\n".join(lines).rstrip() + "\n", qllm_values, origins, filled)



AttackMode = str


def synthesise_attack(prompt: str, user_calls: Sequence[Any], user_outputs: Sequence[Any],
                      attacker_calls: Sequence[Any], *, mode: AttackMode = "chain",
                      required_args: dict[str, Iterable[str]] | None = None,
                      print_mode: PrintMode = PRINT_FOR_SECURITY,
                      empty_value: Callable[[str, str], Any] = _no_schema) -> Synthesis:
    if mode == "chain":
        return synthesise(prompt, [*user_calls, *attacker_calls],
                          [*user_outputs, *([""] * len(attacker_calls))],
                          required_args=required_args, print_mode=print_mode,
                          empty_value=empty_value)
    if mode != "substitute":
        raise ValueError(f"unknown attack mode {mode!r}")

    if not user_calls or not attacker_calls:
        return synthesise(prompt, user_calls, user_outputs, required_args=required_args,
                          print_mode=print_mode, empty_value=empty_value)

    target, source = user_calls[-1], attacker_calls[-1]
    merged = dict(target.args)
    swapped = False
    for name, value in source.args.items():
        if name in merged:
            merged[name] = value
            swapped = True
    if not swapped:
        return synthesise_attack(prompt, user_calls, user_outputs, attacker_calls,
                                 mode="chain", required_args=required_args,
                                 print_mode=print_mode, empty_value=empty_value)

    hijacked = dataclasses.replace(target) if dataclasses.is_dataclass(target) else _Call(
        target.function, merged)
    if dataclasses.is_dataclass(target):
        hijacked = _Call(target.function, merged)
    return synthesise(prompt, [*user_calls[:-1], hijacked], user_outputs,
                      required_args=required_args, print_mode=print_mode,
                      empty_value=empty_value)


@dataclasses.dataclass
class _Call:

    function: str
    args: dict[str, Any]
