
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .tools import is_side_effect


class Provenance(str, Enum):
    USER = "user"
    VISUAL_DERIVED = "visual_derived"
    TOOL_DERIVED = "tool_derived"
    CONVENTION = "convention"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    RESIDUAL = "residual"


@dataclass
class TaggedArgument:
    name: str
    value: Any
    provenance: Provenance


@dataclass
class TaggedToolCall:
    function: str
    arguments: list[TaggedArgument]
    policy: PolicyDecision | None = None
    is_legitimate: bool | None = None


@dataclass
class PolicyResult:
    total_calls: int = 0
    residual_calls: int = 0
    allowed_calls: int = 0
    tagged_calls: list[TaggedToolCall] = field(default_factory=list)


def tag_provenance(
    function: str,
    arguments: dict[str, Any],
    user_prompt: str,
    visual_params: set[str] | None = None,
) -> TaggedToolCall:
    tagged = []
    prompt_lower = user_prompt.lower()
    for name, value in arguments.items():
        if visual_params is not None:
            prov = (Provenance.VISUAL_DERIVED if name in visual_params
                    else Provenance.USER)
        else:
            str_value = str(value).lower().strip()
            if len(str_value) > 2 and str_value in prompt_lower:
                prov = Provenance.USER
            else:
                prov = Provenance.VISUAL_DERIVED
        tagged.append(TaggedArgument(name=name, value=value, provenance=prov))
    return TaggedToolCall(function=function, arguments=tagged)


def apply_policy(calls: list[TaggedToolCall]) -> PolicyResult:
    result = PolicyResult(total_calls=len(calls))
    for call in calls:
        if not is_side_effect(call.function):
            call.policy = PolicyDecision.ALLOW
            result.allowed_calls += 1
        elif any(a.provenance == Provenance.VISUAL_DERIVED for a in call.arguments):
            call.policy = PolicyDecision.RESIDUAL
            result.residual_calls += 1
        else:
            call.policy = PolicyDecision.ALLOW
            result.allowed_calls += 1
        result.tagged_calls.append(call)
    return result
