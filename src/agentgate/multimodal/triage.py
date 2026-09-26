
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ..labels import Label
from ..camel_bridge.slm import SYSTEM as SLM_SYSTEM, ArgumentView, CallView

from .policy import PolicyDecision, Provenance, TaggedToolCall


@dataclass
class TriageVerdict:
    function: str
    label: Label
    rationale: str | None = None
    evidence_field: str | None = None
    latency_ms: float = 0.0


@dataclass
class TriageResult:
    verdicts: list[TriageVerdict] = field(default_factory=list)
    blocked: int = 0
    reviewed: int = 0
    failures: int = 0


def _call_view_from_tagged(call: TaggedToolCall, goal: str) -> CallView:
    args = tuple(
        ArgumentView(
            name=a.name,
            value=str(a.value)[:160],
            trusted=(a.provenance == Provenance.USER),
            public=True,
            readers=("anyone",),
        )
        for a in call.arguments
    )
    visual_deps = tuple(
        str(a.value)[:160] for a in call.arguments
        if a.provenance == Provenance.VISUAL_DERIVED
    )
    reason = None
    if visual_deps:
        names = [a.name for a in call.arguments if a.provenance == Provenance.VISUAL_DERIVED]
        reason = (f"argument(s) {', '.join(names)} derived from visual extraction, "
                  "not directly from the user")
    return CallView(
        tool=call.function,
        arguments=args,
        non_public_dependencies=visual_deps,
        goal=goal,
        policy_reason=reason,
    )


def triage_residuals(
    calls: list[TaggedToolCall],
    user_prompt: str,
    *,
    gate: Any = None,
) -> TriageResult:
    result = TriageResult()

    for call in calls:
        if call.policy != PolicyDecision.RESIDUAL:
            continue

        if gate is None:
            label = Label.REVIEW
            rationale = "no SLM gate configured; defaulting to REVIEW"
            latency = 0.0
        else:
            view = _call_view_from_tagged(call, user_prompt)
            prompt = f"{SLM_SYSTEM}\n\n{view.structured()}\n"
            try:
                from ..camel_bridge.verdict import FAIL_CLOSED, parse
                generation = gate._backend.generate([prompt], gate._config)[0]
                verdict = parse(generation.text)
                label = verdict.label
                if label == Label.ALLOW:
                    label = Label.REVIEW
                rationale = verdict.rationale
                latency = generation.latency_ms
            except Exception as exc:
                result.failures += 1
                label = Label.REVIEW
                rationale = f"error: {exc}"
                latency = 0.0

        v = TriageVerdict(
            function=call.function,
            label=label,
            rationale=rationale,
            latency_ms=latency,
        )
        result.verdicts.append(v)
        if label == Label.BLOCK:
            result.blocked += 1
        else:
            result.reviewed += 1

    return result
