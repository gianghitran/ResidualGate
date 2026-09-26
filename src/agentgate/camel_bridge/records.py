
from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any, Iterable


class DenialSource(str, enum.Enum):

    NO_SIDE_EFFECT = "no_side_effect"
    ALLOWED = "allowed"
    DATAFLOW_GUARD = "dataflow_guard"
    TOOL_POLICY = "tool_policy"
    COVERAGE_FALLBACK = "coverage_fallback"
    POLICY_ERROR = "policy_error"

    def __str__(self) -> str:
        return self.value

    @property
    def is_residual(self) -> bool:
        return self in _RESIDUAL_SOURCES

    @property
    def partition(self) -> str:
        if self is DenialSource.POLICY_ERROR:
            return "policy_error"
        return "policy_residual" if self.is_residual else "policy_decidable"


_RESIDUAL_SOURCES = frozenset(
    {DenialSource.DATAFLOW_GUARD, DenialSource.TOOL_POLICY, DenialSource.COVERAGE_FALLBACK}
)

_DATAFLOW_MARKER = "is state-changing and depends on private values"
_FALLBACK_MARKER = "No security policy matched for tool"


def classify(*, denied: bool, reason: str | None, tool_name: str, no_side_effect_tools: frozenset[str],
             raised: bool = False) -> DenialSource:
    if raised:
        return DenialSource.POLICY_ERROR
    if not denied:
        return DenialSource.NO_SIDE_EFFECT if tool_name in no_side_effect_tools else DenialSource.ALLOWED
    text = reason or ""
    if _DATAFLOW_MARKER in text:
        return DenialSource.DATAFLOW_GUARD
    if _FALLBACK_MARKER in text:
        return DenialSource.COVERAGE_FALLBACK
    return DenialSource.TOOL_POLICY


@dataclasses.dataclass(frozen=True, slots=True)
class PolicyCallRecord:

    trace_id: str
    suite: str
    user_task_id: str
    injection_task_id: str | None
    attack_name: str | None
    call_index: int
    tool_name: str
    denied: bool
    denial_source: DenialSource
    reason: str | None
    is_tool: bool = True
    raised: bool = False
    gateway_label: str | None = None
    model_label: str | None = None
    system: str | None = None
    payload_readers: str | None = None

    @property
    def partition(self) -> str:
        return self.denial_source.partition

    @property
    def provenance_group(self) -> str:
        return f"{self.suite}:{'attack' if self.attack_name else 'utility'}"

    @property
    def is_enforced_residual(self) -> bool:
        return self.is_tool and self.denial_source.is_residual

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["denial_source"] = self.denial_source.value
        d["partition"] = self.partition
        d["provenance_group"] = self.provenance_group
        return d


def write_jsonl(records: Iterable[PolicyCallRecord | dict], path: Path) -> int:
    n = 0
    with path.open("w") as fh:
        for record in records:
            row = record if isinstance(record, dict) else record.to_json()
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n

