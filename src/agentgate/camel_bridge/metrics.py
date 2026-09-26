
from __future__ import annotations

import collections
import dataclasses
from typing import Iterable, Sequence

from .records import DenialSource, PolicyCallRecord

__all__ = ["TraceOutcome", "CallMetrics", "TraceMetrics", "call_metrics", "trace_metrics",
           "call_metrics_by_group", "trace_metrics_by_group"]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclasses.dataclass(frozen=True)
class TraceOutcome:

    trace_id: str
    suite: str
    succeeded_without_policy: bool
    succeeded: bool
    breached: bool | None = None
    halted_by_gateway: bool = False
    attacked: bool = False

    @property
    def provenance_group(self) -> str:
        return f"{self.suite}:{'attack' if self.attacked else 'utility'}"

    @property
    def counts(self) -> bool:
        return self.succeeded_without_policy


@dataclasses.dataclass
class CallMetrics:

    n: int
    n_unenforced: int
    rsr: float | None
    rsr_by_source: dict[str, float]
    counts_by_source: dict[str, int]
    dr: float | None = None
    review_calls: int = 0
    block_calls: int = 0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TraceMetrics:

    n: int
    n_excluded: int
    n_attacked: int
    utility: float | None
    ulr: float | None
    breach_rate: float | None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def call_metrics(records: Sequence[PolicyCallRecord]) -> CallMetrics:
    enforced = [r for r in records if r.is_tool]
    by_source = collections.Counter(r.denial_source.value for r in enforced)
    residual = sum(1 for r in enforced if r.is_enforced_residual)
    total = len(enforced)

    labelled = [r for r in enforced if r.gateway_label is not None]
    review = sum(1 for r in labelled if r.gateway_label == "REVIEW")
    block = sum(1 for r in labelled if r.gateway_label == "BLOCK")

    return CallMetrics(
        n=total,
        n_unenforced=len(records) - total,
        rsr=_rate(residual, total),
        rsr_by_source={s.value: (_rate(by_source.get(s.value, 0), total) or 0.0)
                       for s in DenialSource if s.is_residual},
        counts_by_source=dict(by_source),
        dr=_rate(review, len(labelled)) if labelled else None,
        review_calls=review,
        block_calls=block,
    )


def trace_metrics(outcomes: Iterable[TraceOutcome]) -> TraceMetrics:
    outcomes = list(outcomes)
    scored = [o for o in outcomes if o.counts]
    lost = sum(1 for o in scored if not o.succeeded)
    attacked = [o for o in outcomes if o.breached is not None]

    return TraceMetrics(
        n=len(scored),
        n_excluded=len(outcomes) - len(scored),
        n_attacked=len(attacked),
        utility=_rate(sum(1 for o in scored if o.succeeded), len(scored)),
        ulr=_rate(lost, len(scored)),
        breach_rate=_rate(sum(1 for o in attacked if o.breached), len(attacked)),
    )



def call_metrics_by_group(records: Sequence[PolicyCallRecord]) -> dict[str, CallMetrics]:
    groups: dict[str, list[PolicyCallRecord]] = collections.defaultdict(list)
    for record in records:
        groups[record.provenance_group].append(record)
    return {name: call_metrics(rows) for name, rows in sorted(groups.items())}


def trace_metrics_by_group(outcomes: Iterable[TraceOutcome]) -> dict[str, TraceMetrics]:
    groups: dict[str, list[TraceOutcome]] = collections.defaultdict(list)
    for outcome in outcomes:
        groups[outcome.provenance_group].append(outcome)
    return {name: trace_metrics(rows) for name, rows in sorted(groups.items())}
