
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def wilson(successes: int, n: int, alpha: float = 0.05) -> Interval | None:
    if n == 0:
        return None
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool], continuity: bool = True) -> tuple[float, float, int, int]:
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    if b + c == 0:
        return 0.0, 1.0, b, c
    if b + c < 25:
        p = float(stats.binomtest(b, b + c, 0.5).pvalue)
        return float(abs(b - c)), p, b, c
    correction = 1 if continuity else 0
    statistic = (abs(b - c) - correction) ** 2 / (b + c)
    return float(statistic), float(stats.chi2.sf(statistic, 1)), b, c


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    for rank, index in enumerate(order):
        if pvalues[index] <= alpha / (m - rank):
            reject[index] = True
        else:
            break
    return reject


def bootstrap_cluster(
    values: Sequence[float],
    clusters: Sequence[str],
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> Interval | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    by_cluster: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        by_cluster.setdefault(cluster, []).append(value)
    keys = list(by_cluster)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [v for j in picked for v in by_cluster[keys[j]]]
        draws[i] = statistic(sample)
    return Interval(
        float(statistic(list(values))),
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def paired_difference_ci(
    a_values: Sequence[float],
    b_values: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> Interval | None:
    if not a_values:
        return None
    diffs = [a - b for a, b in zip(a_values, b_values)]
    return bootstrap_cluster(diffs, clusters, n_boot=n_boot, alpha=alpha, seed=seed)


def tost_equivalence(interval: Interval, margin: float) -> bool:
    return interval.low > -margin and interval.high < margin


@dataclass
class LadderPoint:
    name: str
    params: float
    correct: list[bool]
    metric: list[float]
    clusters: list[str]


@dataclass
class SaturationResult:
    naive_floor: str | None
    equivalence_floor: str | None
    margin: float
    detail: list[dict]


def saturation_point(points: Sequence[LadderPoint], margin: float = 0.05, alpha: float = 0.05) -> SaturationResult:
    ordered = sorted(points, key=lambda p: p.params)
    detail: list[dict] = []
    pvalues: list[float] = []

    for previous, current in zip(ordered, ordered[1:]):
        _, p, b, c = mcnemar(current.correct, previous.correct)
        pvalues.append(p)
        detail.append({"step": f"{previous.name}->{current.name}", "p_mcnemar": p, "b": b, "c": c})
    rejected = holm_bonferroni(pvalues) if pvalues else []
    for entry, flag in zip(detail, rejected):
        entry["significant_holm"] = flag

    naive = next((d["step"].split("->")[0] for d, flag in zip(detail, rejected) if not flag), None)

    top = ordered[-1]
    equivalence_floor = None
    for point in ordered:
        interval = paired_difference_ci(point.metric, top.metric, point.clusters, alpha=alpha)
        equivalent = interval is not None and tost_equivalence(interval, margin)
        detail.append({
            "point": point.name,
            "vs_top": top.name,
            "diff_ci": str(interval) if interval else None,
            "equivalent_within_margin": equivalent,
        })
        if equivalent and equivalence_floor is None:
            equivalence_floor = point.name

    return SaturationResult(naive, equivalence_floor, margin, detail)
