
from __future__ import annotations

from .labels import Disposition, Label, PolicyLabel

__all__ = ["Label", "Disposition", "tighten", "rescue", "release_bounded", "tighten_bounded"]


def tighten(policy: Label, slm: Label) -> Label:
    return Label(max(policy, slm))


def rescue(defer_floor: Label, refuse_label: Label, slm: Label) -> Label:
    candidate = defer_floor if slm is Label.ALLOW else refuse_label
    return Label(max(defer_floor, candidate))


def release_bounded(policy: Label, slm: Label, inside_envelope: bool) -> Label:
    if policy == Label.REVIEW and slm == Label.ALLOW and inside_envelope:
        return Label.ALLOW
    return Label(max(policy, slm))


tighten_bounded = release_bounded


def dispose(policy: PolicyLabel, disposition: Disposition) -> Label:
    return policy.to_label(disposition)
