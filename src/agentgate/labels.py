
from __future__ import annotations

from enum import IntEnum
from typing import Literal


class Label(IntEnum):

    ALLOW = 0
    REVIEW = 1
    BLOCK = 2

    @classmethod
    def parse(cls, value: str | int | "Label") -> "Label":
        if isinstance(value, Label):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]


Disposition = Literal["defer", "refuse"]

DISPOSITIONS: tuple[Disposition, ...] = ("defer", "refuse")


class PolicyLabel(IntEnum):

    ALLOW = 0
    UNDECIDED = 1
    BLOCK = 2

    @classmethod
    def parse(cls, value: str | int | "PolicyLabel") -> "PolicyLabel":
        if isinstance(value, PolicyLabel):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]

    def to_label(self, disposition: Disposition = "defer") -> Label:
        if self is PolicyLabel.ALLOW:
            return Label.ALLOW
        if self is PolicyLabel.BLOCK:
            return Label.BLOCK
        return Label.BLOCK if disposition == "refuse" else Label.REVIEW


LABELS: tuple[Label, ...] = (Label.ALLOW, Label.REVIEW, Label.BLOCK)
