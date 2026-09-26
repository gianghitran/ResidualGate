
from __future__ import annotations

import dataclasses
import hashlib
import json
import re

from ..labels import Label

__all__ = ["DECISION_SCHEMA", "SCHEMA_HASH", "FAIL_CLOSED", "ModelVerdict", "parse"]

DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["ALLOW", "REVIEW", "BLOCK"]},
        "evidence_field": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "evidence_field", "rationale"],
}

SCHEMA_HASH = hashlib.sha256(json.dumps(DECISION_SCHEMA, sort_keys=True).encode()).hexdigest()[:16]

FAIL_CLOSED = Label.REVIEW

_JSON_RE = re.compile(r"\{.*\}", re.S)
_BARE_LABEL_RE = re.compile(r"\b(ALLOW|REVIEW|BLOCK)\b")


@dataclasses.dataclass(frozen=True)
class ModelVerdict:

    label: Label
    status: str
    evidence_field: str | None = None
    rationale: str | None = None



def parse(text: str) -> ModelVerdict:
    text = text or ""
    match = _JSON_RE.search(text)
    if match:
        try:
            payload = json.loads(match.group(0))
            return ModelVerdict(Label.parse(str(payload["label"])), "ok",
                                payload.get("evidence_field"), payload.get("rationale"))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    bare = _BARE_LABEL_RE.search(text.upper())
    if bare:
        return ModelVerdict(Label.parse(bare.group(1)), "ok")
    return ModelVerdict(FAIL_CLOSED, "failed")
