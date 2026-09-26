
from __future__ import annotations


def ensure_model_names_registered() -> int:
    import camel.models
    from agentdojo.models import MODEL_NAMES

    return len(MODEL_NAMES)
