
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Generation:
    text: str
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    score: float | None = None


@dataclass
class BackendConfig:
    model: str
    dtype: str = "float16"
    max_model_len: int = 4096
    max_tokens: int = 256
    batch_size: int = 8
    tensor_parallel_size: int = 1
    seed: int = 1234
    gpu_memory_utilization: float | None = None
    guided_json: dict | None = None
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def generate(self, prompts: list[str], config: BackendConfig) -> list[Generation]: ...
