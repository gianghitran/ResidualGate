
from __future__ import annotations

import time

from .base import BackendConfig, Generation


class VLLMOfflineBackend:
    name = "vllm-offline"

    def __init__(self, config: BackendConfig) -> None:
        from vllm import LLM

        kwargs = dict(
            model=config.model,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            tensor_parallel_size=config.tensor_parallel_size,
            seed=config.seed,
        )
        if config.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization
        self.llm = LLM(**kwargs)
        self.structured_mode: str | None = None

    def _sampling_params(self, config: BackendConfig):
        from vllm import SamplingParams

        base = dict(temperature=0.0, max_tokens=config.max_tokens, seed=config.seed)
        if config.guided_json is None:
            return SamplingParams(**base)
        try:
            from vllm.sampling_params import StructuredOutputsParams

            self.structured_mode = "structured_outputs"
            return SamplingParams(**base, structured_outputs=StructuredOutputsParams(json=config.guided_json))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            self.structured_mode = "guided_decoding"
            return SamplingParams(**base, guided_decoding=GuidedDecodingParams(json=config.guided_json))

    def generate(self, prompts: list[str], config: BackendConfig) -> list[Generation]:
        params = self._sampling_params(config)
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, params)
        elapsed = (time.perf_counter() - start) * 1000 / max(len(prompts), 1)
        return [
            Generation(
                text=o.outputs[0].text,
                latency_ms=elapsed,
                prompt_tokens=len(o.prompt_token_ids) if o.prompt_token_ids else None,
            )
            for o in outputs
        ]
