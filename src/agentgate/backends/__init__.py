
from __future__ import annotations

from typing import Callable

from .base import Backend, BackendConfig, Generation

__all__ = ["Backend", "BackendConfig", "Generation",
           "build_backend", "available_backends"]


def _vllm_offline(config: BackendConfig, **kwargs):
    from .vllm_offline import VLLMOfflineBackend

    return VLLMOfflineBackend(config, **kwargs)


def _openai_compat(config: BackendConfig, **kwargs):
    from .openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(config, **kwargs)


def _hf(config: BackendConfig, **kwargs):
    from .hf import HFBackend

    return HFBackend(config, **kwargs)


_REGISTRY: dict[str, Callable[..., Backend]] = {
    "vllm-offline": _vllm_offline,
    "hf": _hf,
    "openai-compat": _openai_compat,
}


def build_backend(kind: str, config: BackendConfig, **kwargs) -> Backend:
    try:
        factory = _REGISTRY[kind]
    except KeyError:
        raise SystemExit(
            f"unknown backend {kind!r}; expected one of {sorted(_REGISTRY)}"
        ) from None
    return factory(config, **kwargs)


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
