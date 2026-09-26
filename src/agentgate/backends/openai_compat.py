
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import BackendConfig, Generation

__all__ = ["OpenAICompatBackend"]

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OpenAICompatBackend:

    name = "openai-compat"

    def __init__(self, config: BackendConfig, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 180.0) -> None:
        self.base_url = (base_url or config.extra.get("base_url")
                         or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-needed"
        self.timeout = timeout

    def generate(self, prompts: list[str], config: BackendConfig) -> list[Generation]:
        return [self._one(prompt, config) for prompt in prompts]

    def _one(self, prompt: str, config: BackendConfig) -> Generation:
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": 0.0,
            "seed": config.seed,
        }
        if config.guided_json is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "decision",
                                                          "schema": config.guided_json}}

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{exc.code} from {self.base_url}: "
                               f"{exc.read()[:200].decode(errors='replace')}") from exc
        elapsed = (time.perf_counter() - started) * 1000

        text = body["choices"][0]["message"].get("content") or ""
        usage = body.get("usage") or {}
        return Generation(text=text, latency_ms=elapsed,
                          prompt_tokens=usage.get("prompt_tokens"))
