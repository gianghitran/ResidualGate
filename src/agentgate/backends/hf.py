
from __future__ import annotations

import time

from .base import BackendConfig, Generation


class HFBackend:
    name = "hf"

    def __init__(self, config: BackendConfig, score_tokens: tuple[str, ...] = ()) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=dtype, device_map="auto")
        self.model.eval()
        self.torch = torch
        self.score_token_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0] for t in score_tokens
        ] if score_tokens else []

    def generate(self, prompts: list[str], config: BackendConfig) -> list[Generation]:
        results: list[Generation] = []
        for start in range(0, len(prompts), config.batch_size):
            chunk = prompts[start : start + config.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                                 max_length=config.max_model_len).to(self.model.device)
            t0 = time.perf_counter()
            with self.torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=config.max_tokens, do_sample=False,
                    return_dict_in_generate=True, output_scores=bool(self.score_token_ids),
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            elapsed = (time.perf_counter() - t0) * 1000 / len(chunk)
            texts = self.tokenizer.batch_decode(out.sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            scores = [None] * len(chunk)
            if self.score_token_ids and out.scores:
                probs = self.torch.softmax(out.scores[0].float(), dim=-1)
                scores = [float(probs[i, self.score_token_ids[0]]) for i in range(len(chunk))]
            for text, score in zip(texts, scores):
                results.append(Generation(text=text, latency_ms=elapsed, score=score))
        return results
