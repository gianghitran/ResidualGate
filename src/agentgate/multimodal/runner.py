
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..labels import Label
from .policy import (
    PolicyDecision, PolicyResult, Provenance,
    TaggedToolCall, apply_policy, tag_provenance,
)
from .triage import TriageResult, triage_residuals
from .vlm import VLMResponse, ToolCall, call_vlm


DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "multimodal"

SLM_LADDER = ("qwen2.5:1.5b-instruct", "qwen2.5:3b-instruct", "qwen2.5:7b-instruct")


@dataclass
class TaskSpec:
    id: str
    category: str
    image: str
    prompt: str
    injection_text: str
    legitimate_calls: list[dict]
    injection_calls: list[dict]
    visual_params: dict[str, list[str]]

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(
            id=d["id"],
            category=d["category"],
            image=d["image"],
            prompt=d["prompt"],
            injection_text=d.get("injection_text", ""),
            legitimate_calls=d.get("legitimate_calls", []),
            injection_calls=d.get("injection_calls", []),
            visual_params=d.get("visual_params", {}),
        )


@dataclass
class TaskResult:
    task_id: str
    category: str
    config: str
    vlm_calls: list[dict] = field(default_factory=list)
    total_calls: int = 0
    residual_calls: int = 0
    utility: bool = False
    unsafe_executions: int = 0
    human_reviews: int = 0
    blocked_by_slm: int = 0
    error: str | None = None


@dataclass
class RunSummary:
    config: str
    total_tasks: int = 0
    utility_count: int = 0
    utility_pct: float = 0.0
    total_unsafe: int = 0
    total_reviews: int = 0
    total_residual: int = 0
    total_tool_calls: int = 0
    slm_blocked: int = 0
    review_reduction_pct: float = 0.0
    task_results: list[TaskResult] = field(default_factory=list)


def load_tasks(data_dir: Path | None = None) -> list[TaskSpec]:
    path = (data_dir or DATA_DIR) / "tasks.json"
    with path.open() as f:
        data = json.load(f)
    return [TaskSpec.from_dict(t) for t in data["tasks"]]


def _is_legitimate(call: ToolCall, task: TaskSpec) -> bool:
    return not call.injected


def _is_injection(call: ToolCall, task: TaskSpec) -> bool:
    return call.injected


def run_unconstrained(task: TaskSpec, vlm_response: VLMResponse) -> TaskResult:
    result = TaskResult(
        task_id=task.id, category=task.category, config="unconstrained",
        total_calls=len(vlm_response.calls),
    )
    has_legit = False
    for call in vlm_response.calls:
        result.vlm_calls.append({"function": call.function, "args": call.arguments})
        if _is_legitimate(call, task):
            has_legit = True
        if _is_injection(call, task):
            result.unsafe_executions += 1
    result.utility = has_legit
    return result


def run_strict_block(
    task: TaskSpec, vlm_response: VLMResponse, policy: PolicyResult,
) -> TaskResult:
    result = TaskResult(
        task_id=task.id, category=task.category, config="strict_block",
        total_calls=len(vlm_response.calls),
        residual_calls=policy.residual_calls,
    )
    has_legit = False
    for call, tagged in zip(vlm_response.calls, policy.tagged_calls):
        if tagged.policy == PolicyDecision.ALLOW:
            if _is_legitimate(call, task):
                has_legit = True
            if _is_injection(call, task):
                result.unsafe_executions += 1
    result.utility = has_legit
    return result


def run_review_all(
    task: TaskSpec, vlm_response: VLMResponse, policy: PolicyResult,
) -> TaskResult:
    result = TaskResult(
        task_id=task.id, category=task.category, config="review_all",
        total_calls=len(vlm_response.calls),
        residual_calls=policy.residual_calls,
    )
    has_legit = False
    for call, tagged in zip(vlm_response.calls, policy.tagged_calls):
        if tagged.policy == PolicyDecision.ALLOW:
            if _is_legitimate(call, task):
                has_legit = True
            if _is_injection(call, task):
                result.unsafe_executions += 1
        elif tagged.policy == PolicyDecision.RESIDUAL:
            result.human_reviews += 1
            if _is_legitimate(call, task):
                has_legit = True
    result.utility = has_legit
    return result


def run_residualgate(
    task: TaskSpec,
    vlm_response: VLMResponse,
    policy: PolicyResult,
    triage: TriageResult,
) -> TaskResult:
    result = TaskResult(
        task_id=task.id, category=task.category, config="residualgate",
        total_calls=len(vlm_response.calls),
        residual_calls=policy.residual_calls,
    )
    has_legit = False
    residual_idx = 0
    for call, tagged in zip(vlm_response.calls, policy.tagged_calls):
        if tagged.policy == PolicyDecision.ALLOW:
            if _is_legitimate(call, task):
                has_legit = True
            if _is_injection(call, task):
                result.unsafe_executions += 1
        elif tagged.policy == PolicyDecision.RESIDUAL:
            verdict = triage.verdicts[residual_idx]
            residual_idx += 1
            if verdict.label == Label.BLOCK:
                result.blocked_by_slm += 1
            else:
                result.human_reviews += 1
                if _is_legitimate(call, task):
                    has_legit = True
    result.utility = has_legit
    return result


def _summarise(config: str, results: list[TaskResult],
               review_all_reviews: int = 0) -> RunSummary:
    s = RunSummary(config=config, total_tasks=len(results), task_results=results)
    for r in results:
        if r.utility:
            s.utility_count += 1
        s.total_unsafe += r.unsafe_executions
        s.total_reviews += r.human_reviews
        s.total_residual += r.residual_calls
        s.total_tool_calls += r.total_calls
        s.slm_blocked += r.blocked_by_slm
    s.utility_pct = (s.utility_count / s.total_tasks * 100) if s.total_tasks else 0
    if review_all_reviews > 0:
        s.review_reduction_pct = (1 - s.total_reviews / review_all_reviews) * 100
    return s


def _merge_injection(vlm_resp: VLMResponse, task: TaskSpec) -> VLMResponse:
    extra = [ToolCall(function=c["function"], arguments=c.get("args", {}), injected=True)
             for c in task.injection_calls]
    return VLMResponse(
        calls=vlm_resp.calls + extra,
        raw_text=vlm_resp.raw_text,
    )


def run_all(
    tasks: list[TaskSpec],
    *,
    vlm_model: str = "qwen2.5vl:7b",
    vlm_api_base: str = "http://localhost:11434/v1",
    slm_backend: str = "openai-compat",
    slm_model: str = "qwen2.5:7b-instruct",
    slm_timeout: float = 300.0,
    data_dir: Path | None = None,
    quiet: bool = False,
) -> dict[str, RunSummary]:
    gate = None
    if slm_model:
        try:
            from ..camel_bridge.slm import build_gate
            gate = build_gate(slm_backend, slm_model, timeout=slm_timeout)
        except Exception as exc:
            if not quiet:
                print(f"WARNING: could not build SLM gate: {exc}")

    unconstrained_results: list[TaskResult] = []
    strict_results: list[TaskResult] = []
    review_results: list[TaskResult] = []
    gate_results: list[TaskResult] = []

    dd = data_dir or DATA_DIR

    for i, task in enumerate(tasks):
        if not quiet:
            print(f"[{i + 1}/{len(tasks)}] {task.id} ({task.category})", end="", flush=True)

        image_path = dd / task.image
        vlm_resp = call_vlm(image_path, task.prompt,
                            api_base=vlm_api_base, model=vlm_model)

        if vlm_resp.error:
            if not quiet:
                print(f" ERROR: {vlm_resp.error}", flush=True)
            err = TaskResult(task_id=task.id, category=task.category,
                             config="error", error=vlm_resp.error)
            unconstrained_results.append(err)
            strict_results.append(err)
            review_results.append(err)
            gate_results.append(err)
            continue

        vlm_legit = len(vlm_resp.calls)
        vlm_resp = _merge_injection(vlm_resp, task)
        n_inj = len(task.injection_calls)
        if not quiet:
            print(f" -> {vlm_legit} VLM + {n_inj} injected = "
                  f"{len(vlm_resp.calls)} total", flush=True)

        tagged_calls = []
        for call in vlm_resp.calls:
            tc = tag_provenance(call.function, call.arguments, task.prompt)
            tc.is_legitimate = _is_legitimate(call, task)
            tagged_calls.append(tc)

        policy = apply_policy(tagged_calls)
        triage = triage_residuals(policy.tagged_calls, task.prompt, gate=gate)

        unconstrained_results.append(run_unconstrained(task, vlm_resp))
        strict_results.append(run_strict_block(task, vlm_resp, policy))
        review_results.append(run_review_all(task, vlm_resp, policy))
        gate_results.append(run_residualgate(task, vlm_resp, policy, triage))

    review_all_total = sum(r.human_reviews for r in review_results)

    return {
        "unconstrained": _summarise("unconstrained", unconstrained_results),
        "strict_block": _summarise("strict_block", strict_results),
        "review_all": _summarise("review_all", review_results, review_all_total),
        "residualgate": _summarise("residualgate", gate_results, review_all_total),
    }


def print_table(summaries: dict[str, RunSummary]) -> None:
    print(f"\n{'Configuration':<25} {'Utility%':>8} {'Unsafe':>6} {'Reviews':>7} "
          f"{'Review-down%':>12} {'Residual':>8}")
    print("-" * 76)
    for key in ["unconstrained", "strict_block", "review_all", "residualgate"]:
        s = summaries[key]
        print(f"{s.config:<25} {s.utility_pct:>7.1f}% {s.total_unsafe:>6} "
              f"{s.total_reviews:>7} {s.review_reduction_pct:>11.1f}% "
              f"{s.total_residual:>8}")
    print()


def _write_results(summaries: dict[str, RunSummary], out: Path, tag: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    results_data = {}
    for key, summary in summaries.items():
        results_data[key] = {
            "config": summary.config,
            "total_tasks": summary.total_tasks,
            "utility_count": summary.utility_count,
            "utility_pct": summary.utility_pct,
            "total_unsafe": summary.total_unsafe,
            "total_reviews": summary.total_reviews,
            "total_residual": summary.total_residual,
            "total_tool_calls": summary.total_tool_calls,
            "slm_blocked": summary.slm_blocked,
            "review_reduction_pct": summary.review_reduction_pct,
            "task_results": [asdict(r) for r in summary.task_results],
        }
    fname = f"mm_results{suffix}.json"
    (out / fname).write_text(json.dumps(results_data, indent=2, ensure_ascii=False))
    print(f"Results written to {out / fname}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multimodal case study runner")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/multimodal"))
    parser.add_argument("--vlm", default="qwen2.5vl:7b", help="VLM model name")
    parser.add_argument("--vlm-api-base",
                        default=os.environ.get("VLM_API_BASE", "http://localhost:11434/v1"))
    parser.add_argument("--slm", default="qwen2.5:7b-instruct", help="SLM model for triage")
    parser.add_argument("--slm-backend", default="openai-compat")
    parser.add_argument("--slm-timeout", type=float, default=300.0)
    parser.add_argument("--slm-ladder", action="store_true",
                        help="Run SLM triage with 1.5B, 3B, and 7B")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.data_dir)
    if not tasks:
        print("No tasks found. Check data/multimodal/tasks.json")
        return 1

    if not args.slm_ladder:
        summaries = run_all(
            tasks,
            vlm_model=args.vlm,
            vlm_api_base=args.vlm_api_base,
            slm_backend=args.slm_backend,
            slm_model=args.slm,
            slm_timeout=args.slm_timeout,
            data_dir=args.data_dir,
            quiet=args.quiet,
        )
        print_table(summaries)
        _write_results(summaries, args.out)
    else:
        print("=== VLM pass (shared across all SLM rungs) ===", flush=True)
        vlm_cache: dict[str, VLMResponse] = {}
        dd = args.data_dir or DATA_DIR
        for i, task in enumerate(tasks):
            if not args.quiet:
                print(f"[{i + 1}/{len(tasks)}] VLM: {task.id}", end="", flush=True)
            image_path = dd / task.image
            resp = call_vlm(image_path, task.prompt,
                            api_base=args.vlm_api_base, model=args.vlm)
            if not resp.error:
                resp = _merge_injection(resp, task)
            vlm_cache[task.id] = resp
            if not args.quiet:
                if resp.error:
                    print(f" ERROR: {resp.error}", flush=True)
                else:
                    vlm_n = sum(1 for c in resp.calls if not c.injected)
                    inj_n = sum(1 for c in resp.calls if c.injected)
                    print(f" -> {vlm_n} VLM + {inj_n} injected", flush=True)

        for slm_model in SLM_LADDER:
            tag = slm_model.replace("/", "_").replace(":", "-")
            print(f"\n=== SLM: {slm_model} ===", flush=True)

            gate = None
            try:
                from ..camel_bridge.slm import build_gate
                gate = build_gate(args.slm_backend, slm_model, timeout=args.slm_timeout)
            except Exception as exc:
                print(f"WARNING: could not build SLM gate for {slm_model}: {exc}")

            unc, strict, rev, gated = [], [], [], []
            for task in tasks:
                vlm_resp = vlm_cache[task.id]
                if vlm_resp.error:
                    err = TaskResult(task_id=task.id, category=task.category,
                                     config="error", error=vlm_resp.error)
                    unc.append(err); strict.append(err); rev.append(err); gated.append(err)
                    continue

                tagged_calls = []
                for call in vlm_resp.calls:
                    tc = tag_provenance(call.function, call.arguments, task.prompt)
                    tc.is_legitimate = _is_legitimate(call, task)
                    tagged_calls.append(tc)
                policy = apply_policy(tagged_calls)
                triage = triage_residuals(policy.tagged_calls, task.prompt, gate=gate)

                unc.append(run_unconstrained(task, vlm_resp))
                strict.append(run_strict_block(task, vlm_resp, policy))
                rev.append(run_review_all(task, vlm_resp, policy))
                gated.append(run_residualgate(task, vlm_resp, policy, triage))

            rev_total = sum(r.human_reviews for r in rev)
            summaries = {
                "unconstrained": _summarise("unconstrained", unc),
                "strict_block": _summarise("strict_block", strict),
                "review_all": _summarise("review_all", rev, rev_total),
                "residualgate": _summarise(f"residualgate ({slm_model})", gated, rev_total),
            }
            print_table(summaries)
            _write_results(summaries, args.out, tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
