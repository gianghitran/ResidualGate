
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..labels import Label
from .metrics import (TraceOutcome, call_metrics, call_metrics_by_group, trace_metrics,
                      trace_metrics_by_group)
from .records import PolicyCallRecord, write_jsonl

__all__ = ["SystemSpec", "SYSTEMS", "run_system", "main"]


@dataclasses.dataclass(frozen=True)
class SystemSpec:

    name: str
    kind: str
    summary: str
    review_execution: str = "halt"
    uses_model: bool = False

    @property
    def is_control(self) -> bool:
        return self.kind == "none"


SYSTEMS: tuple[SystemSpec, ...] = (
    SystemSpec("b1-none", "none", "no policy -- the 9.3 control and utility ceiling"),
    SystemSpec("b1-defer", "defer", "Denied -> REVIEW; the review floor"),
    SystemSpec("b1-refuse", "refuse", "Denied -> BLOCK; stock CaMeL"),
    SystemSpec("b2-model", "model_only", "SLM is the whole defence; no policy floor", uses_model=True),
    SystemSpec("b3a", "strict", "tighten from the floor", uses_model=True),
    SystemSpec("b3b", "bounded", "cross below the floor inside the envelope", uses_model=True),
    SystemSpec("b3c", "rescue", "recover refused work into the human queue", uses_model=True),
)

SYSTEMS_BY_NAME = {s.name: s for s in SYSTEMS}


def _build_engine(spec: SystemSpec, suite_engine, *, recorder, envelope_tools: Sequence[str],
                  model: Callable[..., Label] | None, tool_names: Sequence[str] = (),
                  consult: str = "residual", reviewer=None):
    from .engines import ComposedEngine, DeferEngine, RefuseEngine, envelope_from

    common = dict(recorder=recorder, review_execution=spec.review_execution,
                  reviewer=reviewer)
    if spec.kind == "defer":
        return DeferEngine(suite_engine, **common)
    if spec.kind == "refuse":
        return RefuseEngine(suite_engine, **common)
    if model is None:
        raise ValueError(f"system {spec.name} needs a model")
    if spec.kind == "bounded" and not envelope_tools:
        raise ValueError("system b3b crosses the soundness floor and needs "
                         "--envelope-tool; an empty envelope is silently inert")
    if spec.kind == "model_only":
        consult = "all"
    return ComposedEngine(
        suite_engine, model=model, mode=spec.kind, tool_names=tool_names, consult=consult,
        envelope=envelope_from(envelope_tools) if spec.kind == "bounded" else None, **common)


def run_system(spec: SystemSpec, trace, *, model=None, envelope_tools: Sequence[str] = (),
               strict: bool = False, gate_factory=None, consult: str = "residual") -> tuple[list[PolicyCallRecord], TraceOutcome]:
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.task_suite.task_suite import (functions_stack_trace_from_messages,
                                                 model_output_from_messages)
    from camel.interpreter import interpreter
    from camel.pipeline_elements.replay_privileged_llm import replay_task
    from camel.pipeline_elements.security_policies.agentdojo_security_policies import ADNoSecurityPolicyEngine
    from camel.security_policy import SecurityPolicyDeniedError

    from .engines import RecordingPolicyEngine
    from .probe import _load_engine_cls, _make_env

    pipeline, suite_name, user_task_id, attack_name, injection_task_id = trace
    suite = get_suite("v1.2", suite_name)
    runtime = FunctionsRuntime(suite.tools)

    user_task = suite.get_user_task_by_id(user_task_id)
    injection_task = (suite.get_injection_task_by_id(injection_task_id)
                      if injection_task_id is not None else None)

    env = _make_env(suite, pipeline, user_task_id, attack_name, injection_task_id)
    env = user_task.init_environment(env)
    pre_env = env.model_copy(deep=True)

    if gate_factory is not None and spec.uses_model:
        model = gate_factory(getattr(user_task, "PROMPT", None))

    trace_id = f"{pipeline}/{suite_name}/{user_task_id}/{attack_name or 'none'}/{injection_task_id or 'none'}"
    recorder = RecordingPolicyEngine(
        ADNoSecurityPolicyEngine(env) if spec.is_control else _load_engine_cls(suite_name)(env),
        trace_id=trace_id, suite=suite_name, user_task_id=user_task_id,
        injection_task_id=injection_task_id, attack_name=attack_name,
        tool_names=[t.name for t in suite.tools])

    engine = recorder if spec.is_control else _build_engine(
        spec, _load_engine_cls(suite_name)(env), recorder=recorder,
        envelope_tools=envelope_tools, model=model, consult=consult,
        tool_names=[t.name for t in suite.tools])

    mode = interpreter.MetadataEvalMode.STRICT if strict else interpreter.MetadataEvalMode.NORMAL
    messages: list = []
    try:
        messages, post_env = replay_task(pipeline, suite_name, user_task_id, injection_task_id,
                                         attack_name, interpreter.EvalArgs(engine, mode), env, runtime)
    except SecurityPolicyDeniedError:
        post_env = env
    halted = getattr(engine, "halted", False)

    stack_trace = functions_stack_trace_from_messages(messages)
    output = model_output_from_messages(messages) or []
    utility = suite._check_task_result(user_task, output, pre_env, post_env, stack_trace)
    breached = (suite._check_task_result(injection_task, output, pre_env, post_env, stack_trace)
                if injection_task is not None else None)

    outcome = TraceOutcome(
        trace_id=trace_id, suite=suite_name,
        succeeded_without_policy=True, succeeded=bool(utility),
        breached=None if breached is None else bool(breached),
        halted_by_gateway=halted, attacked=injection_task is not None)
    return [dataclasses.replace(r, system=spec.name) for r in recorder.records], outcome


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--system", action="append", choices=[s.name for s in SYSTEMS],
                    help="default: the three model-free systems")
    ap.add_argument("--suite", action="append")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--review-execution", choices=["halt", "proceed"], default="halt",
                    help="what a REVIEW does to the run. 'halt' is the pessimistic "
                         "bound (no review can cause a breach, every deferred task "
                         "fails); 'proceed' is the optimistic one (the reviewer "
                         "approves). Report both -- the gap is the value of review.")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--backend", default="openai-compat", help="openai-compat | vllm-offline | hf")
    ap.add_argument("--model", default="qwen2.5:7b-instruct", help="key passed to the backend")
    ap.add_argument("--model-timeout", type=float, default=None,
                    help="seconds to wait per model call. Raise it when the GPU "
                         "is shared: a timeout becomes a fail-closed verdict that "
                         "looks like a real one in the label counts")
    ap.add_argument("--consult", choices=["residual", "all"], default="residual",
                    help="where the model gets a say. 'residual' only where the "
                         "policy was undecided; 'all' on every call, which under "
                         "a sound policy can only lose utility")
    ap.add_argument("--envelope-tool", action="append", default=[],
                    help="tools B3b may loosen inside (7.4). No default: crossing "
                         "the soundness floor has to be declared, never inherited")
    args = ap.parse_args(argv)

    from .probe import discover
    from .slm import build_gate

    names = args.system or ["b1-none", "b1-defer", "b1-refuse"]
    print(f"review_execution={args.review_execution}")
    specs = [dataclasses.replace(SYSTEMS_BY_NAME[n], review_execution=args.review_execution)
             for n in names]
    traces = discover(args.logs)
    if args.suite:
        traces = [t for t in traces if t[1] in set(args.suite)]
    if not traces:
        print(f"no traces under {args.logs.resolve()}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    needs_model = any(s.uses_model for s in specs)
    gate_factory = None
    if needs_model:
        gates = []

        def gate_factory(goal):
            gate = build_gate(args.backend, args.model, goal=goal,
                              timeout=args.model_timeout)
            gates.append(gate)
            return gate
        print(f"backend={args.backend} model={args.model} consult={args.consult}")
    if not any(s.is_control for s in specs):
        print("WARNING: b1-none is not in this run, so the 9.3 gate excludes nothing "
              "and every ULR below is computed over traces that may never have been "
              "doable. Comparable within this run only.")
    control: dict[str, bool] = {}
    summary = {}
    for spec in sorted(specs, key=lambda s: not s.is_control):
        records: list[PolicyCallRecord] = []
        outcomes: list[TraceOutcome] = []
        for trace in traces:
            try:
                rec, out = run_system(spec, trace, strict=args.strict,
                                      gate_factory=gate_factory, consult=args.consult,
                                      envelope_tools=args.envelope_tool)
            except Exception as exc:
                print(f"  skip {trace[1]}/{trace[2]}: {type(exc).__name__}: {str(exc)[:90]}")
                continue
            if spec.is_control:
                control[out.trace_id] = out.succeeded
            else:
                out = dataclasses.replace(out, succeeded_without_policy=control.get(out.trace_id, True))
            records.extend(rec)
            outcomes.append(out)

        cm, tm = call_metrics(records), trace_metrics(outcomes)
        cg, tg = call_metrics_by_group(records), trace_metrics_by_group(outcomes)
        summary[spec.name] = {
            "calls": cm.as_dict(), "traces": tm.as_dict(),
            "by_provenance_group": {
                g: {"calls": cg[g].as_dict(), "traces": tg[g].as_dict()}
                for g in sorted(set(cg) | set(tg)) if g in cg and g in tg},
        }
        write_jsonl(records, args.out / f"replay_{spec.name}_{args.review_execution}.jsonl")
        def pct(x):
            return "--" if x is None else f"{x:.1%}"
        print(f"{spec.name:<10} calls={cm.n:<5} RSR={pct(cm.rsr):<7} DR={pct(cm.dr):<7}"
              f" traces={tm.n:<4} excl={tm.n_excluded:<3} utility={pct(tm.utility):<7}"
              f" ULR={pct(tm.ulr):<7} breach={pct(tm.breach_rate)} (n={tm.n_attacked})")
        for g in sorted(set(cg) & set(tg)):
            c, tt = cg[g], tg[g]
            print(f"    {g:<20} calls={c.n:<5} RSR={pct(c.rsr):<7} traces={tt.n:<4}"
                  f" utility={pct(tt.utility):<7} ULR={pct(tt.ulr):<7}"
                  f" breach={pct(tt.breach_rate)} (n={tt.n_attacked})")

    if needs_model:
        failed = sum(g.failures for g in gates)
        verdicts = [v for g in gates for v in g.verdicts]
        print(f"model calls: {len(verdicts)}, failures: {failed}")
        if verdicts:
            vpath = args.out / f"verdicts_{args.model.replace('/', '_').replace(':', '-')}.jsonl"
            vpath.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in verdicts))
            counts = collections.Counter(v["label"] for v in verdicts)
            print(f"  labels: {dict(counts)} -> {vpath.name}")
        if failed:
            errs = {g._last_error for g in gates if g._last_error}
            print(f"  WARNING: the model failed {failed} times -- "
                  f"results are not usable. First: {sorted(errs)[0]}")

    out_file = args.out / f"replay_summary_{args.review_execution}.json"
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"-> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
