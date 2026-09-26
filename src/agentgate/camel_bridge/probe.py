
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from .records import DenialSource, PolicyCallRecord, write_jsonl

ENGINES = {
    "banking": ("camel.pipeline_elements.security_policies.banking", "BankingSecurityPolicyEngine"),
    "slack": ("camel.pipeline_elements.security_policies.slack", "SlackSecurityPolicyEngine"),
    "travel": ("camel.pipeline_elements.security_policies.travel", "TravelSecurityPolicyEngine"),
    "workspace": ("camel.pipeline_elements.security_policies.workspace", "WorkspaceSecurityPolicyEngine"),
}


def _load_engine_cls(suite: str):
    module_name, cls_name = ENGINES[suite]
    import importlib

    return getattr(importlib.import_module(module_name), cls_name)


def discover(logs: Path) -> list[tuple[str, str, str, str | None, str | None]]:
    found = []
    for trace in sorted(logs.glob("*/*/user_task_*/*/*.json")):
        injection, attack, user_task, suite = (
            trace.stem, trace.parent.name, trace.parent.parent.name, trace.parent.parent.parent.name)
        pipeline = trace.parent.parent.parent.parent.name
        found.append((pipeline, suite, user_task,
                      None if attack == "none" else attack,
                      None if injection == "none" else injection))
    return found


def _make_env(suite, pipeline: str, user_task_id: str, attack_name: str | None,
              injection_task_id: str | None):
    from agentdojo import attacks
    from agentdojo.agent_pipeline import AgentPipeline

    from .camel_models import ensure_model_names_registered

    if attack_name is None or injection_task_id is None:
        return suite.load_and_inject_default_environment({})
    ensure_model_names_registered()
    dummy = AgentPipeline([])
    dummy.name = pipeline
    attack = attacks.load_attack(attack_name, suite, dummy)
    injections = attack.attack(suite.get_user_task_by_id(user_task_id),
                               suite.get_injection_task_by_id(injection_task_id))
    return suite.load_and_inject_default_environment(injections)


def replay_one(pipeline: str, suite_name: str, user_task_id: str, attack_name: str | None,
               injection_task_id: str | None, *, strict: bool = False) -> list[PolicyCallRecord]:
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    from camel.interpreter import interpreter
    from camel.pipeline_elements.replay_privileged_llm import replay_task
    from camel.security_policy import SecurityPolicyDeniedError

    from .engines import RecordingPolicyEngine

    suite = get_suite("v1.2", suite_name)
    runtime = FunctionsRuntime(suite.tools)

    env = _make_env(suite, pipeline, user_task_id, attack_name, injection_task_id)

    trace_id = f"{pipeline}/{suite_name}/{user_task_id}/{attack_name or 'none'}/{injection_task_id or 'none'}"
    recorder = RecordingPolicyEngine(
        _load_engine_cls(suite_name)(env), trace_id=trace_id, suite=suite_name,
        user_task_id=user_task_id, injection_task_id=injection_task_id, attack_name=attack_name,
        tool_names=[t.name for t in suite.tools])

    mode = interpreter.MetadataEvalMode.STRICT if strict else interpreter.MetadataEvalMode.NORMAL
    try:
        replay_task(pipeline, suite_name, user_task_id, injection_task_id, attack_name,
                    interpreter.EvalArgs(recorder, mode), env, runtime)
    except SecurityPolicyDeniedError:
        pass
    return recorder.records


def summarise(records: list[PolicyCallRecord]) -> dict[str, object]:
    tools = [r for r in records if r.is_tool]
    by_source = collections.Counter(r.denial_source.value for r in tools)
    by_tool = collections.Counter(r.tool_name for r in tools if r.denied)
    residual = sum(1 for r in tools if r.is_enforced_residual)
    total = len(tools)
    return {
        "calls": total,
        "unenforced_calls": len(records) - total,
        "residual": residual,
        "rsr": round(residual / total, 4) if total else 0.0,
        "by_source": dict(by_source),
        "top_denied_tools": dict(by_tool.most_common(10)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--suite", action="append", help="restrict to these suites")
    ap.add_argument("--strict", action="store_true", help="MetadataEvalMode.STRICT")
    ap.add_argument("--out", type=Path, help="write per-call records as JSONL")
    args = ap.parse_args(argv)

    traces = discover(args.logs)
    if args.suite:
        traces = [t for t in traces if t[1] in set(args.suite)]
    if not traces:
        print(f"no traces under {args.logs.resolve()} -- see proposal 9.2", file=sys.stderr)
        return 1

    records: list[PolicyCallRecord] = []
    failed = []
    for pipeline, suite, user_task, attack, injection in traces:
        try:
            records.extend(replay_one(pipeline, suite, user_task, attack, injection, strict=args.strict))
        except Exception as exc:
            failed.append((f"{pipeline}/{suite}/{user_task}", type(exc).__name__, str(exc)[:120]))

    summary = summarise(records)
    print(f"traces replayed : {len(traces) - len(failed)}/{len(traces)}")
    print(f"check_policy    : {summary['calls']} enforced tool calls "
          f"(+{summary['unenforced_calls']} builtin calls consulted but discarded)")
    print(f"residual (RSR)  : {summary['residual']}  ({summary['rsr']:.1%})")
    print("by denial_source:")
    for source in DenialSource:
        count = summary["by_source"].get(source.value, 0)
        flag = {"policy_residual": "RESIDUAL", "policy_error": "POLICY ERROR"}.get(
            source.partition, "decidable")
        print(f"   {source.value:<18} {count:>5}   {flag}")
    if summary["top_denied_tools"]:
        print("denied by tool  :", summary["top_denied_tools"])
    for name, kind, msg in failed:
        print(f"  FAILED {name}: {kind}: {msg}", file=sys.stderr)

    if args.out:
        n = write_jsonl(records, args.out)
        print(f"wrote {n} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
