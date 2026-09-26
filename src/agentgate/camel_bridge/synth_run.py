
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .records import DenialSource, PolicyCallRecord, write_jsonl
from .progress import Phase
from .synthesis import (_atoms, _norm, PRINT_FOR_SECURITY, PRINT_FOR_UTILITY, Synthesis, empty_like,
                        synthesise, synthesise_attack)

SUITES = ("banking", "slack", "travel", "workspace")
import os as _os
ATTACK_PIPELINE = _os.environ.get("AGENTGATE_ATTACK_PIPELINE", "gemma4-31b+camel")


def empty_value_for(suite):
    schemas: dict[str, dict] = {}
    for tool in suite.tools:
        try:
            schemas[tool.name] = tool.parameters.model_json_schema().get("properties", {})
        except Exception:
            schemas[tool.name] = {}

    def empty(function: str, name: str):
        return empty_like(schemas.get(function, {}).get(name))

    return empty


def dangling_placeholder_tasks(suite) -> frozenset[str]:
    flagged = set()
    for task_id, task in suite.injection_tasks.items():
        try:
            calls = task.ground_truth(suite.environment_type.model_construct())
        except Exception:
            continue
        for index, call in enumerate(calls):
            placeholders = [v for v in (call.placeholder_args or {}).values()
                            if isinstance(v, str) and v.startswith("$")]
            if placeholders and index == 0:
                flagged.add(task_id)
                break
    return frozenset(flagged)
_KEYERROR = re.compile(r"KeyError: '([^']+)'")

def _namespace(env, runtime):
    from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment
    from agentdojo.default_suites.v1.workspace.user_tasks import WorkspaceEnvironment
    from camel.interpreter.namespace import Namespace
    from camel.pipeline_elements.agentdojo_function import make_agentdojo_namespace

    ns = Namespace.with_builtins()
    drop = ({"datetime", "timedelta", "date", "time", "NaiveDatetime", "timezone"}
            if not isinstance(env, (WorkspaceEnvironment, TravelEnvironment)) else set())
    if not isinstance(env, WorkspaceEnvironment):
        ns = dataclasses.replace(
            ns, variables={k: v for k, v in ns.variables.items() if k not in drop})
    return ns.add_variables(make_agentdojo_namespace(ns, runtime, env))


def _result_text(res: Any) -> str:
    value = getattr(getattr(res, "value", None), "raw", None)
    return "" if value is None else str(value)


def _ground_truth_outputs(suite, task, env, runtime) -> tuple[list, list]:
    calls = task.ground_truth(env)
    outputs = []
    for call in calls:
        try:
            result, error = runtime.run_function(env, call.function, call.args, raise_on_error=False)
            outputs.append(result if error is None else "")
        except Exception:
            outputs.append("")
    return list(calls), outputs


@dataclasses.dataclass
class TaskResult:
    task_id: str
    suite: str
    records: list[PolicyCallRecord]
    synthesis: Synthesis | None
    utility: bool
    error: str | None = None
    utility_without_policy: bool | None = None

    @property
    def counts(self) -> bool:
        return bool(self.utility_without_policy)


def run_task(suite_name: str, task_id: str, *, max_retries: int = 3,
             policy: bool = True, print_mode: str = PRINT_FOR_UTILITY,
             system: str = "b1-refuse", gate_factory=None,
             review_execution: str = "halt",
             envelope_tools: "Sequence[str]" = ()) -> TaskResult:
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    from camel.interpreter import interpreter
    from camel.security_policy import SecurityPolicyDeniedError

    from agentdojo import types as ad_types
    from camel.pipeline_elements.privileged_llm import extract_print_output, make_ad_tool_calls
    from camel.pipeline_elements.security_policies.agentdojo_security_policies import (
        ADNoSecurityPolicyEngine)

    from .engines import RecordingPolicyEngine
    from .probe import _load_engine_cls

    suite = get_suite("v1.2", suite_name)
    task = suite.get_user_task_by_id(task_id)
    required: dict[str, list[str]] = collections.defaultdict(list)
    synthesis = None

    for _ in range(max_retries + 1):
        env = task.init_environment(suite.load_and_inject_default_environment({}))
        pre_env = env.model_copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)

        try:
            calls, outputs = _ground_truth_outputs(suite, task, env, runtime)
        except Exception as exc:
            return TaskResult(task_id, suite_name, [], None, False, f"ground_truth: {exc}")

        env = task.init_environment(suite.load_and_inject_default_environment({}))
        runtime = FunctionsRuntime(suite.tools)
        synthesis = synthesise(task.PROMPT, calls, outputs, required_args=required,
                               print_mode=print_mode, empty_value=empty_value_for(suite))

        from .replay import SYSTEMS_BY_NAME, _build_engine

        spec = dataclasses.replace(SYSTEMS_BY_NAME[system], review_execution=review_execution)
        tool_names = [t.name for t in suite.tools]
        bare = not policy or spec.is_control
        engine_cls = ADNoSecurityPolicyEngine if bare else _load_engine_cls(suite_name)
        recorder = RecordingPolicyEngine(
            engine_cls(env), trace_id=f"synth/{suite_name}/{task_id}",
            suite=suite_name, user_task_id=task_id, tool_names=tool_names)

        if bare:
            engine = recorder
        else:
            engine = _build_engine(
                spec, _load_engine_cls(suite_name)(env), recorder=recorder,
                envelope_tools=envelope_tools, tool_names=tool_names,
                reviewer=(lambda *_: True) if review_execution == "oracle" else None,
                model=gate_factory(task.PROMPT) if spec.uses_model else None)
        args = interpreter.EvalArgs(engine, interpreter.MetadataEvalMode.NORMAL)

        error, chain, res = None, (), None
        try:
            res, _, chain, _ = interpreter.parse_and_interpret_code(
                synthesis.fenced, _namespace(env, runtime), [], (), args)
            exc = getattr(res, "error", None)
            if exc is not None:
                error = f"{type(exc.exception).__name__}: {exc.exception}"
        except SecurityPolicyDeniedError:
            pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        missing = _KEYERROR.search(error or "")
        culprit = next((r.tool_name for r in reversed(recorder.records)
                        if r.denial_source is DenialSource.POLICY_ERROR), None)
        if missing and culprit and missing.group(1) not in required[culprit]:
            required[culprit].append(missing.group(1))
            continue

        printed = f"{extract_print_output(chain)}\n{_result_text(res)}".strip()
        stack_trace = [call for call, _ in make_ad_tool_calls(chain)]
        model_output = [ad_types.text_content_block_from_string(printed)] if printed else []
        utility = bool(suite._check_task_result(task, model_output, pre_env, env, stack_trace))
        return TaskResult(task_id, suite_name, recorder.records, synthesis, utility, error)

    return TaskResult(task_id, suite_name, [], synthesis, False, "gave up refilling arguments")


def _attack_main(args) -> int:
    from agentdojo.task_suite.load_suites import get_suite

    from .replay import SYSTEMS_BY_NAME
    from .slm import build_gate

    gate_factory, gates = None, []
    if any(SYSTEMS_BY_NAME[s].uses_model for s in args.system):
        def gate_factory(goal):
            gate = build_gate(args.backend, args.model, goal=goal,
                              timeout=args.model_timeout)
            gates.append(gate)
            return gate
        consult = ("all" if all(SYSTEMS_BY_NAME[s].kind == "model_only" for s in args.system)
                   else "residual" if not any(SYSTEMS_BY_NAME[s].kind == "model_only" for s in args.system)
                   else "mixed(model_only=all,others=residual)")
        print(f"backend={args.backend} model={args.model} consult={consult}")

    records: list[PolicyCallRecord] = []
    pair_rows: list[dict] = []
    rows: dict[str, dict] = {}
    for suite_name in (args.suite or list(SUITES)):
        suite = get_suite("v1.2", suite_name)
        dangling = dangling_placeholder_tasks(suite)
        if dangling:
            print(f"{suite_name}: dangling placeholder in {sorted(dangling)} "
                  f"-- reported separately, see dangling_placeholder_tasks")
        for mode in args.mode:
            stats = {s: dict(gated=0, blocked=0, util=0) for s in args.system}
            pairs = gated = 0
            total = len(suite.user_tasks) * len(suite.injection_tasks)
            phase = Phase(f"{suite_name}:{mode}", total, every=25, enabled=not args.quiet)
            for user_task_id in suite.user_tasks:
                if args.limit is not None and gated >= args.limit:
                    break
                for injection_task_id in suite.injection_tasks:
                    pairs += 1
                    control = run_attack(suite_name, user_task_id, injection_task_id,
                                         mode=mode, policy=False, print_mode=args.print_mode,
                                         attack_family=args.attack_family)
                    if not control.breached:
                        phase.step()
                        continue
                    gated += 1
                    if args.limit is not None and gated > args.limit:
                        gated -= 1
                        break
                    for system in args.system:
                        got = run_attack(suite_name, user_task_id, injection_task_id,
                                         mode=mode, policy=True, system=system,
                                         gate_factory=gate_factory, print_mode=args.print_mode,
                                         review_execution=args.review_execution,
                                         envelope_tools=args.envelope_tool or (),
                                         attack_family=args.attack_family)
                        got = dataclasses.replace(got, breached_without_policy=True)
                        records.extend(dataclasses.replace(r, system=system) for r in got.records)
                        stats[system]["gated"] += 1
                        stats[system]["blocked"] += not got.breached
                        stats[system]["util"] += got.utility
                        enforced = [r for r in got.records if r.is_tool]
                        decisive = enforced[-1] if enforced else None
                        pair_rows.append({
                            "suite": suite_name, "mode": mode, "system": system,
                            "user_task_id": user_task_id,
                            "injection_task_id": injection_task_id,
                            "breached": got.breached, "utility": got.utility,
                            "breached_without_policy": True,
                            "error": got.error,
                            "decisive_tool": decisive.tool_name if decisive else None,
                            "denial_source": str(decisive.denial_source) if decisive else None,
                            "payload_readers": decisive.payload_readers if decisive else None,
                            "dangling_placeholder": injection_task_id in dangling,
                        })
                    phase.step(works=1, blocked=int(not got.breached))
            phase.done()
            for system, s in stats.items():
                mine = [r for r in pair_rows if r["suite"] == suite_name
                        and r["mode"] == mode and r["system"] == system]
                rows[f"{suite_name}:{mode}:{system}"] = {
                    "pairs": pairs, "attacks_that_work": gated,
                    "blocked": s["blocked"], "utility_kept": s["util"],
                    "print_mode": args.print_mode,
                    "review_execution": args.review_execution,
                    "asr_no_policy": round(gated / pairs, 4) if pairs else None,
                    "asr": round((s["gated"] - s["blocked"]) / s["gated"], 4) if s["gated"] else None,
                    "utility_rate": round(s["util"] / s["gated"], 4) if s["gated"] else None,
                    **_strata(mine),
                }

    def pct(v):
        return "--" if v is None else f"{v:.1%}"

    print(f"{'suite:mode:system':<34}{'pairs':>7}{'works':>7}{'ASR none':>10}"
          f"{'ASR':>8}{'ASR pub':>9}{'ASR restr':>11}{'util kept':>11}")
    for name, s in rows.items():
        print(f"{name:<34}{s['pairs']:>7}{s['attacks_that_work']:>7}"
              f"{pct(s['asr_no_policy']):>10}{pct(s['asr']):>8}"
              f"{pct(s['asr_public_payload']):>9}{pct(s['asr_restricted_payload']):>11}"
              f"{pct(s['utility_rate']):>11}")
    print("\nASR pub / ASR restr split the same attacks by whether the decisive call's\n"
          "arguments were readable by anyone. CaMeL's write policies are confidentiality\n"
          "policies, so this is the variable they actually turn on -- an ASR that pools\n"
          "the two reports the sensitivity of the paired user tasks, not the defence.")
    excluded = sum(1 for r in pair_rows if r["dangling_placeholder"])
    if excluded:
        print(f"{excluded} pair-outcomes carry a dangling placeholder and are excluded from\n"
              "asr_excl_dangling; see synth_run.dangling_placeholder_tasks.")

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, args.out / "synth_attack_records.jsonl")
    write_jsonl(pair_rows, args.out / "synth_attack_pairs.jsonl")
    (args.out / "synth_attack_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    if gates:
        _flush_verdict_cache(gates)
    print(f"-> {args.out / 'synth_attack_summary.json'}")
    return 0


def _strata(pair_rows: list[dict]) -> dict:
    def rate(rows):
        return (round(sum(r["breached"] for r in rows) / len(rows), 4), len(rows)) if rows else (None, 0)

    public = [r for r in pair_rows if r["payload_readers"] == "public"]
    restricted = [r for r in pair_rows if r["payload_readers"] == "restricted"]
    kept = [r for r in pair_rows if not r["dangling_placeholder"]]
    asr_public, n_public = rate(public)
    asr_restricted, n_restricted = rate(restricted)
    asr_kept, n_kept = rate(kept)
    return {
        "asr_public_payload": asr_public, "n_public_payload": n_public,
        "asr_restricted_payload": asr_restricted, "n_restricted_payload": n_restricted,
        "asr_excl_dangling": asr_kept, "n_excl_dangling": n_kept,
    }



def _oracle_reviewer(attacker_calls):
    poisoned = set()
    for call in attacker_calls:
        for value in getattr(call, "args", {}).values():
            for atom in _atoms(value):
                text = _norm(atom)
                if len(text) >= 4:
                    poisoned.add(text)

    def review(tool_name, kwargs):
        for value in kwargs.values():
            for atom in _atoms(getattr(value, "raw", value)):
                if _norm(atom) in poisoned:
                    return False
        return True

    return review


def _flush_verdict_cache(gates) -> None:
    from .slm import CachingGate

    cached = [g for g in gates if isinstance(g, CachingGate)]
    if not cached:
        return
    hits = sum(g.hits for g in cached)
    misses = sum(g.misses for g in cached)
    CachingGate.flush()
    if hits + misses == 0:
        print("verdict cache: no model calls")
        return
    print(f"verdict cache: {hits} hit, {misses} miss "
          f"({hits / (hits + misses) * 100:.0f}% reused)")



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", action="append", choices=SUITES)
    ap.add_argument("--attack", action="store_true", help="run the security half")
    ap.add_argument("--attack-family", default="important_instructions",
                    help="AgentDojo attack family for injection text "
                         "(default: important_instructions, the only one this "
                         "study evaluates; others are a threat to external validity)")
    ap.add_argument("--mode", action="append", choices=["chain", "substitute"],
                    help="attack shape; default: both")
    ap.add_argument("--system", action="append",
                    help="rows of replay.SYSTEMS; default: b1-refuse")
    ap.add_argument("--backend", default="openai-compat")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--model-timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the pairs that pass the gate per suite/mode. A "
                         "sample, declared as one: the full matrix is 629 pairs "
                         "and a model-backed system pays for every residual call")
    ap.add_argument("--print-mode", choices=["all", "last", "none"], default=None,
                    help="what the synthesised program says out loud. This is "
                         "graded, not logged. Default: 'all' for utility runs, "
                         "'last' for security runs -- the two pull opposite ways")
    ap.add_argument("--envelope-tool", action="append", default=None,
                    help="B3b only: a tool allowed to cross the soundness floor "
                         "(REVIEW->ALLOW) inside the declared risk envelope. "
                         "Repeatable. An empty envelope makes b3b silently inert, "
                         "so _build_engine refuses it.")
    ap.add_argument("--review-execution", default="halt", choices=["halt", "proceed", "oracle"],
                    help="what a REVIEW does. 'halt' for security numbers (no "
                         "REVIEW can cause a breach); 'proceed' for review-load "
                         "and utility numbers. Belongs in every results table")
    ap.add_argument("--quiet", action="store_true", help="no phase reporting on stderr")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    args.mode = args.mode or ["chain", "substitute"]
    args.system = args.system or ["b1-refuse"]
    if args.print_mode is None:
        args.print_mode = PRINT_FOR_SECURITY if args.attack else PRINT_FOR_UTILITY
    if args.attack:
        return _attack_main(args)

    from agentdojo.task_suite.load_suites import get_suite

    from .replay import SYSTEMS_BY_NAME
    from .slm import build_gate

    gate_factory, gates = None, []
    if any(SYSTEMS_BY_NAME[n].uses_model for n in args.system):
        def gate_factory(goal):
            gate = build_gate(args.backend, args.model, goal=goal,
                              timeout=args.model_timeout)
            gates.append(gate)
            return gate
        print(f"backend={args.backend} model={args.model} "
              f"review_execution={args.review_execution}")

    records: list[PolicyCallRecord] = []
    task_rows: list[dict] = []
    origins: collections.Counter = collections.Counter()
    filled: collections.Counter = collections.Counter()
    failures: list[tuple[str, str]] = []
    per_suite: dict[str, dict] = {}

    for suite_name in (args.suite or list(SUITES)):
        suite = get_suite("v1.2", suite_name)
        doable_ids, controls = [], {}
        phase = Phase(f"{suite_name}:control", len(suite.user_tasks), enabled=not args.quiet)
        for task_id in suite.user_tasks:
            control = run_task(suite_name, task_id, policy=False, print_mode=args.print_mode)
            controls[task_id] = control.utility
            if control.utility:
                doable_ids.append(task_id)
            phase.step(doable=int(control.utility))
        phase.done()

        for system in args.system:
            rows, ok, lost = [], 0, 0
            phase = Phase(f"{suite_name}:{system}", len(doable_ids), enabled=not args.quiet)
            for task_id in doable_ids:
                result = run_task(suite_name, task_id, print_mode=args.print_mode,
                                  system=system, gate_factory=gate_factory,
                                  review_execution=args.review_execution,
                                  envelope_tools=args.envelope_tool or ())
                result = dataclasses.replace(result, utility_without_policy=controls[task_id])
                if result.error and not result.records:
                    failures.append((f"{suite_name}/{task_id}/{system}", result.error[:90]))
                rows.extend(dataclasses.replace(r, system=system) for r in result.records)
                ok += result.utility
                lost += not result.utility
                task_rows.append({
                    "suite": suite_name, "user_task_id": task_id, "system": system,
                    "utility": bool(result.utility),
                    "doable_without_policy": bool(controls[task_id]),
                    "review_execution": args.review_execution,
                    "error": result.error,
                })
                if result.synthesis:
                    origins.update(result.synthesis.origins)
                    filled.update(result.synthesis.filled)
                phase.step(done=int(result.utility),
                           calls=sum(1 for r in result.records if r.is_tool))
            phase.done()
            enforced = [r for r in rows if r.is_tool]
            residual = sum(1 for r in enforced if r.is_enforced_residual)
            errors = sum(1 for r in enforced if r.denial_source is DenialSource.POLICY_ERROR)
            review = sum(1 for r in enforced if r.gateway_label == "REVIEW")
            doable = len(doable_ids)
            per_suite[f"{suite_name}:{system}"] = {
                "suite": suite_name, "system": system,
                "tasks": len(suite.user_tasks), "doable_without_policy": doable,
                "utility": ok, "ulr": round(lost / doable, 4) if doable else None,
                "calls": len(enforced),
                "residual": residual, "policy_error": errors, "review_calls": review,
                "dr": round(review / len(enforced), 4) if enforced else None,
                "print_mode": args.print_mode,
                "review_execution": args.review_execution,
                "model": args.model if gate_factory else None,
                "rsr": round(residual / len(enforced), 4) if enforced else None,
                "by_source": dict(collections.Counter(r.denial_source.value for r in enforced)),
            }
            records.extend(rows)

    print(f"{'suite:system':<24}{'tasks':>6}{'doable':>8}{'done':>6}{'ULR':>8}"
          f"{'calls':>7}{'resid':>7}{'RSR':>8}{'DR':>7}{'pol_err':>9}")
    for name, s in per_suite.items():
        f = lambda k: "--" if s[k] is None else f"{s[k]:.1%}"
        print(f"{name:<24}{s['tasks']:>6}{s['doable_without_policy']:>8}{s['utility']:>6}"
              f"{f('ulr'):>8}{s['calls']:>7}{s['residual']:>7}{f('rsr'):>8}"
              f"{f('dr'):>7}{s['policy_error']:>9}")
    print(f"\nargument origins: {dict(origins)}")
    if filled:
        print(f"arguments the policy demanded but the task omits: {dict(filled)}")
    for name, err in failures[:8]:
        print(f"  FAILED {name}: {err}")
    if failures:
        print(f"  ({len(failures)} tasks produced no records)")

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, args.out / "synth_records.jsonl")
    write_jsonl(task_rows, args.out / "synth_tasks.jsonl")
    (args.out / "synth_summary.json").write_text(json.dumps(per_suite, indent=2, sort_keys=True))
    if gates:
        verdicts = [v for g in gates for v in g.verdicts]
        failed = sum(g.failures for g in gates)
        print(f"model calls: {len(verdicts)}, failures: {failed}")
        _flush_verdict_cache(gates)
        if verdicts:
            slug = args.model.replace("/", "_").replace(":", "-")
            vpath = args.out / f"synth_verdicts_{slug}.jsonl"
            vpath.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in verdicts))
            print(f"  labels: {dict(collections.Counter(v['label'] for v in verdicts))} "
                  f"-> {vpath.name}")
    print(f"-> {args.out / 'synth_summary.json'}")
    return 0





@dataclasses.dataclass
class AttackResult:
    user_task_id: str
    injection_task_id: str
    suite: str
    mode: str
    records: list[PolicyCallRecord]
    breached: bool
    utility: bool
    error: str | None = None
    breached_without_policy: bool | None = None

    @property
    def counts(self) -> bool:
        return bool(self.breached_without_policy)


def run_attack(suite_name: str, user_task_id: str, injection_task_id: str, *,
               mode: str = "chain", policy: bool = True, max_retries: int = 3,
               system: str = "b1-refuse", gate_factory=None, strict: bool = False,
               print_mode: str = PRINT_FOR_SECURITY,
               review_execution: str = "halt",
               envelope_tools: "Sequence[str]" = (),
               attack_family: str = "important_instructions") -> AttackResult:
    from agentdojo import attacks, types as ad_types
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    from camel.interpreter import interpreter
    from camel.pipeline_elements.privileged_llm import extract_print_output, make_ad_tool_calls
    from camel.pipeline_elements.security_policies.agentdojo_security_policies import (
        ADNoSecurityPolicyEngine)
    from camel.security_policy import SecurityPolicyDeniedError

    from .camel_models import ensure_model_names_registered
    from .engines import RecordingPolicyEngine
    from .probe import _load_engine_cls

    ensure_model_names_registered()
    suite = get_suite("v1.2", suite_name)
    user_task = suite.get_user_task_by_id(user_task_id)
    injection_task = suite.get_injection_task_by_id(injection_task_id)

    dummy = AgentPipeline([]); dummy.name = ATTACK_PIPELINE
    attack = attacks.load_attack(attack_family, suite, dummy)
    injections = attack.attack(user_task, injection_task)

    required: dict[str, list[str]] = collections.defaultdict(list)
    for _ in range(max_retries + 1):
        env = user_task.init_environment(suite.load_and_inject_default_environment(injections))
        runtime = FunctionsRuntime(suite.tools)
        try:
            user_calls, outputs = _ground_truth_outputs(suite, user_task, env, runtime)
            attacker_calls = list(injection_task.ground_truth(env))
        except Exception as exc:
            return AttackResult(user_task_id, injection_task_id, suite_name, mode, [],
                                False, False, f"ground_truth: {exc}")

        env = user_task.init_environment(suite.load_and_inject_default_environment(injections))
        pre_env = env.model_copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)
        synthesis = synthesise_attack(user_task.PROMPT, user_calls, outputs, attacker_calls,
                                      mode=mode, required_args=required, print_mode=print_mode,
                                      empty_value=empty_value_for(suite))

        from .replay import SYSTEMS_BY_NAME, _build_engine

        spec = dataclasses.replace(SYSTEMS_BY_NAME[system], review_execution=review_execution)
        tool_names = [tool.name for tool in suite.tools]
        bare = not policy or spec.is_control
        recorder = RecordingPolicyEngine(
            (ADNoSecurityPolicyEngine if bare else _load_engine_cls(suite_name))(env),
            trace_id=f"synth/{suite_name}/{user_task_id}/{mode}/{injection_task_id}",
            suite=suite_name, user_task_id=user_task_id, injection_task_id=injection_task_id,
            attack_name="synthesised", tool_names=tool_names)

        if bare:
            engine = recorder
        else:
            engine = _build_engine(
                spec, _load_engine_cls(suite_name)(env), recorder=recorder,
                envelope_tools=envelope_tools, tool_names=tool_names,
                reviewer=(_oracle_reviewer(attacker_calls)
                          if review_execution == "oracle" else None),
                model=gate_factory(user_task.PROMPT) if spec.uses_model else None)
        mode_ = (interpreter.MetadataEvalMode.STRICT if strict
                 else interpreter.MetadataEvalMode.NORMAL)
        args = interpreter.EvalArgs(engine, mode_)

        error, chain, res = None, (), None
        try:
            res, _, chain, _ = interpreter.parse_and_interpret_code(
                synthesis.fenced, _namespace(env, runtime), [], (), args)
            exc = getattr(res, "error", None)
            if exc is not None:
                error = f"{type(exc.exception).__name__}: {exc.exception}"
        except SecurityPolicyDeniedError:
            pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        missing = _KEYERROR.search(error or "")
        culprit = next((r.tool_name for r in reversed(recorder.records)
                        if r.denial_source is DenialSource.POLICY_ERROR), None)
        if missing and culprit and missing.group(1) not in required[culprit]:
            required[culprit].append(missing.group(1))
            continue

        printed = f"{extract_print_output(chain)}\n{_result_text(res)}".strip()
        stack = [call for call, _ in make_ad_tool_calls(chain)]
        output = [ad_types.text_content_block_from_string(printed)] if printed else []
        breached = bool(suite._check_task_result(injection_task, output, pre_env, env, stack))
        utility = bool(suite._check_task_result(user_task, output, pre_env, env, stack))
        return AttackResult(user_task_id, injection_task_id, suite_name, mode,
                            recorder.records, breached, utility, error)

    return AttackResult(user_task_id, injection_task_id, suite_name, mode, [], False, False,
                        "gave up refilling arguments")


if __name__ == "__main__":
    raise SystemExit(main())
