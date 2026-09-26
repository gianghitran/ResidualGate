
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stats import bootstrap_cluster, mcnemar, wilson

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"


def _load(path: Path) -> Any | None:
    return json.loads(path.read_text()) if path.exists() else None


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _rate_ci(k: int, n: int) -> dict | None:
    iv = wilson(k, n)
    if iv is None:
        return None
    return {"k": k, "n": n, "point": round(iv.point, 4),
            "low": round(iv.low, 4), "high": round(iv.high, 4)}


def _attack_block(summary_path: Path, pairs_path: Path, label: str) -> dict:
    summary = _load(summary_path) or {}
    pairs = _jsonl(pairs_path)
    out: dict[str, Any] = {"regime": label, "rows": {}, "aggregate": {}, "mcnemar": {}}

    for name, v in summary.items():
        gated = v.get("attacks_that_work") or 0
        leaks = round((v.get("asr") or 0) * gated)
        out["rows"][name] = {
            "asr": _rate_ci(leaks, gated),
            "asr_public": _rate_ci(round((v.get("asr_public_payload") or 0)
                                         * (v.get("n_public_payload") or 0)),
                                   v.get("n_public_payload") or 0),
            "asr_restricted": _rate_ci(round((v.get("asr_restricted_payload") or 0)
                                             * (v.get("n_restricted_payload") or 0)),
                                       v.get("n_restricted_payload") or 0),
        }

    by_system: dict[str, list[dict]] = {}
    for r in pairs:
        by_system.setdefault(r["system"], []).append(r)
    for system, rows in sorted(by_system.items()):
        tier = {t: [r for r in rows if r.get("payload_readers") == t]
                for t in ("public", "restricted")}
        boot = bootstrap_cluster([float(r["breached"]) for r in rows],
                                 [r["suite"] for r in rows])
        out["aggregate"][system] = {
            "asr": _rate_ci(sum(r["breached"] for r in rows), len(rows)),
            "asr_cluster_boot": None if boot is None else
                {"point": round(boot.point, 4), "low": round(boot.low, 4),
                 "high": round(boot.high, 4)},
            "asr_public": _rate_ci(sum(r["breached"] for r in tier["public"]),
                                   len(tier["public"])),
            "asr_restricted": _rate_ci(sum(r["breached"] for r in tier["restricted"]),
                                       len(tier["restricted"])),
        }

    def keyed(system: str, field: str) -> dict:
        return {(r["suite"], r["mode"], r["user_task_id"], r["injection_task_id"]): r[field]
                for r in pairs if r["system"] == system}

    base = keyed("b1-refuse", "breached")
    for system in ("b3a", "b3c"):
        other = keyed(system, "breached")
        shared = sorted(base.keys() & other.keys())
        if not shared:
            continue
        a = [not base[k] for k in shared]
        b = [not other[k] for k in shared]
        stat, p, disc_b, disc_c = mcnemar(a, b)
        out["mcnemar"][f"b1-refuse_vs_{system}_breach"] = {
            "n_pairs": len(shared), "discordant_b": disc_b, "discordant_c": disc_c,
            "statistic": round(stat, 4), "p_value": round(p, 4)}
        au = [keyed("b1-refuse", "utility")[k] for k in shared]
        bu = [keyed(system, "utility")[k] for k in shared]
        stat, p, disc_b, disc_c = mcnemar(au, bu)
        out["mcnemar"][f"b1-refuse_vs_{system}_utility"] = {
            "n_pairs": len(shared), "discordant_b": disc_b, "discordant_c": disc_c,
            "statistic": round(stat, 4), "p_value": round(p, 4)}
    return out


def _task_mcnemar(tasks_path: Path) -> dict:
    rows = _jsonl(tasks_path)
    if not rows:
        return {}
    def keyed(system):
        return {(r["suite"], r["user_task_id"]): r["utility"]
                for r in rows if r["system"] == system}
    base = keyed("b1-refuse")
    out = {}
    for system in ("b3a", "b3c", "b1-defer"):
        other = keyed(system)
        shared = sorted(base.keys() & other.keys())
        if not shared:
            continue
        a = [base[k] for k in shared]
        b = [other[k] for k in shared]
        stat, pval, disc_b, disc_c = mcnemar(a, b)
        out[f"b1-refuse_vs_{system}_utility"] = {
            "n_tasks": len(shared), "discordant_b": disc_b, "discordant_c": disc_c,
            "statistic": round(stat, 4), "p_value": round(pval, 4)}
    return out


_LADDER_TASK_DIRS = {
    "1.5B": "ladder_tasks/qwen2.5-1.5b-instruct",
    "3B": "ladder_tasks/qwen2.5-3b-instruct",
    "7B": "variance_run1",
}


def _size_mcnemar(system: str = "b3a") -> dict:
    per_size = {}
    for size, rel in _LADDER_TASK_DIRS.items():
        rows = _jsonl(RESULTS / rel / "synth_tasks.jsonl")
        keyed = {(r["suite"], r["user_task_id"]): r["utility"]
                 for r in rows if r["system"] == system}
        if keyed:
            per_size[size] = keyed
    out: dict[str, Any] = {}
    order = [s for s in ("1.5B", "3B", "7B") if s in per_size]
    for lo, hi in zip(order, order[1:]):
        shared = sorted(per_size[lo].keys() & per_size[hi].keys())
        if not shared:
            continue
        a = [per_size[lo][k] for k in shared]
        b = [per_size[hi][k] for k in shared]
        stat, pval, disc_b, disc_c = mcnemar(a, b)
        out[f"{lo}_vs_{hi}_{system}"] = {
            "n_tasks": len(shared),
            "util_lo": sum(a), "util_hi": sum(b),
            "gained_by_larger": disc_c, "lost_by_larger": disc_b,
            "statistic": round(stat, 4), "p_value": round(pval, 4)}
    if len(order) >= 2 and order[0] != order[-1]:
        lo, hi = order[0], order[-1]
        shared = sorted(per_size[lo].keys() & per_size[hi].keys())
        a = [per_size[lo][k] for k in shared]
        b = [per_size[hi][k] for k in shared]
        stat, pval, disc_b, disc_c = mcnemar(a, b)
        out[f"{lo}_vs_{hi}_{system}_endpoints"] = {
            "n_tasks": len(shared),
            "util_lo": sum(a), "util_hi": sum(b),
            "gained_by_larger": disc_c, "lost_by_larger": disc_b,
            "statistic": round(stat, 4), "p_value": round(pval, 4)}
    return out


def _utility_block() -> dict:
    out: dict[str, Any] = {}
    for d in sorted((RESULTS / "ladder").glob("*/")):
        summary = _load(d / "synth_summary.json")
        if not summary:
            continue
        suites = sorted({k.split(":")[0] for k in summary})
        systems = sorted({k.split(":")[1] for k in summary})
        doable = sum(summary[f"{su}:b1-none"]["doable_without_policy"] for su in suites)
        out[d.name] = {
            sysn: _rate_ci(sum(summary[f"{su}:{sysn}"]["utility"] for su in suites), doable)
            for sysn in systems}
    return out


def _accounting() -> dict:
    try:
        from agentdojo.task_suite.load_suites import get_suite
    except Exception:
        return {}
    suites, tot = {}, 0
    for su in ("banking", "slack", "travel", "workspace"):
        s = get_suite("v1.2", su)
        u, i = len(s.user_tasks), len(s.injection_tasks)
        suites[su] = {"user": u, "injection": i, "combos": u * i, "with_2_shapes": u * i * 2}
        tot += u * i
    pairs = _jsonl(RESULTS / "attack_proceed" / "synth_attack_pairs.jsonl") or         _jsonl(RESULTS / "synth_attack_pairs.jsonl")
    gated = len([r for r in pairs if r["system"] == "b1-refuse"])
    return {"per_suite": suites, "combos": tot, "raw_with_2_shapes": tot * 2, "gated": gated}


def _reviewer_sensitivity() -> dict:
    halt = _jsonl(RESULTS / "synth_attack_pairs.jsonl")
    proc = _jsonl(RESULTS / "attack_proceed" / "synth_attack_pairs.jsonl")
    if not halt or not proc:
        return {}
    def keyed(rows, sysn):
        return {(r["suite"], r["mode"], r["user_task_id"], r["injection_task_id"]): r["breached"]
                for r in rows if r["system"] == sysn}
    out = {}
    n = len({k for k in keyed(halt, "b1-refuse")})
    for sysn in ("b1-refuse", "b3a", "b3c"):
        h, pz = keyed(halt, sysn), keyed(proc, sysn)
        auto = sum(h.values())
        gated = sum(1 for k in h if pz.get(k) and not h[k])
        out[sysn] = {"auto_breaches": auto, "reviewer_gated": gated, "n": n,
                     "asr_at_p": {str(q): round((auto + gated * q) / n, 4)
                                  for q in (0.0, 0.01, 0.05, 0.10, 0.20, 1.0)}}
    return out


def _latency() -> dict:
    import statistics
    out = {}
    for d in sorted((RESULTS / "ladder").glob("*/")):
        for f in d.glob("synth_verdicts_*.jsonl"):
            lat = [r["latency_ms"] for r in _jsonl(f) if r.get("latency_ms")]
            if lat:
                out[d.name] = {"n": len(lat), "median_ms": round(statistics.median(lat)),
                               "mean_ms": round(statistics.mean(lat)),
                               "p95_ms": round(sorted(lat)[int(len(lat) * 0.95)]),
                               "max_ms": max(lat)}
    return out


def main(argv: list[str] | None = None) -> int:
    report = {
        "attack_halt": _attack_block(
            RESULTS / "synth_attack_summary.json",
            RESULTS / "synth_attack_pairs.jsonl", "halt"),
        "attack_proceed": _attack_block(
            RESULTS / "attack_proceed" / "synth_attack_summary.json",
            RESULTS / "attack_proceed" / "synth_attack_pairs.jsonl", "proceed"),
        "utility": _utility_block(),
        "accounting": _accounting(),
        "reviewer_sensitivity": _reviewer_sensitivity(),
        "latency": _latency(),
        "utility_mcnemar": {
            "top_level": _task_mcnemar(RESULTS / "synth_tasks.jsonl"),
            **{d.name: _task_mcnemar(d / "synth_tasks.jsonl")
               for d in sorted((RESULTS / "ladder").glob("*/"))},
        },
        "size_mcnemar": {
            "b3a": _size_mcnemar("b3a"),
            "b3c": _size_mcnemar("b3c"),
        },
    }
    out_path = RESULTS / "stats_summary.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    def show(iv: dict | None) -> str:
        return "--" if iv is None else f"{iv['point']:.1%} [{iv['low']:.1%},{iv['high']:.1%}] {iv['k']}/{iv['n']}"

    for regime in ("attack_halt", "attack_proceed"):
        blk = report[regime]
        if not blk["aggregate"]:
            print(f"\n=== {regime}: no data ===")
            continue
        print(f"\n=== {regime} · aggregate ASR (Wilson 95%) ===")
        for system, v in blk["aggregate"].items():
            print(f"  {system:10} all={show(v['asr'])}")
            print(f"  {'':10} public={show(v['asr_public'])}  restricted={show(v['asr_restricted'])}")
        if blk["mcnemar"]:
            print(f"  -- McNemar (paired on same pairs) --")
            for name, m in blk["mcnemar"].items():
                print(f"     {name:32} discordant {m['discordant_b']}/{m['discordant_c']}"
                      f"  p={m['p_value']:.4f}  n={m['n_pairs']}")

    print(f"\n=== utility (Wilson 95%, over 83 doable) ===")
    for model, systems in report["utility"].items():
        cells = "  ".join(f"{s}={show(iv)}" for s, iv in systems.items() if s in ("b1-refuse", "b3a", "b3c"))
        print(f"  {model:24} {cells}")

    um = {k: v for k, v in report["utility_mcnemar"].items() if v}
    if um:
        print(f"\n=== utility McNemar (paired per task) ===")
        for src, tests in um.items():
            for name, m in tests.items():
                print(f"  {src:24} {name:28} discordant {m['discordant_b']}/{m['discordant_c']}"
                      f"  p={m['p_value']:.4f}  n={m['n_tasks']}")
    else:
        print(f"\n(utility McNemar: no synth_tasks.jsonl yet -- re-run synth-run/ladder to populate)")

    sm = {k: v for k, v in report["size_mcnemar"].items() if v}
    if sm:
        print(f"\n=== model-size McNemar (paired per task, same seam) ===")
        for system, tests in sm.items():
            for name, m in tests.items():
                print(f"  {system:6} {name:28} {m['util_lo']}->{m['util_hi']}/{m['n_tasks']}"
                      f"  +{m['gained_by_larger']}/-{m['lost_by_larger']}"
                      f"  p={m['p_value']:.4f}")
    else:
        print(f"\n(size McNemar: need synth_tasks.jsonl under results/ladder_tasks/*)")

    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
