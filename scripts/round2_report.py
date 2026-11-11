"""Matched, problem-clustered comparisons after all predefined evaluations finish."""
import json
from pathlib import Path
import random
import time
from repairrl.common import read_jsonl
from scripts.round2_train import atomic_json


def paired(reference, candidate, key, repetitions=2000):
    if [str(r["id"]) for r in reference] != [str(r["id"]) for r in candidate]:
        raise ValueError("Unmatched task IDs")
    groups = {}
    wins = losses = 0
    for left, right in zip(reference, candidate, strict=True):
        delta = int(right[key]) - int(left[key])
        wins += delta > 0
        losses += delta < 0
        groups.setdefault(str(left["problem_id"]), []).append(delta)
    values = list(groups.values())
    rng = random.Random(12345)
    samples = []
    for _ in range(repetitions):
        drawn = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(sum(group) for group in drawn) / sum(len(group) for group in drawn))
    samples.sort()
    return {"delta": sum(sum(g) for g in values)/len(reference), "wins": wins, "losses": losses,
            "ci95": [samples[int(.025*repetitions)], samples[min(repetitions-1, int(.975*repetitions))]],
            "clusters": len(groups), "bootstrap_repetitions": repetitions}


def report():
    root = Path("results/round2")
    files = list((root / "baselines").glob("*.summary.json"))
    files += list((root / "baselines-common-parent").glob("*.summary.json"))
    for family in ("dpo", "grpo"):
        files += list((root / "remote" / (family + "-selected")).glob("M*.summary.json"))
    reports = {}
    for path in files:
        value = json.loads(path.read_text())
        p = value["protocol"]
        if p["controller"] != "v1":
            continue
        if p["arm"] == "M1" and p["model"] != "checkpoints/M1-merged-round2":
            continue
        reports[(p["arm"], p["controller"], p["benchmark"])] = (path, value)
    expected = [(arm, controller, benchmark) for arm in ("M0", "M1", "M2-hard", "M3-long")
                for controller in ("v1",) for benchmark in ("runbugrun", "humanevalpack")]
    missing = [key for key in expected if key not in reports]
    atomic_json(root / "report-status.json", {"state": "waiting" if missing else "completed",
                                              "completed_evaluations": len(reports), "missing": missing})
    if missing:
        return False
    lines = ["# RepairRL round 2 — measured results", "", "Models were selected on dev; V1 is the active controller; historical V3 results remain archived.", ""]
    comparisons = []
    keys = ["controller", "benchmark", "data_sha256", "tests_sha256", "task_ids", "max_context",
            "max_new_tokens", "max_attempts", "max_diagnoses", "diagnosis_tokens", "retry_temperature",
            "seed", "quantized", "controller_sha256", "generator_sha256", "grader"]
    for controller in ("v1",):
        for benchmark in ("runbugrun", "humanevalpack"):
            lines += [f"## {benchmark} / {controller}", "", "| Model | pass@1 | agent@3 | validity | diagnosis calls |", "|---|---:|---:|---:|---:|"]
            baseline_path, baseline = reports[("M1", controller, benchmark)]
            reference = read_jsonl(baseline_path.with_name(baseline_path.name.replace(".summary.json", ".jsonl")))
            for arm in ("M0", "M1", "M2-hard", "M3-long"):
                path, value = reports[(arm, controller, benchmark)]
                if any(value["protocol"].get(k) != baseline["protocol"].get(k) for k in keys):
                    raise ValueError(f"Protocol mismatch for {arm}/{controller}/{benchmark}")
                m = value["metrics"]
                lines.append(f"| {arm} | {m['pass@1']:.1%} | {m['agent@3']:.1%} | {m['validity']:.1%} | {m['diagnosis_calls']} |")
                if arm in {"M2-hard", "M3-long"}:
                    candidates = read_jsonl(path.with_name(path.name.replace(".summary.json", ".jsonl")))
                    for key in ("pass_at_1", "agent_at_3"):
                        result = paired(reference, candidates, key)
                        comparisons.append({"arm": arm, "controller": controller, "benchmark": benchmark,
                                            "metric": key, **result})
            lines += [""]
    lines += ["## Paired changes versus M1", "", "Intervals use problem-clustered bootstrap; they do not correct for all experiment choices.", ""]
    for item in comparisons:
        lo, hi = item["ci95"]
        lines.append(f"- {item['arm']} / {item['controller']} / {item['benchmark']} / {item['metric']}: "
            f"{item['delta']*100:+.1f} pp; 95% interval [{lo*100:+.1f}, {hi*100:+.1f}] pp; "
            f"{item['wins']} wins / {item['losses']} losses.")
    lines += ["", "Results from the old NF4/512-token protocol remain separate. Round 2 uses FP16 inference and a 768-token code cap for every arm.",
              "RunBugRun uses selected 2–10-test suites. HumanEvalPack may overlap foundation-model pretraining. No learning improvement is assumed."]
    (root / "FINAL_ROUND2.md").write_text("\n".join(lines) + "\n")
    atomic_json(root / "paired-comparisons.json", comparisons)
    return True


if __name__ == "__main__":
    while not report():
        time.sleep(60)
