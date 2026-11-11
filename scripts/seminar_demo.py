"""Offline replay of measured, explicitly curated cases; never prints hidden tests."""

import argparse
import json
from pathlib import Path
from repairrl.common import read_jsonl
from scripts.seminar_snapshot import ROOT, DOC, RUNS


def build():
    banks = {}
    for benchmark in ["runbugrun", "humanevalpack"]:
        banks[benchmark] = {}
        for arm, (folder, prefix) in RUNS.items():
            rows = read_jsonl(ROOT / "results" / folder / f"{prefix}-{benchmark}.jsonl")
            banks[benchmark][arm] = {str(row["id"]): row for row in rows}
    specifications = [
        ("SFT first-attempt rescue", "runbugrun", "25202"),
        ("Feedback-guided recovery", "runbugrun", "284537"),
        ("SFT success / GRPO regression", "runbugrun", "219946"),
        ("Weak-DPO external-benchmark success", "humanevalpack", "Python/6"),
    ]
    bank = banks["runbugrun"]
    failures = [
        k for k in bank["M1"] if all(not bank[a][k]["agent_at_3"] for a in bank)
    ]
    runtime = [
        k
        for k in failures
        if bank["M1"][k]["trace"][0]["grade"]["category"] == "runtime_error"
    ]
    specifications.append(
        (
            "Persistent failure, including runtime feedback",
            "runbugrun",
            (runtime or failures)[0],
        )
    )
    source = {
        "runbugrun": {
            str(r["id"]): r
            for r in read_jsonl(ROOT / "data/prepared/scale-verified/test.jsonl")
        },
        "humanevalpack": {
            str(r["id"]): r
            for r in read_jsonl(ROOT / "data/external/humanevalpack.jsonl")
        },
    }
    public, private = [], []
    for n, (label, benchmark, rid) in enumerate(specifications, 1):
        models = banks[benchmark]
        facts = {
            "case": n,
            "label": label,
            "benchmark": benchmark,
            "id": rid,
            "problem_id": str(models["M1"][rid]["problem_id"]),
            "curated_by_observed_outcome": True,
            "models": {},
        }
        examples = {}
        for arm in models:
            r = models[arm][rid]
            facts["models"][arm] = {
                "pass_at_1": r["pass_at_1"],
                "agent_at_3": r["agent_at_3"],
                "attempts": r["attempts"],
                "categories": [t["grade"]["category"] for t in r["trace"]],
            }
            examples[arm] = {
                "candidate": r["candidate"],
                "trace": [
                    {
                        "attempt": t["attempt"],
                        "generation": t["generation"],
                        "grade": {
                            k: t["grade"][k]
                            for k in [
                                "passed",
                                "total",
                                "category",
                                "valid",
                                "success",
                                "fraction",
                            ]
                        },
                    }
                    for t in r["trace"]
                ],
            }
        task = source[benchmark][rid]
        public.append(facts)
        private.append(
            facts
            | {
                "source_task": {
                    k: task[k]
                    for k in [
                        "statement",
                        "buggy_code",
                        "fixed_code",
                        "entry_point",
                        "declaration",
                    ]
                    if k in task
                },
                "recorded_models": examples,
            }
        )
    return public, private


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export-private", type=Path)
    p.add_argument(
        "--packet",
        type=Path,
        default=Path("/home/thao/Thuc/RepairRL-preservation/demo-packet.json"),
    )
    p.add_argument("--case", type=int)
    p.add_argument("--arm", choices=["M0", "M1", "M2", "M3"], default="M1")
    p.add_argument("--show-code", action="store_true")
    a = p.parse_args()
    if a.export_private:
        public, private = build()
        a.export_private.parent.mkdir(parents=True, exist_ok=True)
        a.export_private.write_text(json.dumps(private, indent=2, ensure_ascii=False))
        (DOC / "generated/demo-cases.json").write_text(json.dumps(public, indent=2))
        lines = [
            "# Curated measured demonstration cases",
            "",
            "Selection is outcome-based and disclosed. These cases are not aggregate evidence.",
            "",
            "| Case | Purpose | Benchmark / ID | M0 agent | M1 agent | M2 agent | M3 agent |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
        for r in public:
            flags = [
                "yes" if r["models"][m]["agent_at_3"] else "no"
                for m in ["M0", "M1", "M2", "M3"]
            ]
            lines.append(
                f"| {r['case']} | {r['label']} | {r['benchmark']} / {r['id']} | "
                + " | ".join(flags)
                + " |"
            )
        (DOC / "generated/DEMO_CASES.md").write_text("\n".join(lines) + "\n")
        print("Exported private packet and public outcome-only case index")
        return
    if not a.packet.exists():
        raise SystemExit(
            "Restore the private demo packet or run --export-private after restoring historical results. No new result was generated."
        )
    packet = json.loads(a.packet.read_text())
    if a.case is None:
        for case in packet:
            print(case["case"], case["label"], case["benchmark"], case["id"])
        return
    case = next(r for r in packet if r["case"] == a.case)
    print(
        "CURATED RECORDED REPLAY:", case["label"], case["benchmark"], case["id"], a.arm
    )
    for trace in case["recorded_models"][a.arm]["trace"]:
        g = trace["grade"]
        print("attempt", trace["attempt"], f"{g['passed']}/{g['total']}", g["category"])
        if a.show_code:
            print(trace["generation"]["text"])
    print("This replays recorded outputs; it does not execute a model or hidden tests.")


if __name__ == "__main__":
    main()
