"""Export completed matched V1 results and figures; never publish raw test data."""

import json
from pathlib import Path
from repairrl.common import digest, read_jsonl
from scripts.round2_report import paired

ROOT = Path("docs/seminar/report")
NAMES = ["PT-base", "M0 Instruct", "M1 SFT", "DPO-hard s66", "GRPO s800"]


def paths(benchmark):
    return [
        Path(f"results/round2/pretrained-base/PT-base-v1-{benchmark}.jsonl"),
        Path(f"results/round2/baselines/M0-v1-{benchmark}.jsonl"),
        Path(f"results/round2/baselines-common-parent/M1-v1-{benchmark}.jsonl"),
        Path(f"results/round2/remote/dpo-selected/M2-hard-v1-{benchmark}.jsonl"),
        Path(
            f"results/round2/grpo-interim/eval-step-800/M3-long-interim-step-800-v1-{benchmark}.jsonl"
        ),
    ]


def main():
    ROOT.mkdir(exist_ok=True)
    exported = []
    banks = {}
    comparisons = []
    for benchmark in ["runbugrun", "humanevalpack"]:
        banks[benchmark] = {}
        for name, path in zip(NAMES, paths(benchmark), strict=True):
            summary = path.with_suffix(".summary.json")
            d = json.loads(summary.read_text())
            r = read_jsonl(path)
            assert d["metrics"]["n"] == len(r)
            banks[benchmark][name] = r
            exported.append(
                {
                    "model": name,
                    "benchmark": benchmark,
                    "metrics": d["metrics"],
                    "first_count": sum(x["pass_at_1"] for x in r),
                    "agent_count": sum(x["agent_at_3"] for x in r),
                    "first_partial": sum(x["trace"][0]["grade"]["fraction"] for x in r)
                    / len(r),
                    "best_partial": sum(
                        max(t["grade"]["fraction"] for t in x["trace"]) for x in r
                    )
                    / len(r),
                    "summary_path": str(summary),
                    "summary_sha256": digest(summary),
                    "trace_sha256": digest(path),
                }
            )
        for ref, candidate in [
            ("M0 Instruct", "M1 SFT"),
            ("M1 SFT", "DPO-hard s66"),
            ("M1 SFT", "GRPO s800"),
        ]:
            for metric in ["pass_at_1", "agent_at_3"]:
                comparisons.append(
                    {
                        "benchmark": benchmark,
                        "reference": ref,
                        "candidate": candidate,
                        "metric": metric,
                        **paired(
                            banks[benchmark][ref], banks[benchmark][candidate], metric
                        ),
                    }
                )
    (ROOT / "results.json").write_text(json.dumps(exported, indent=2))
    (ROOT / "paired-comparisons.json").write_text(json.dumps(comparisons, indent=2))
    cases = []
    private = []
    for benchmark, rid, title in [
        ("runbugrun", "21346", "Cup swap: prevent a double update"),
        ("runbugrun", "186967", "An unwanted debug print returns after DPO"),
        ("humanevalpack", "Python/24", "Largest divisor: formatting and arithmetic"),
    ]:
        source = (
            "data/prepared/scale-verified/test.jsonl"
            if benchmark == "runbugrun"
            else "data/external/humanevalpack.jsonl"
        )
        task = next(r for r in read_jsonl(source) if str(r["id"]) == rid)
        rows = {
            n: next(r for r in bank if str(r["id"]) == rid)
            for n, bank in banks[benchmark].items()
        }
        public = {
            "id": rid,
            "problem_id": task["problem_id"],
            "benchmark": benchmark,
            "title": title,
            "curated_by_observed_outcome": True,
            "models": {
                n: {
                    "pass_at_1": r["pass_at_1"],
                    "agent_at_3": r["agent_at_3"],
                    "first_fraction": r["trace"][0]["grade"]["fraction"],
                    "first_category": r["trace"][0]["grade"]["category"],
                }
                for n, r in rows.items()
            },
        }
        cases.append(public)
        # Owner-only replay data excludes hidden tests; never add this file to Git.
        clean = {
            n: {
                "attempts": [
                    {
                        "code": t["generation"]["text"],
                        "fraction": t["grade"]["fraction"],
                        "category": t["grade"]["category"],
                    }
                    for t in r["trace"]
                ]
            }
            for n, r in rows.items()
        }
        private.append(
            {
                "id": rid,
                "benchmark": benchmark,
                "statement": task["statement"],
                "buggy_code": task["buggy_code"],
                "fixed_code": task["fixed_code"],
                "models": clean,
            }
        )
    (ROOT / "case-studies.json").write_text(json.dumps(cases, indent=2))
    Path(
        "/home/thao/Thuc/RepairRL-preservation/report-case-studies-private.json"
    ).write_text(json.dumps(private, indent=2))
    history = read_jsonl("results/round2/remote/grpo-dense/dev-history.jsonl")
    (ROOT / "grpo-dev.json").write_text(
        json.dumps(
            [
                {
                    "step": r["step"],
                    "pass1": r["metrics"]["pass@1"],
                    "partial": r["metrics"]["partial_fraction"],
                }
                for r in history
            ],
            indent=2,
        )
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, benchmark in zip(axes, ["runbugrun", "humanevalpack"], strict=True):
        rows = [r for r in exported if r["benchmark"] == benchmark]
        x = np.arange(len(rows))
        w = 0.36
        for offset, metric, label in [
            (-w / 2, "pass@1", "First attempt"),
            (w / 2, "agent@3", "Within 3 attempts"),
        ]:
            bars = ax.bar(
                x + offset, [r["metrics"][metric] * 100 for r in rows], w, label=label
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_xticks(x, NAMES, rotation=20, ha="right")
        ax.set_ylim(0, 80)
        ax.set_ylabel("Tasks solved (%)")
        ax.set_title(benchmark + " held-out")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    for ext in ["png", "svg"]:
        fig.savefig(ROOT / f"v1-heldout-results.{ext}")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    ax.plot(
        [r["step"] for r in history],
        [r["metrics"]["pass@1"] * 100 for r in history],
        marker="o",
        label="Dev pass@1",
    )
    ax.plot(
        [r["step"] for r in history],
        [r["metrics"]["partial_fraction"] * 100 for r in history],
        marker="s",
        label="Dev partial test fraction",
    )
    ax.set(
        xlabel="GRPO optimizer step",
        ylabel="Percentage",
        title="GRPO: fixed 60-problem dev screen (not held-out)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    for ext in ["png", "svg"]:
        fig.savefig(ROOT / f"grpo-dev-curve.{ext}")
    print(
        "Exported", len(exported), "completed summaries,", len(cases), "curated cases"
    )
    for d in comparisons:
        if d["candidate"] == "GRPO s800":
            print(d)


if __name__ == "__main__":
    main()
