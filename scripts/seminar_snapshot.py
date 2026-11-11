"""Generate public aggregate evidence; never export statements, code or hidden tests."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from repairrl.common import digest, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/seminar"
RUNS = {
    "M0": ("full-sft-20261110T150016Z", "baseline-M0"),
    "M1": ("full-sft-20261110T150016Z", "post-sft-M1"),
    "M2": ("full-dpo-recovery-20261110T182750Z", "M2"),
    "M3": ("full-grpo-20261110T182750Z", "M3"),
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def snapshot():
    generated = DOC / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    complete, histories = [], {}
    for arm, (folder, prefix) in RUNS.items():
        for benchmark in ("runbugrun", "humanevalpack"):
            path = ROOT / "results" / folder / f"{prefix}-{benchmark}.summary.json"
            report = json.loads(path.read_text())
            raw = read_jsonl(
                path.with_name(path.name.replace(".summary.json", ".jsonl"))
            )
            complete.append(
                {
                    "arm": arm,
                    "benchmark": benchmark,
                    "n": len(raw),
                    "pass1_correct": sum(r["pass_at_1"] for r in raw),
                    "agent3_correct": sum(r["agent_at_3"] for r in raw),
                    "metrics": report["metrics"],
                    "wall_seconds": report["wall_seconds"],
                    "summary_path": str(path.relative_to(ROOT)),
                    "summary_sha256": digest(path),
                    "raw_path": str(path.with_suffix("").relative_to(ROOT)).replace(
                        ".summary", ""
                    )
                    + ".jsonl",
                    "protocol": {
                        "max_context": 2048,
                        "max_new_tokens": 512,
                        "controller": "v1",
                        "quantization": "NF4",
                    },
                }
            )
        if arm != "M0":
            path = ROOT / "checkpoints" / arm / "repairrl-training.json"
            training = json.loads(path.read_text())
            histories[arm] = {
                k: training[k]
                for k in [
                    "metrics",
                    "log_history",
                    "training_seconds",
                    "peak_allocated_gb",
                    "dataset_hashes",
                ]
            }
            histories[arm]["metadata_sha256"] = digest(path)
    dump(generated / "round1-results.json", complete)
    dump(generated / "round1-training-curves.json", histories)
    with (generated / "round1-results.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "arm",
                "benchmark",
                "n",
                "pass1_correct",
                "agent3_correct",
                "pass1",
                "agent3",
                "wall_seconds",
            ]
        )
        for row in complete:
            writer.writerow(
                [
                    row[k]
                    for k in [
                        "arm",
                        "benchmark",
                        "n",
                        "pass1_correct",
                        "agent3_correct",
                    ]
                ]
                + [
                    row["metrics"]["pass@1"],
                    row["metrics"]["agent@3"],
                    row["wall_seconds"],
                ]
            )
    lines = [
        "# Completed round-1 evidence",
        "",
        "Automatically regenerated from immutable local summaries and raw traces.",
        "",
        "| Model | Benchmark | First attempt | Agent@3 |",
        "|---|---|---:|---:|",
    ]
    for row in complete:
        lines.append(
            f"| {row['arm']} | {row['benchmark']} | {row['pass1_correct']}/{row['n']} ({row['metrics']['pass@1']:.1%}) | {row['agent3_correct']}/{row['n']} ({row['metrics']['agent@3']:.1%}) |"
        )
    (generated / "COMPLETED_RESULTS.md").write_text("\n".join(lines) + "\n")
    paths = [
        "repairrl/common.py",
        "repairrl/model.py",
        "data/extract_runbugrun.py",
        "data/filter_and_split.py",
        "data/filter_context.py",
        "training/common.py",
        "training/train_sft.py",
        "training/train_dpo.py",
        "training/train_grpo.py",
        "training/resume.py",
        "evaluation/runner.py",
        "evaluation/reward.py",
        "evaluation/reward_v2.py",
        "evaluation/benchmark.py",
        "evaluation/round2.py",
        "agent/sandbox.py",
        "agent/repair.py",
        "agent/repair_v3.py",
        "agent/repair_v31.py",
        "scripts/round2_train.py",
        "scripts/round2_fast_rollouts.py",
        "scripts/round2_jobs.py",
        "scripts/round2_finish.py",
    ]
    index = [
        "# Implementation index",
        "",
        "Line numbers refer to this checkout; hashes identify exact files.",
        "",
    ]
    for name in paths:
        p = ROOT / name
        tree = ast.parse(p.read_text())
        index += [f"## [{name}](../../../{name})", "", f"SHA256: `{digest(p)}`", ""]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                index.append(
                    f"- [`{node.name}`](../../../{name}#L{node.lineno}) — line {node.lineno}"
                )
        index.append("")
    (generated / "CODE_INDEX.md").write_text("\n".join(index))
    ongoing = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "round2": {},
        "local_agent": {},
    }
    remote = ROOT / "results/round2/remote"
    for relative in [
        "rollouts-fp16/status.json",
        "grpo-dense/selection.json",
        "grpo-dense/variance.json",
        "dpo-queue.json",
        "grpo-followup-queue.json",
    ]:
        p = remote / relative
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        if relative.endswith("variance.json"):
            d = {k: d[k] for k in ["passed", "mixed_group_fraction", "reward_version"]}
        elif relative.endswith("selection.json"):
            d = {
                "state": d["state"],
                "latest_dev_step": d["steps"],
                "selected_dev": d["selected"]["metrics"],
                "parent_dev": d["baseline"]["metrics"],
                "improved_over_parent": d["improved_over_parent"],
            }
        ongoing["round2"][relative] = d
    for controller in ["v1", "no-diagnosis", "v31"]:
        p = ROOT / f"results/round2/local-agent/{controller}.summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            ongoing["local_agent"][controller] = {
                "status": "completed",
                "metrics": d["metrics"],
                "n": d["metrics"]["n"],
            }
        else:
            raw = ROOT / f"results/round2/local-agent/{controller}.jsonl"
            if raw.exists():
                count = sum(1 for line in raw.open() if line.strip())
                ongoing["local_agent"][controller] = {
                    "status": "partial",
                    "completed_tasks": count,
                    "planned_tasks": 40,
                }
            else:
                ongoing["local_agent"][controller] = {"status": "not_started"}
    log = Path("/home/thao/Thuc/RepairRL-preservation/remote-live/logs/grpo.log")
    if log.exists():
        text = log.read_text(errors="replace")
        steps = re.findall(r"\|\s*(\d+)/2000", text)
        ongoing["grpo_optimizer_step_observed"] = int(steps[-1]) if steps else None
    dump(generated / "LIVE_STATUS.json", ongoing)
    # Export completed second-round summaries separately; partial coverage is
    # labelled as such and can never silently replace the original four-arm table.
    candidates = list((ROOT / "results/round2/baselines").glob("*.summary.json"))
    candidates += list(
        (ROOT / "results/round2/baselines-common-parent").glob("*.summary.json")
    )
    for family in ["dpo", "grpo"]:
        candidates += list((remote / (family + "-selected")).glob("M*.summary.json"))
    final_rows = {}
    for path in candidates:
        report = json.loads(path.read_text())
        protocol = report["protocol"]
        if (
            protocol["arm"] == "M1"
            and protocol["model"] != "checkpoints/M1-merged-round2"
        ):
            continue
        key = (protocol["arm"], protocol["controller"], protocol["benchmark"])
        final_rows[key] = {
            "arm": key[0],
            "controller": key[1],
            "benchmark": key[2],
            "metrics": report["metrics"],
            "summary_path": str(path.relative_to(ROOT)),
            "summary_sha256": digest(path),
            "status": "completed-evaluation",
        }
    dump(generated / "round2-results.json", list(final_rows.values()))
    table = [
        "# Second-round evaluation progress",
        "",
        f"Completed summary files: {len(final_rows)}/16.",
        "",
        "Separate FP16/768-token protocol. Missing rows remain pending. This is not a complete comparison until all required rows exist.",
        "",
        "| Model | Controller | Benchmark | n | pass@1 | agent@3 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key in sorted(final_rows):
        row = final_rows[key]
        m = row["metrics"]
        table.append(
            f"| {row['arm']} | {row['controller']} | {row['benchmark']} | {m['n']} | {m['pass@1']:.1%} | {m['agent@3']:.1%} |"
        )
    (generated / "ROUND2_RESULTS.md").write_text("\n".join(table) + "\n")
    dev_history = []
    for path in remote.glob("*/dev-history.jsonl"):
        try:
            records = read_jsonl(path)
        except ValueError:
            continue
        dev_history.extend(
            {
                "trial": path.parent.name,
                "step": r["step"],
                "epoch": r.get("epoch"),
                "metrics": r["metrics"],
            }
            for r in records
        )
    dump(generated / "round2-dev-history.json", dev_history)
    if (ROOT / "results/round2/FINAL_ROUND2.md").exists():
        (generated / "FINAL_ROUND2.md").write_bytes(
            (ROOT / "results/round2/FINAL_ROUND2.md").read_bytes()
        )
    md = [
        "# Live experiment status",
        "",
        f"Snapshot: {ongoing['captured_at_utc']}",
        "",
        "This is a point-in-time view, not completed held-out evidence. A running service may be waiting, not optimizing.",
        "",
        f"Latest GRPO optimizer step observed in mirrored log: {ongoing.get('grpo_optimizer_step_observed', 'unknown')}.",
        "",
        "## Remote queues",
        "",
        "```json",
        json.dumps(ongoing["round2"], indent=2),
        "```",
        "",
        "## Local controller ablation",
        "",
        "```json",
        json.dumps(ongoing["local_agent"], indent=2),
        "```",
        "",
        "Async dev-scoring code under experimental/ is deferred and was not deployed. Current remote dev scoring is synchronous.",
    ]
    (generated / "LIVE_STATUS.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    import time

    while True:
        snapshot()
        if not args.watch:
            break
        time.sleep(60)
