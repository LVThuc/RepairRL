"""Rerank screened checkpoints on full dev at deployment precision, then test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from repairrl.common import read_jsonl, digest
from scripts.round2_train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["dpo", "grpo"], required=True)
    p.add_argument("--trials", nargs="+", type=Path, required=True)
    a = p.parse_args()
    out = Path("results/round2") / (a.family + "-selected")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for trial in a.trials:
        if not (trial / "selection.json").exists() or json.loads((trial / "selection.json").read_text())["state"] != "completed":
            continue
        records.extend(r for r in read_jsonl(trial / "dev-history.jsonl") if r["step"] > 0)
    if not records:
        raise ValueError("No completed trial has a selectable trained checkpoint")
    top = sorted(records, key=lambda r: (-r["metrics"]["pass@1"], -r["metrics"]["partial_fraction"], r["step"]))[:3]
    commands = [("parent", None)] + [(f"candidate-{i}", r["adapter"]) for i, r in enumerate(top)]
    for label, adapter in commands:
        path = out / (label + "-dev.jsonl")
        if not path.with_suffix(".summary.json").exists():
            cmd = [sys.executable, "-u", "-m", "evaluation.round2", "--model", "checkpoints/M1-merged",
                   "--arm", label, "--controller", "single", "--dev", "--limit", "147", "--fp16", "--output", str(path)]
            if adapter:
                cmd += ["--adapter", adapter]
            subprocess.run(cmd, check=True)
    dev = [json.loads((out / f"candidate-{i}-dev.summary.json").read_text()) for i in range(len(top))]
    winner = max(range(len(top)), key=lambda i: (dev[i]["metrics"]["pass@1"], dev[i]["metrics"]["partial_fraction"], -i))
    parent = json.loads((out / "parent-dev.summary.json").read_text())
    chosen = {"family": a.family, "state": "selected_on_dev", "selected": top[winner],
              "dev_full": dev[winner], "parent_dev_full": parent,
              "improved_dev_pass1": dev[winner]["metrics"]["pass@1"] > parent["metrics"]["pass@1"],
              "candidate_dev_summaries": [r["metrics"] for r in dev],
              "note": "Top three screened snapshots reranked on all 147 distinct dev problems at FP16 deployment precision."}
    atomic_json(out / "selection.json", chosen)
    # The local controller ablation freezes this before any new headline testing.
    freeze = Path("results/round2/frozen-protocol.json")
    while not freeze.exists():
        print("Waiting for controller dev ablation/protocol freeze", flush=True)
        time.sleep(30)
    protocol = json.loads(freeze.read_text())
    for path, sha in protocol["source_hashes"].items():
        if digest(path) != sha:
            raise ValueError(f"Frozen source changed: {path}")
    label = "M2-hard" if a.family == "dpo" else "M3-long"
    for controller in ("v1", "v31"):
        for benchmark in ("runbugrun", "humanevalpack"):
            path = out / f"{label}-{controller}-{benchmark}.jsonl"
            if path.with_suffix(".summary.json").exists():
                continue
            subprocess.run([sys.executable, "-u", "-m", "evaluation.round2", "--model", "checkpoints/M1-merged",
                "--adapter", top[winner]["adapter"], "--arm", label, "--controller", controller,
                "--benchmark", benchmark, "--fp16", "--output", str(path)], check=True)
    chosen["state"] = "completed"
    atomic_json(out / "selection.json", chosen)


if __name__ == "__main__":
    main()
