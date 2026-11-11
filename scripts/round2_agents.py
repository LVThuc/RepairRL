"""Local GPU controller ablation: identical dev cohort, including sampling control."""
import json
from pathlib import Path
import subprocess
import sys
from scripts.round2_train import atomic_json


def main():
    out = Path("results/round2/local-agent")
    out.mkdir(parents=True, exist_ok=True)
    for controller in ("v1", "no-diagnosis", "v31"):
        target = out / f"{controller}.jsonl"
        if target.with_suffix(".summary.json").exists():
            continue
        atomic_json(out / "status.json", {"state": "running", "controller": controller})
        with (out / f"{controller}.log").open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", "evaluation.round2", "--arm", "M1",
                "--adapter", "checkpoints/M1", "--fp16", "--controller", controller, "--dev", "--limit", "40",
                "--output", str(target)], stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = {k: json.loads((out / f"{k}.summary.json").read_text()) for k in ("v1", "no-diagnosis", "v31")}
    if len({tuple(r["protocol"]["task_ids"]) for r in reports.values()}) != 1:
        raise ValueError("Controller cohorts differ")
    atomic_json(out / "comparison.json", reports)
    atomic_json(out / "status.json", {"state": "completed", "report": str(out / "comparison.json")})


if __name__ == "__main__":
    main()
