"""Freeze controller after dev ablation; use local GPU for common-protocol baselines."""
import json
from pathlib import Path
import subprocess
import sys
import time
from repairrl.common import digest
from scripts.round2_train import atomic_json


def main():
    out = Path("results/round2")
    status = out / "baseline-queue.json"
    while True:
        atomic_json(status, {"state": "waiting_for_controller_dev"})
        try:
            report = json.loads((out / "local-agent/status.json").read_text())
            if report["state"] == "completed":
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(20)
    comparison = json.loads((out / "local-agent/comparison.json").read_text())
    sources = ["agent/repair_v31.py", "agent/repair.py", "agent/repair_v2.py", "agent/repair_v3.py",
               "evaluation/round2.py", "evaluation/grading_service.py", "repairrl/common.py", "repairrl/model.py"]
    protocol = {"version": "repairrl-round2-20261111", "source_hashes": {p: digest(p) for p in sources},
                "precision": "FP16 inference for every arm", "max_context": 4096, "max_new_tokens": 768,
                "max_attempts": 3, "max_diagnoses": 2, "diagnosis_tokens": 128, "seed": 42,
                "retry_temperature": .4, "controllers": ["v1", "v31"],
                "dev_ablation_sha256": digest(out / "local-agent/comparison.json"),
                "dev_controller_scores": {k: v["metrics"] for k, v in comparison.items()},
                "note": "Both controllers reported; model selection uses dev only. V1 has zero diagnosis calls and greedy retries."}
    freeze = out / "frozen-protocol.json"
    if freeze.exists() and json.loads(freeze.read_text()) != protocol:
        raise ValueError("Cannot overwrite frozen protocol")
    atomic_json(freeze, protocol)
    destination = out / "baselines"
    destination.mkdir(exist_ok=True)
    for arm, model in [("M0", "Qwen/Qwen2.5-Coder-1.5B-Instruct"), ("M1", "checkpoints/M1-merged-round2")]:
        for controller in ("v1", "v31"):
            for benchmark in ("runbugrun", "humanevalpack"):
                for source, sha in protocol["source_hashes"].items():
                    if digest(source) != sha:
                        raise ValueError(f"Frozen source changed: {source}")
                path = destination / f"{arm}-{controller}-{benchmark}.jsonl"
                if path.with_suffix(".summary.json").exists():
                    continue
                atomic_json(status, {"state": "evaluating", "arm": arm, "controller": controller, "benchmark": benchmark})
                with path.with_suffix(".log").open("a") as log:
                    subprocess.run([sys.executable, "-u", "-m", "evaluation.round2", "--model", model,
                        "--arm", arm, "--controller", controller, "--benchmark", benchmark, "--fp16", "--output", str(path)],
                        stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(status, {"state": "completed"})


if __name__ == "__main__":
    main()
