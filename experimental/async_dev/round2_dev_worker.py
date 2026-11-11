"""Local 3060 scores remote snapshots while both remote GPUs keep training."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from scripts.round2_train import atomic_json

HOST = "root@125.240.239.50"


def main():
    root = Path("results/round2")
    while True:
        try:
            ready = json.loads((root / "local-agent/status.json").read_text())["state"] == "completed"
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
        if ready and (Path("checkpoints/M1-merged-round2") / "model.safetensors").exists() and not (root / "parent-transfer.pending").exists():
            break
        atomic_json(root / "dev-worker-status.json", {"state": "waiting_for_agent_ablation_and_exact_parent"})
        time.sleep(15)
    while True:
        pending = []
        for path in (root / "remote").glob("*/snapshots/step-*/round2-checkpoint.json"):
            value = json.loads(path.read_text())
            if value.get("settings", {}).get("async_dev") and "dev_pending" in value:
                parent = path.parents[2].name
                step = value["dev_pending"]["step"]
                output = root / "dev-evaluations" / parent / f"step-{step}.jsonl"
                if not output.with_suffix(".uploaded").exists():
                    pending.append((step, parent, path, value, output))
        pending.sort(key=lambda x: (x[0], x[1]))
        if not pending:
            atomic_json(root / "dev-worker-status.json", {"state": "idle_waiting_for_snapshots"})
            time.sleep(15)
            continue
        step, parent, path, value, output = pending[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "dev-worker-status.json", {"state": "evaluating", "trial": parent, "step": step})
        if not output.with_suffix(".summary.json").exists():
            with output.with_suffix(".log").open("a") as log:
                subprocess.run([sys.executable, "-u", "-m", "evaluation.round2", "--model", "checkpoints/M1-merged-round2",
                    "--adapter", str(path.parent), "--arm", parent, "--controller", "single", "--dev", "--limit", "60",
                    "--batch-size", "8", "--fp16", "--output", str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(output.with_suffix(".summary.json").read_text())
        record = value["dev_pending"] | {"metrics": report["metrics"], "seconds": report["wall_seconds"],
            "protocol": report["protocol"], "local_raw": str(output)}
        returned = output.with_suffix(".result.json")
        atomic_json(returned, record)
        remote_dir = f"/workspace/RepairRL/results/round2/{parent}/dev-results"
        subprocess.run(["ssh", "-p", "40401", "-o", "BatchMode=yes", HOST,
                        "mkdir -p " + shlex.quote(remote_dir)], check=True)
        target = f"{remote_dir}/step-{step}.json"
        subprocess.run(["scp", "-P", "40401", "-o", "BatchMode=yes", str(returned), HOST+":"+target+".tmp"], check=True)
        subprocess.run(["ssh", "-p", "40401", "-o", "BatchMode=yes", HOST,
                        "mv " + shlex.quote(target+".tmp") + " " + shlex.quote(target)], check=True)
        output.with_suffix(".uploaded").write_text("returned to remote trainer\n")


if __name__ == "__main__":
    main()
