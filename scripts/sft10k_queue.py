"""Wait for verified expanded data and finished DPO evaluation on GPU 0."""

import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.round2_train import atomic_json

ROOT = Path("results/round2/sft10k")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    def state(name, **kw):
        atomic_json(
            ROOT / "status.json", dict(state=name, updated_at=time.time(), **kw)
        )

    while True:
        ready = []
        for path, key, value in [
            ("data/prepared/sft10k/gate.json", "passed", True),
            ("results/round2/dpo-selected/selection.json", "state", "completed"),
        ]:
            try:
                ready.append(json.loads(Path(path).read_text()).get(key) == value)
            except (FileNotFoundError, json.JSONDecodeError):
                ready.append(False)
        if all(ready):
            break
        state("waiting", data_verified=ready[0], dpo_evaluation_completed=ready[1])
        time.sleep(30)
    model = Path("results/round2/sft10k-model")
    if not (model / "repairrl-training.json").exists():
        state("training", source="M0 Instruct; fresh branch", pairs=10000)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "training.train_sft10k",
                "--config",
                "configs/sft10k.yaml",
            ],
            check=True,
        )
    selection = json.loads((model / "selection.json").read_text())
    for benchmark in ("runbugrun", "humanevalpack"):
        out = ROOT / f"M1-sft10k-v1-{benchmark}.jsonl"
        if out.with_suffix(".summary.json").exists():
            continue
        state("evaluating", benchmark=benchmark, selected=selection["selected"])
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "evaluation.round2",
                "--model",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "--adapter",
                selection["selected"]["adapter"],
                "--arm",
                "M1-sft10k",
                "--controller",
                "v1",
                "--benchmark",
                benchmark,
                "--fp16",
                "--output",
                str(out),
            ],
            check=True,
        )
    state("completed")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(
            ROOT / "status.json",
            {"state": "failed", "error": repr(e), "updated_at": time.time()},
        )
        raise
