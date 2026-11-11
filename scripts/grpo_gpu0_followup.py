"""Use GPU0 after DPO evaluation for one improved dev-selected GRPO snapshot."""

import json
from pathlib import Path
import subprocess
import sys
import time
from repairrl.common import digest
from scripts.round2_train import atomic_json

ROOT = Path("results/round2/grpo-interim-gpu0")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    def state(name, **kw):
        atomic_json(
            ROOT / "status.json", dict(state=name, updated_at=time.time(), **kw)
        )

    request = ROOT / "request.json"
    while not request.exists():
        try:
            dpo = json.loads(
                Path("results/round2/dpo-selected/selection.json").read_text()
            )
            selection = json.loads(
                Path("results/round2/grpo-dense/selection.json").read_text()
            )
            records = [
                json.loads(line)
                for line in Path("results/round2/grpo-dense/dev-history.jsonl")
                .read_text()
                .splitlines()
            ]
            baseline = next(r for r in records if r["step"] == 800)
            best = selection["selected"]

            def score(r):
                return (r["metrics"]["pass@1"], r["metrics"]["partial_fraction"])

            if (
                dpo["state"] == "completed"
                and best["step"] > 800
                and score(best) > score(baseline)
            ):
                atomic_json(
                    request,
                    {
                        "selected": best,
                        "baseline_step800": baseline,
                        "criterion": "Strictly better dev score tuple; no held-out selection",
                        "adapter_sha256": digest(
                            Path(best["adapter"]) / "adapter_model.safetensors"
                        ),
                    },
                )
                break
            if selection["state"] == "completed" and dpo["state"] == "completed":
                state(
                    "skipped",
                    reason="No dev improvement over step800; GPU0 remains idle",
                )
                return
            state(
                "waiting",
                dpo_complete=dpo["state"] == "completed",
                best_step=best["step"],
                note="GPU0 idle while waiting; no model loaded",
            )
        except (FileNotFoundError, json.JSONDecodeError, StopIteration):
            state("waiting_for_manifests")
        time.sleep(60)
    chosen = json.loads(request.read_text())
    best = chosen["selected"]
    step = best["step"]
    adapter = best["adapter"]
    if digest(Path(adapter) / "adapter_model.safetensors") != chosen["adapter_sha256"]:
        raise ValueError("Selected adapter changed")
    frozen = json.loads(Path("results/round2/frozen-protocol.json").read_text())
    for source, sha in frozen["source_hashes"].items():
        if digest(source) != sha:
            raise ValueError(f"Frozen source changed: {source}")
    out = ROOT / f"step-{step}"
    out.mkdir(exist_ok=True)
    for benchmark in ("runbugrun", "humanevalpack"):
        path = out / f"M3-long-interim-step-{step}-v1-{benchmark}.jsonl"
        if path.with_suffix(".summary.json").exists():
            continue
        state("evaluating", step=step, benchmark=benchmark)
        with path.with_suffix(".log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "evaluation.round2",
                    "--model",
                    "checkpoints/M1-merged",
                    "--adapter",
                    adapter,
                    "--arm",
                    f"M3-long-interim-step-{step}",
                    "--controller",
                    "v1",
                    "--benchmark",
                    benchmark,
                    "--fp16",
                    "--output",
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    state("completed", step=step)


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
