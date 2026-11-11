"""Evaluate one immutable dev-selected GRPO snapshot locally while training continues."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from repairrl.common import digest
from scripts.round2_train import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    a = parser.parse_args()
    record = json.loads(a.request.read_text())
    step = record["selected"]["step"]
    root = a.request.parent / f"eval-step-{step}"
    root.mkdir(exist_ok=True)

    def state(name, **kw):
        atomic_json(
            root / "status.json",
            dict(state=name, updated_at=time.time(), step=step, **kw),
        )

    try:
        adapter = Path(record["local_adapter"])
        if digest(adapter / "adapter_model.safetensors") != record["adapter_sha256"]:
            raise ValueError("Adapter changed")
        metadata = json.loads((adapter / "round2-checkpoint.json").read_text())
        if (
            digest(Path(record["model"]) / "repairrl-merge.json")
            != metadata["parent_sha256"]
        ):
            raise ValueError("Local parent differs from remote GRPO parent")
        frozen = json.loads(Path("results/round2/frozen-protocol.json").read_text())
        for source, sha in frozen["source_hashes"].items():
            if digest(source) != sha:
                raise ValueError(f"Frozen source changed: {source}")
        for benchmark in ("runbugrun", "humanevalpack"):
            output = root / f"M3-long-interim-step-{step}-v1-{benchmark}.jsonl"
            if output.with_suffix(".summary.json").exists():
                continue
            state("evaluating", benchmark=benchmark)
            with output.with_suffix(".log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "evaluation.round2",
                        "--model",
                        record["model"],
                        "--adapter",
                        str(adapter),
                        "--arm",
                        f"M3-long-interim-step-{step}",
                        "--controller",
                        "v1",
                        "--benchmark",
                        benchmark,
                        "--fp16",
                        "--output",
                        str(output),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        state("completed")
    except BaseException as error:
        state("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
