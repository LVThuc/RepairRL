"""Non-interrupting correction of a queued baseline's old M1-parent path."""

import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path("results/round2")
    while True:
        try:
            if (
                json.loads((root / "baseline-queue.json").read_text())["state"]
                == "completed"
            ):
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(30)
    target = root / "baselines-common-parent"
    target.mkdir(exist_ok=True)
    for controller in ["v1", "v31"]:
        for benchmark in ["runbugrun", "humanevalpack"]:
            output = target / f"M1-{controller}-{benchmark}.jsonl"
            if output.with_suffix(".summary.json").exists():
                continue
            with output.with_suffix(".log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "evaluation.round2",
                        "--arm",
                        "M1",
                        "--model",
                        "checkpoints/M1-merged-round2",
                        "--controller",
                        controller,
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


if __name__ == "__main__":
    main()
