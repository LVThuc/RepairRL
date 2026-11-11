"""Independent GPU queues; dev tuning and automatic selected-checkpoint evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.round2_train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["dpo", "grpo-followup"], required=True)
    a = p.parse_args()
    out = Path("results/round2")
    status = out / (a.kind + "-queue.json")
    def state(name, **extra):
        atomic_json(status, {"state": name, "pid": os.getpid(), "updated_at": time.time(), **extra})
    def trial(name, options):
        destination = out / name
        if (destination / "selection.json").exists() and json.loads((destination / "selection.json").read_text())["state"] == "completed":
            return destination
        state("training", trial=name)
        with (out / (name + ".log")).open("a") as log:
            result = subprocess.run([sys.executable, "-u", "-m", "scripts.round2_train", "--output", str(destination),
                "--resume"] + options, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            text = (out / (name + ".log")).read_text(errors="replace")
            if "out of memory" in text.lower() and "--full-precision" in options:
                fallback = name + "-nf4"
                state("memory_fallback", trial=name, fallback=fallback,
                      note="Fresh M1 branch, NF4 optimization; final ranking still FP16 on dev")
                return trial(fallback, [x for x in options if x != "--full-precision"])
            raise RuntimeError(f"Trial {name} failed; inspect its log")
        return destination
    try:
        trials = []
        if a.kind == "dpo":
            bank = out / "rollouts-fp16"
            deadline = time.monotonic() + 24*3600
            while True:
                state("waiting_for_rollouts", bank=str(bank))
                try:
                    progress = json.loads((bank / "status.json").read_text())
                    if progress["state"] == "completed":
                        break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Rollout bank exceeded 24-hour preparation budget")
                time.sleep(20)
            common = ["--kind", "dpo", "--bank", str(bank), "--epochs", "3", "--dev-tasks", "60", "--full-precision"]
            trials.append(trial("dpo-lr3e6", common + ["--learning-rate", "0.000003", "--beta", "0.1"]))
            trials.append(trial("dpo-lr1e5-rpo", common + ["--learning-rate", "0.00001", "--beta", "0.05", "--rpo-alpha", "0.2"]))
            family = "dpo"
        else:
            dense = out / "grpo-dense"
            deadline = time.monotonic() + 24*3600
            while True:
                state("waiting_for_grpo_dense")
                try:
                    selection = json.loads((dense / "selection.json").read_text())
                    if selection["state"] == "completed":
                        break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("GRPO dense did not complete")
                time.sleep(20)
            trials.append(dense)
            # A predeclared alternative is triggered by dev, never test results.
            if not selection["improved_over_parent"]:
                trials.append(trial("grpo-correctness", ["--kind", "grpo", "--reward", "correctness",
                    "--max-steps", "1200", "--learning-rate", "0.000002", "--beta", "0.02", "--full-precision"]))
            family = "grpo"
        state("reranking_and_evaluating", trials=[str(t) for t in trials])
        subprocess.run([sys.executable, "-u", "-m", "scripts.round2_finish", "--family", family,
                        "--trials"] + [str(t) for t in trials], check=True)
        state("completed", trials=[str(t) for t in trials])
    except BaseException as error:
        state("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
