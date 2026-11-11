"""Supervise M0 baselines -> fresh SFT -> reload -> identical M1 evaluations."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from repairrl.common import digest, load_config, read_jsonl
from training.common import check_data


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/full-sft.yaml")
    p.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    if cfg["arm"] != "M1" or cfg["model"] != "Qwen/Qwen2.5-Coder-1.5B-Instruct":
        p.error("Full SFT must start independently from the untouched Qwen base")
    if cfg.get("minimum_training_rows", 0) < 1000:
        p.error("Full-run config must require at least 1000 verified training examples")
    rows, dev = check_data(cfg)
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        p.error("Output contains files; choose a fresh checkpoint directory")
    a.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = a.run_dir / "status.json"
    if status_path.exists():
        p.error("Run directory already has a job; use a fresh directory")
    import yaml
    frozen_config = a.run_dir / "config.resolved.yaml"
    frozen_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    held_path = Path(cfg["train_data"]).parent / "test.jsonl"
    held = read_jsonl(held_path)
    external = Path("data/external/humanevalpack.jsonl")
    external_gate = json.loads(Path("results/humanevalpack-gate.json").read_text())
    if not external_gate.get("passed") or external_gate["data_sha256"] != digest(external):
        p.error("HumanEvalPack gold sanity gate missing or stale")
    overlap = json.loads(Path("results/scale-overlap-audit.json").read_text())
    if not overlap.get("passed") or overlap["train_sha256"] != digest(cfg["train_data"]) or overlap["external_sha256"] != digest(external):
        p.error("Full training overlap audit missing, failed or stale")
    source_paths = ["repairrl/common.py", "repairrl/model.py", "agent/repair.py", "agent/sandbox.py",
                    "evaluation/runner.py", "evaluation/humanevalpack_adapter.py", "evaluation/benchmark.py",
                    "training/common.py", "training/train_sft.py"]
    status = {"state": "starting", "started_at": utc_now(), "supervisor_pid": os.getpid(),
              "config": str(frozen_config.resolve()), "config_sha256": digest(frozen_config),
              "train_rows": len(rows), "dev_rows": len(dev), "test_rows": len(held),
              "heldout_sha256": digest(held_path), "external_sha256": digest(external),
              "pre_sft_baselines": [], "post_sft_evaluations": [],
              "checkpoint": str(output.resolve()), "seed": cfg["seed"],
              "dataset_hashes": {k: digest(cfg[k]) for k in ("train_data", "dev_data", "tests")},
              "source_hashes": {name: digest(name) for name in source_paths},
              "requirements_sha256": digest("requirements.lock"),
              "train_log": str((a.run_dir / "train.log").resolve())}
    def save_status():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(status_path)
    def check_frozen():
        check_data(cfg)
        paths = status["source_hashes"] | {str(held_path): status["heldout_sha256"],
                 str(external): status["external_sha256"]} | {cfg[k]: v for k, v in status["dataset_hashes"].items()}
        if any(digest(path) != value for path, value in paths.items()):
            raise ValueError("Frozen data or experiment code changed during the run")
    save_status()
    Path("results/active-sft.json").write_text(json.dumps({"status": str(status_path.resolve()),
                                                         "supervisor_pid": os.getpid()}, indent=2))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    specs = [("runbugrun", held_path), ("humanevalpack", external)]
    def evaluate(arm):
        for benchmark, data_path in specs:
            check_frozen()
            name = f"{'baseline' if arm == 'M0' else 'post-sft'}-{arm}-{benchmark}"
            result_path = a.run_dir / f"{name}.jsonl"
            command = [sys.executable, "-u", "-m", "evaluation.benchmark", "--arm", arm,
                       "--model", cfg["model"], "--revision", cfg["revision"],
                       "--benchmark", benchmark, "--data", str(data_path), "--tests", cfg["tests"],
                       "--max-new-tokens", str(cfg["max_new_tokens"]), "--max-context", str(cfg["max_length"]),
                       "--seed", str(cfg["seed"]), "--timeout", str(cfg["sandbox_timeout"]), "--output", str(result_path)]
            if arm == "M1":
                command += ["--adapter", str(output)]
            log_path = a.run_dir / f"{name}.log"
            with log_path.open("x") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, env=env)
                status.update(state=f"{'baseline' if arm == 'M0' else 'post_sft'}_{benchmark}",
                              evaluation_pid=child.pid, evaluation_log=str(log_path.resolve()))
                save_status()
                code = child.wait()
            if code:
                raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")
            summary_path = result_path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text())
            if summary["protocol"]["data_sha256"] != digest(data_path) or summary["metrics"]["n"] != len(read_jsonl(data_path)):
                raise ValueError("Incomplete or stale execution evaluation")
            key = "pre_sft_baselines" if arm == "M0" else "post_sft_evaluations"
            status[key].append({"benchmark": benchmark, "summary": str(summary_path.resolve()), "metrics": summary["metrics"]})
            save_status()
    started = time.monotonic()
    try:
        evaluate("M0")
        check_frozen()
        with (a.run_dir / "train.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "training.train_sft", "--config", str(frozen_config)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
            status.update(state="training", training_pid=child.pid, training_started_at=utc_now())
            save_status()
            code = child.wait()
        status.update(training_exit_code=code, training_finished_at=utc_now())
        if code:
            raise RuntimeError(f"SFT exited with code {code}; see train.log")
        status["state"] = "checking_reload"
        save_status()
        with (a.run_dir / "reload.log").open("x") as log:
            check = subprocess.run([sys.executable, "-u", "-m", "scripts.reload_smoke",
                "--adapter", str(output), "--data", cfg["dev_data"], "--tests", cfg["tests"],
                "--output", str(a.run_dir / "reload.json")], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, env=env)
        status["reload_exit_code"] = check.returncode
        if check.returncode:
            raise RuntimeError("Reload/valid-code check failed; inspect reload.log before continuing")
        evaluate("M1")
        for benchmark, _ in specs:
            with (a.run_dir / f"comparison-{benchmark}.md").open("x") as report:
                subprocess.run([sys.executable, "-m", "evaluation.compare",
                    str(a.run_dir / f"baseline-M0-{benchmark}.summary.json"),
                    str(a.run_dir / f"post-sft-M1-{benchmark}.summary.json")],
                    stdout=report, stderr=subprocess.STDOUT, check=True)
        status.update(state="completed", finished_at=utc_now(), elapsed_seconds=time.monotonic() - started,
                      note="Measured M0/M1 comparison complete; inspect results, improvement is not assumed")
        save_status()
    except BaseException as e:
        status.update(state="failed", error=f"{type(e).__name__}: {e}", finished_at=utc_now())
        save_status()
        raise


if __name__ == "__main__":
    main()
