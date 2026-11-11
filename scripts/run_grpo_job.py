"""Overnight GRPO supervisor: GPU dependency -> variance -> pilot -> train -> eval.

M3 always branches from full M1. DPO failure does not invalidate that parent.
All processes and deadlines are explicit; a scientific gate failure is recorded.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from repairrl.common import digest, load_config, read_jsonl


def now():
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid):
    if not pid:
        return False
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--after-status", type=Path, required=True)
    p.add_argument("--sft-status", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", default="configs/full-grpo.yaml")
    a = p.parse_args()
    if a.run_dir.exists():
        p.error("Run directory exists; choose a fresh path")
    a.run_dir.mkdir(parents=True)
    status_path = a.run_dir / "status.json"
    cfg = load_config(a.config)
    status = {"state": "waiting_for_gpu", "queued_at": now(), "supervisor_pid": os.getpid(),
              "after_status": str(a.after_status.resolve()), "sft_status": str(a.sft_status.resolve()),
              "training_steps": cfg["max_steps"], "pilot_steps": 20, "events": [], "evaluations": [],
              "supervisor_source_sha256": digest(__file__)}
    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(status_path)
    save()
    Path("results/active-grpo.json").write_text(json.dumps({"status": str(status_path.resolve()),
                                                         "supervisor_pid": os.getpid()}, indent=2))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    def stage(name, command, timeout=5400):
        logfile = a.run_dir / f"{name}.log"
        with logfile.open("x") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
            status.update(state=name, child_pid=child.pid, log=str(logfile.resolve()), stage_started_at=now())
            save()
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                code = 124
        status["events"].append({"stage": name, "exit_code": code, "finished_at": now(), "log": str(logfile.resolve())})
        save()
        return code, logfile
    def required(name, command, timeout=5400):
        code, log = stage(name, command, timeout)
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {log}")
    import yaml
    def write_config(label):
        path = a.run_dir / f"{label}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return str(path)
    def variance(label, generations):
        cfg["num_generations"] = generations
        cfg["gradient_accumulation_steps"] = generations
        cfg["variance_report"] = str(a.run_dir / f"variance-{label}.json")
        config = write_config(f"config-{label}")
        code, log = stage(f"variance-{label}", [sys.executable, "-u", "-m", "evaluation.variance",
            "--config", config, "--tasks", "30", "--generations", str(generations),
            "--output", cfg["variance_report"]])
        report_path = Path(cfg["variance_report"])
        if report_path.exists():
            report = json.loads(report_path.read_text())
            status["variance"] = {"report": str(report_path.resolve()), "passed": report.get("passed"),
                                  "mixed_group_fraction": report.get("mixed_group_fraction")}
            save()
            if code == 0 and report.get("passed"):
                return True
            if report.get("passed") is False:
                return False
        raise RuntimeError(f"Variance execution failed ({code}); inspect {log}")
    try:
        deadline = time.monotonic() + 24 * 3600
        while True:
            upstream = json.loads(a.after_status.read_text())
            terminal = upstream["state"] in {"completed", "failed"}
            dead_parent = not process_alive(upstream.get("supervisor_pid"))
            if (terminal or dead_parent) and not process_alive(upstream.get("child_pid")):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("GPU dependency did not finish within 24 hours")
            time.sleep(15)
        sft = json.loads(a.sft_status.read_text())
        if sft["state"] != "completed":
            raise ValueError("Full M1 training/evaluation prerequisite is incomplete")
        for key, value in sft["dataset_hashes"].items():
            if digest(cfg[key]) != value:
                raise ValueError(f"Frozen dataset changed: {key}")
        for path, value in sft["source_hashes"].items():
            if digest(path) != value:
                raise ValueError(f"Frozen model/evaluator source changed: {path}")
        adapter = Path(sft["checkpoint"])
        merged = Path(cfg["model"])
        if not merged.exists():
            required("merge-m1", [sys.executable, "-u", "-m", "scripts.merge_sft",
                                  "--adapter", str(adapter), "--output", str(merged)], 600)
        parent = json.loads((merged / "repairrl-merge.json").read_text())
        if parent["adapter_sha256"] != digest(adapter / "adapter_model.safetensors") or parent["training_sha256"] != digest(adapter / "repairrl-training.json"):
            raise ValueError("GRPO parent is not the completed full M1")
        status.update(parent_sha256=digest(merged / "repairrl-merge.json"),
                      parent_model=str(merged.resolve()), dataset_hashes=sft["dataset_hashes"])
        save()
        if not variance("g4", 4) and not variance("g8", 8):
            status.update(state="gate_failed", finished_at=now(),
                          reason="Fewer than 20% mixed-reward groups with both 4 and 8 generations; RL not forced")
            save()
            return
        config = write_config("config-pilot")
        required("validate-grpo", [sys.executable, "-u", "-m", "training.train_grpo",
                                   "--config", config, "--validate-only"], 120)
        pilot = Path("checkpoints/grpo-preflight/M3")
        code, log = stage("pilot", [sys.executable, "-u", "-m", "training.train_grpo",
            "--config", config, "--max-steps", "20", "--output", str(pilot)], 2700)
        if code:
            text = log.read_text(errors="replace")
            if "out of memory" not in text.lower() and "OutOfMemoryError" not in text:
                # One fresh retry handles a transient execution error without
                # mixing partial checkpoint or reward artifacts into the run.
                pilot = Path("checkpoints/grpo-preflight/M3-retry")
                required("pilot-retry", [sys.executable, "-u", "-m", "training.train_grpo",
                    "--config", config, "--max-steps", "20", "--output", str(pilot)], 2700)
            else:
                cfg["max_completion_length"] = 256
                cfg["num_generations"] = cfg["gradient_accumulation_steps"] = 4
                if not variance("lowmem-g4", 4):
                    status.update(state="gate_failed", reason="Low-memory reward variance gate failed", finished_at=now())
                    save()
                    return
                config = write_config("config-lowmem")
                pilot = Path("checkpoints/grpo-preflight/M3-lowmem")
                required("pilot-lowmem", [sys.executable, "-u", "-m", "training.train_grpo",
                    "--config", config, "--max-steps", "20", "--output", str(pilot)], 2700)
        metadata = json.loads((pilot / "repairrl-training.json").read_text())
        from safetensors import safe_open
        with safe_open(str(pilot / "adapter_model.safetensors"), framework="pt", device="cpu") as weights:
            b_tensors = [weights.get_tensor(key) for key in weights.keys() if "lora_B" in key]
            import torch
            updated = any(bool(tensor.count_nonzero()) for tensor in b_tensors)
            finite = all(bool(torch.isfinite(tensor).all()) for tensor in b_tensors)
        status["pilot_checks"] = {"updated_lora_weights": updated, "finite_adapter": finite,
                                  "training_seconds": metadata["training_seconds"],
                                  "peak_allocated_gb": metadata["peak_allocated_gb"]}
        save()
        if not updated or not finite:
            status.update(state="gate_failed", reason="Pilot produced no finite nonzero LoRA updates", finished_at=now())
            save()
            return
        required("pilot-reload", [sys.executable, "-u", "-m", "scripts.reload_smoke",
            "--adapter", str(pilot), "--data", cfg["dev_data"], "--tests", cfg["tests"],
            "--output", str(a.run_dir / "pilot-reload.json")], 300)
        config = write_config("config-full")
        output = Path(cfg["output_dir"])
        code, _ = stage("train-grpo", [sys.executable, "-u", "-m", "training.train_grpo",
                                     "--config", config, "--output", str(output)], 10800)
        if code:
            # One clean retry, preserving the failed run for inspection.
            output = Path("checkpoints/M3-retry")
            required("train-grpo-retry", [sys.executable, "-u", "-m", "training.train_grpo",
                     "--config", config, "--output", str(output)], 10800)
        status["checkpoint"] = str(output.resolve())
        save()
        required("reload-grpo", [sys.executable, "-u", "-m", "scripts.reload_smoke",
            "--adapter", str(output), "--data", cfg["dev_data"], "--tests", cfg["tests"],
            "--output", str(a.run_dir / "reload.json")], 300)
        for benchmark in ("runbugrun", "humanevalpack"):
            reference = a.sft_status.parent / f"baseline-M0-{benchmark}.summary.json"
            protocol = json.loads(reference.read_text())["protocol"]
            successful = None
            for attempt in range(2):
                result = a.run_dir / f"M3-{benchmark}{'-retry' if attempt else ''}.jsonl"
                code, _ = stage(f"evaluate-{benchmark}-{attempt}", [sys.executable, "-u", "-m", "evaluation.benchmark",
                    "--arm", "M3", "--model", str(merged), "--adapter", str(output),
                    "--revision", cfg["revision"], "--benchmark", benchmark, "--data", protocol["data"],
                    "--tests", protocol["tests"], "--max-new-tokens", str(protocol["max_new_tokens"]),
                    "--max-context", str(protocol["max_context"]), "--seed", str(protocol["seed"]),
                    "--timeout", str(protocol["timeout"]), "--output", str(result)], 10800)
                if code == 0:
                    successful = result.with_suffix(".summary.json")
                    break
            if successful is None:
                raise RuntimeError(f"Both {benchmark} evaluation attempts failed")
            summary = json.loads(successful.read_text())
            if summary["metrics"]["n"] != len(read_jsonl(protocol["data"])):
                raise ValueError("Incomplete M3 evaluation")
            status["evaluations"].append({"benchmark": benchmark, "summary": str(successful.resolve()), "metrics": summary["metrics"]})
            save()
            comparisons = [str(reference), str(a.sft_status.parent / f"post-sft-M1-{benchmark}.summary.json")]
            m2 = a.after_status.parent / f"M2-{benchmark}.summary.json"
            if m2.exists():
                comparisons.append(str(m2))
            required(f"compare-{benchmark}", [sys.executable, "-m", "evaluation.compare", *comparisons, str(successful)], 60)
        status.update(state="completed", finished_at=now(),
                      note="Inspect measured results; finishing GRPO does not establish improvement")
        save()
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        save()
        raise


if __name__ == "__main__":
    main()
