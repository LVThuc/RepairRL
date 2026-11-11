"""Wait for successful M1 evaluation, then merge M1, train DPO and evaluate M2."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from repairrl.common import digest, load_config, read_jsonl


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft-status", type=Path, required=True)
    p.add_argument("--config", default="configs/full-dpo.yaml")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    a = p.parse_args()
    cfg = load_config(a.config)
    if cfg["arm"] != "M2" or not 500 <= a.limit <= 1000:
        p.error("Require M2 config and 500–1000 weak preference candidates")
    if a.run_dir.exists():
        p.error("Run directory already exists; use a fresh destination")
    a.run_dir.mkdir(parents=True)
    import yaml
    config_path = a.run_dir / "config.resolved.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    upstream = json.loads(a.sft_status.read_text())
    expected_hashes = upstream["source_hashes"] | {"training/train_dpo.py": digest("training/train_dpo.py")}
    status = {"state": "waiting_for_m1_evaluation", "queued_at": now(),
              "supervisor_pid": os.getpid(), "upstream_status": str(a.sft_status.resolve()),
              "config": str(config_path.resolve()), "config_sha256": digest(config_path),
              "candidate_limit": a.limit, "preference_kind": "weak-gold-vs-original-bug",
              "checkpoint": str(Path(cfg["output_dir"]).resolve()), "evaluations": []}
    status_path = a.run_dir / "status.json"
    def save():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(status_path)
    save()
    Path("results/active-dpo.json").write_text(json.dumps({"status": str(status_path.resolve()),
                                                         "supervisor_pid": os.getpid()}, indent=2))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               REPAIRRL_DPO_REF_CACHE=str((a.run_dir / "reference-cache").resolve()),
               REPAIRRL_DPO_PARENT_MANIFEST=str((Path(cfg["model"]) / "repairrl-merge.json").resolve()))
    status["allocator"] = env["PYTORCH_CUDA_ALLOC_CONF"]
    status["reference_cache"] = env["REPAIRRL_DPO_REF_CACHE"]
    save()
    def frozen():
        if any(digest(path) != value for path, value in expected_hashes.items()):
            raise ValueError("Experiment code changed since the M0/M1 run")
        if any(digest(cfg[key]) != value for key, value in upstream["dataset_hashes"].items()):
            raise ValueError("Frozen SFT/DPO dataset changed")
    def stage(name, command):
        log_path = a.run_dir / f"{name}.log"
        with log_path.open("x") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=env)
            status.update(state=name, child_pid=child.pid, log=str(log_path.resolve()),
                          stage_started_at=now())
            save()
            code = child.wait()
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {log_path}")
    try:
        deadline = time.monotonic() + 24 * 3600
        while True:
            upstream = json.loads(a.sft_status.read_text())
            if upstream["state"] == "completed":
                break
            if upstream["state"] == "failed":
                raise RuntimeError(f"M1 pipeline failed: {upstream.get('error')}")
            if time.monotonic() > deadline:
                raise TimeoutError("M1 pipeline did not complete within 24 hours")
            if not Path(f"/proc/{upstream['supervisor_pid']}").exists():
                raise RuntimeError("M1 supervisor stopped before successful evaluation")
            time.sleep(15)
        if len(upstream.get("post_sft_evaluations", [])) != 2 or upstream.get("reload_exit_code") != 0:
            raise ValueError("M1 evaluation/reload gate incomplete")
        frozen()
        adapter = Path(upstream["checkpoint"])
        sft_meta = json.loads((adapter / "repairrl-training.json").read_text())
        if sft_meta["arm"] != "M1" or len(sft_meta["config"]["training_row_ids"]) < 1000:
            raise ValueError("DPO requires the full M1, not the integration pilot")
        merged = Path(cfg["model"])
        if merged.exists():
            parent = json.loads((merged / "repairrl-merge.json").read_text())
            if parent["adapter_sha256"] != digest(adapter / "adapter_model.safetensors") or parent["training_sha256"] != digest(adapter / "repairrl-training.json"):
                raise ValueError("Existing M1 merge belongs to a different SFT checkpoint")
        else:
            stage("merging_m1", [sys.executable, "-u", "-m", "scripts.merge_sft",
                                 "--adapter", str(adapter), "--output", str(merged)])
        output = Path(cfg["output_dir"])
        if output.exists() and any(output.iterdir()):
            raise ValueError("DPO checkpoint directory contains files; refusing overwrite")
        status.update(parent_sha256=digest(merged / "repairrl-merge.json"),
                      dataset_hashes=upstream["dataset_hashes"])
        save()
        stage("validating_dpo", [sys.executable, "-u", "-m", "training.train_dpo",
              "--config", str(config_path), "--limit", str(a.limit), "--validate-only"])
        try:
            stage("training_dpo", [sys.executable, "-u", "-m", "scripts.train_dpo_cached",
                  "--config", str(config_path), "--limit", str(a.limit)])
        except RuntimeError:
            log_text = Path(status["log"]).read_text(errors="replace")
            if "out of memory" not in log_text.lower() and "OutOfMemoryError" not in log_text:
                raise
            # A smaller context is an explicit, logged training-only fallback;
            # evaluation budgets and M1 parent remain fixed. Targets are filtered,
            # never truncated. The failed attempt is preserved.
            cfg["max_length"] = 1536
            output = Path("checkpoints/M2-lowmem")
            cfg["output_dir"] = str(output)
            config_path = a.run_dir / "config.lowmem.yaml"
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            status.update(memory_fallback=True, checkpoint=str(output.resolve()),
                          fallback_config=str(config_path.resolve()))
            save()
            stage("training_dpo_lowmem", [sys.executable, "-u", "-m", "scripts.train_dpo_cached",
                  "--config", str(config_path), "--limit", str(a.limit)])
        stage("checking_reload", [sys.executable, "-u", "-m", "scripts.reload_smoke",
              "--adapter", str(output), "--data", cfg["dev_data"], "--tests", cfg["tests"],
              "--output", str(a.run_dir / "reload.json")])
        sft_run = a.sft_status.parent
        for benchmark in ("runbugrun", "humanevalpack"):
            frozen()
            reference_path = sft_run / f"baseline-M0-{benchmark}.summary.json"
            protocol = json.loads(reference_path.read_text())["protocol"]
            result = a.run_dir / f"M2-{benchmark}.jsonl"
            stage(f"evaluating_m2_{benchmark}", [sys.executable, "-u", "-m", "evaluation.benchmark",
                  "--arm", "M2", "--model", str(merged), "--adapter", str(output),
                  "--revision", cfg["revision"], "--benchmark", benchmark, "--data", protocol["data"],
                  "--tests", protocol["tests"], "--max-new-tokens", str(protocol["max_new_tokens"]),
                  "--max-context", str(protocol["max_context"]), "--seed", str(protocol["seed"]),
                  "--timeout", str(protocol["timeout"]), "--output", str(result)])
            summary_path = result.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text())
            if summary["metrics"]["n"] != len(read_jsonl(protocol["data"])):
                raise ValueError("Incomplete M2 evaluation")
            status["evaluations"].append({"benchmark": benchmark, "summary": str(summary_path.resolve()),
                                           "metrics": summary["metrics"]})
            save()
            stage(f"comparing_{benchmark}", [sys.executable, "-m", "evaluation.compare", str(reference_path),
                  str(sft_run / f"post-sft-M1-{benchmark}.summary.json"), str(summary_path)])
        status.update(state="completed", finished_at=now(),
                      note="M0/M1/M2 comparisons available; inspect measured results")
        save()
    except BaseException as e:
        status.update(state="failed", error=f"{type(e).__name__}: {e}", finished_at=now())
        save()
        raise


if __name__ == "__main__":
    main()
