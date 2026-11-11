"""Execution-reward GRPO, gated by measured reward variance on M1 rollouts."""
import json
from pathlib import Path
from training.common import arguments, check_data, setup_model, format_rows, lora_config, trainer_options, run_and_save
from repairrl.common import digest


def main():
    a, cfg = arguments("M3")
    rows, _ = check_data(cfg, a.limit)
    report_path = Path(cfg["variance_report"])
    if not report_path.exists():
        raise SystemExit(f"NO-GO: variance report {report_path} missing; complete evaluation.variance first")
    report = json.loads(report_path.read_text())
    if not report.get("passed") or report["train_sha256"] != digest(cfg["train_data"]) or report["tests_sha256"] != digest(cfg["tests"]):
        raise ValueError("NO-GO: missing, failed or stale GRPO variance gate")
    if report["parent_sha256"] != digest(Path(cfg["model"]) / "repairrl-merge.json"):
        raise ValueError("Variance measured on a different M1 parent")
    if len(report.get("groups", [])) < 30 or report.get("rollout_config") != {
        k: cfg[k] for k in ("seed", "max_completion_length", "max_prompt_length", "sandbox_timeout")}:
        raise ValueError("Variance gate population or rollout settings do not match")
    if a.validate_only:
        print(f"M3 data and variance gates passed: {len(rows)} rows")
        return
    from trl import GRPOConfig, GRPOTrainer
    from evaluation.reward import ExecutionReward
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    suites = json.loads(Path(cfg["tests"]).read_text())
    model, tokenizer = setup_model(cfg)
    trainer = GRPOTrainer(model=model,
              args=GRPOConfig(**trainer_options(cfg), num_generations=cfg["num_generations"],
                  max_prompt_length=cfg["max_prompt_length"], max_completion_length=cfg["max_completion_length"],
                  beta=cfg["beta"], temperature=0.8, top_p=0.8, top_k=20, repetition_penalty=1.1, use_vllm=False, remove_unused_columns=False,
                  mask_truncated_completions=True),
              train_dataset=format_rows(rows, tokenizer, cfg, "grpo"), processing_class=tokenizer,
              reward_funcs=ExecutionReward(suites, str(out / "rewards.jsonl"), cfg["sandbox_timeout"]),
              peft_config=lora_config(cfg))
    run_and_save(trainer, tokenizer, cfg)


if __name__ == "__main__":
    main()
