"""Shared trainer contracts and provenance. Sibling branches require an M1 merge."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
from repairrl.common import load_config, read_jsonl, digest, messages, PROMPT_VERSION


def arguments(arm):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=f"configs/{ {'M1': 'sft', 'M2': 'dpo', 'M3': 'grpo'}[arm] }.yaml")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--output")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    cfg["row_limit"] = a.limit
    if cfg["arm"] != arm:
        p.error(f"Expected {arm} config")
    if a.max_steps is not None:
        cfg["max_steps"] = a.max_steps
    if a.output:
        cfg["output_dir"] = a.output
    return a, cfg


def check_data(cfg, limit=None):
    train = Path(cfg["train_data"])
    gate_path = train.parent / "gate.json"
    gate = json.loads(gate_path.read_text())
    if not gate.get("passed"):
        raise ValueError("NO-GO: data gate not passed")
    for path in (train, Path(cfg["dev_data"]), Path(cfg["tests"])):
        if digest(path) != gate["hashes"].get(path.name):
            raise ValueError(f"Dataset changed since verification: {path}")
    rows, dev = read_jsonl(train), read_jsonl(cfg["dev_data"])
    held = read_jsonl(train.parent / "test.jsonl")
    if digest(train.parent / "test.jsonl") != gate["hashes"].get("test.jsonl"):
        raise ValueError("Held-out manifest changed")
    groups = [{str(r["problem_id"]) for r in records} for records in (rows, dev, held)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Problem IDs leak across splits")
    if not rows or not dev or not held or not all(r.get("verified") for r in rows + dev + held):
        raise ValueError("Require nonempty, execution-verified train/dev/test manifests")
    if limit:
        rows = rows[:limit]
    if len(rows) < cfg.get("minimum_training_rows", 1):
        raise ValueError(f"NO-GO: only {len(rows)} training rows; require {cfg['minimum_training_rows']}")
    cfg["candidate_row_ids"] = [r["id"] for r in rows]
    if cfg["arm"] in {"M2", "M3"}:
        parent = Path(cfg["model"]) / "repairrl-merge.json"
        meta = json.loads(parent.read_text())
        if meta.get("arm") != "M1" or not meta.get("merged"):
            raise ValueError("DPO/GRPO require a separately merged M1 parent")
    return rows, dev


def lora_config(cfg):
    from peft import LoraConfig
    return LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                      lora_dropout=cfg["lora_dropout"], task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def setup_model(cfg):
    from peft import prepare_model_for_kbit_training
    from repairrl.model import load_model
    from transformers import set_seed
    set_seed(cfg["seed"])
    model, tokenizer = load_model(cfg["model"], cfg["revision"], quantized=cfg["load_in_4bit"])
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    return model, tokenizer


def format_rows(rows, tokenizer, cfg, kind):
    from datasets import Dataset
    formatted, dropped, kept_ids = [], 0, []
    for row in rows:
        prompt = tokenizer.apply_chat_template(messages(row["statement"], row["buggy_code"]),
                    tokenize=False, add_generation_prompt=True)
        chosen = row["fixed_code"] + tokenizer.eos_token
        rejected = row["buggy_code"] + tokenizer.eos_token
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if kind == "grpo":
            fits = prompt_len <= cfg["max_prompt_length"]
        else:
            completions = (chosen, rejected) if kind == "dpo" else (chosen,)
            fits = all(len(tokenizer(prompt + code, add_special_tokens=False)["input_ids"]) <= cfg["max_length"] for code in completions)
        if not fits:
            dropped += 1
            continue
        kept_ids.append(row["id"])
        if kind == "sft":
            formatted.append({"prompt": prompt, "completion": chosen})
        elif kind == "dpo":
            formatted.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        else:
            formatted.append({"prompt": prompt, "problem_id": str(row["problem_id"])})
    if len(formatted) < cfg.get("minimum_training_rows", 1):
        raise ValueError("NO-GO: too few training rows after context filtering")
    cfg["training_row_ids"] = kept_ids
    print(f"{kind}: kept {len(formatted)}, excluded {dropped} overlength rows", flush=True)
    if not formatted:
        raise ValueError("No rows fit context; inspect statements and code lengths")
    return Dataset.from_list(formatted)


def trainer_options(cfg):
    return dict(output_dir=cfg["output_dir"], seed=cfg["seed"], data_seed=cfg["seed"],
                per_device_train_batch_size=cfg["per_device_train_batch_size"], per_device_eval_batch_size=1,
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                num_train_epochs=cfg["num_train_epochs"], max_steps=cfg["max_steps"],
                learning_rate=cfg["learning_rate"], fp16=True, bf16=False,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=cfg["logging_steps"], save_steps=cfg["save_steps"], save_total_limit=2,
                report_to="none", optim="adamw_torch", eval_strategy="no")


def run_and_save(trainer, tokenizer, cfg):
    import torch
    output = Path(cfg["output_dir"])
    if (output / "adapter_config.json").exists():
        raise ValueError("Output adapter exists; select a new --output")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    packages = {n: importlib.metadata.version(n) for n in ("torch", "trl", "transformers", "peft", "bitsandbytes", "datasets")}
    meta = {"arm": cfg["arm"], "config": cfg, "prompt_version": PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(), "packages": packages,
            "model_revision": getattr(trainer.model.config, "_commit_hash", None),
            "dataset_hashes": {k: digest(cfg[k]) for k in ("train_data", "dev_data", "tests")},
            "training_seconds": time.monotonic() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "metrics": result.metrics, "log_history": trainer.state.log_history,
            "status": "trained; reload and independent execution evaluation required",
            "preference_kind": "weak-gold-vs-original-bug" if cfg["arm"] == "M2" else None}
    if cfg["arm"] != "M1":
        meta["parent_sha256"] = digest(Path(cfg["model"]) / "repairrl-merge.json")
    (output / "repairrl-training.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("arm", "training_seconds", "peak_allocated_gb", "status")}, indent=2))
