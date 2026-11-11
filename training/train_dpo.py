"""Weak-negative DPO. Frozen reference = merged M1 with the new adapter disabled."""
from training.common import arguments, check_data, setup_model, format_rows, lora_config, trainer_options, run_and_save


def main():
    a, cfg = arguments("M2")
    rows, _ = check_data(cfg, a.limit)
    if a.validate_only:
        print(f"M2 weak-negative contract passed: {len(rows)} rows; parent = {cfg['model']}")
        return
    from trl import DPOConfig, DPOTrainer
    model, tokenizer = setup_model(cfg)
    trainer = DPOTrainer(model=model, ref_model=None,
                        args=DPOConfig(**trainer_options(cfg), beta=cfg["beta"],
                            max_length=cfg["max_length"], max_prompt_length=None,
                            max_completion_length=None, precompute_ref_log_probs=True),
                        train_dataset=format_rows(rows, tokenizer, cfg, "dpo"),
                        processing_class=tokenizer, peft_config=lora_config(cfg))
    run_and_save(trainer, tokenizer, cfg)


if __name__ == "__main__":
    main()
