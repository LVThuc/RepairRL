from training.common import arguments, check_data, setup_model, format_rows, lora_config, trainer_options, run_and_save


def main():
    a, cfg = arguments("M1")
    rows, _ = check_data(cfg, a.limit)
    if a.validate_only:
        print(f"M1 data contract passed: {len(rows)} rows")
        return
    from trl import SFTConfig, SFTTrainer
    model, tokenizer = setup_model(cfg)
    dataset = format_rows(rows, tokenizer, cfg, "sft")
    trainer = SFTTrainer(model=model, args=SFTConfig(**trainer_options(cfg),
                         max_length=cfg["max_length"], completion_only_loss=True,
                         packing=False, dataset_kwargs={"add_special_tokens": False}),
                         train_dataset=dataset, processing_class=tokenizer, peft_config=lora_config(cfg))
    # Fail early if prompt/completion masking yields no response labels.
    batch = next(iter(trainer.get_train_dataloader()))
    if not (batch["labels"] != -100).any() or not (batch["labels"] == -100).any():
        raise ValueError("Response-only loss mask failed")
    run_and_save(trainer, tokenizer, cfg)


if __name__ == "__main__":
    main()
