"""Fresh M0 SFT with periodic dev monitoring and resumable optimizer saves."""

import json
from pathlib import Path
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from training.common import (
    arguments,
    check_data,
    setup_model,
    format_rows,
    lora_config,
    trainer_options,
    run_and_save,
)
from evaluation.round2 import dev_rows, attached_generator, evaluate
from repairrl.common import read_jsonl, digest
from scripts.round2_train import atomic_json


def main():
    a, cfg = arguments("M1")
    rows, _ = check_data(cfg)
    if len(rows) != 10000:
        raise ValueError("Require exactly 10000 verified pairs")
    if a.validate_only:
        return
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    model, tokenizer = setup_model(cfg)
    dataset = format_rows(rows, tokenizer, cfg, "sft")
    provenance = {
        "config": cfg,
        "hashes": {k: digest(cfg[k]) for k in ("train_data", "dev_data", "tests")},
    }
    manifest = root / "sft10k-plan.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise ValueError("Training inputs changed during resume")
    atomic_json(manifest, provenance)

    class Monitor(TrainerCallback):
        def score(self, state, model):
            records = (
                read_jsonl(root / "dev-history.jsonl")
                if (root / "dev-history.jsonl").exists()
                else []
            )
            if any(r["step"] == state.global_step for r in records):
                return
            training = model.training
            model.eval()
            metrics = evaluate(
                dev_rows(60), attached_generator(model, tokenizer), "single"
            )
            model.train(training)
            dest = root / "snapshots" / f"step-{state.global_step}"
            model.save_pretrained(dest)
            tokenizer.save_pretrained(dest)
            record = {
                "step": state.global_step,
                "metrics": metrics,
                "adapter": str(dest),
                "precision": "NF4 live training model; final held-out inference FP16",
            }
            with (root / "dev-history.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            trained = [r for r in records + [record] if r["step"] > 0]
            best = max(
                trained or [record],
                key=lambda r: (
                    r["metrics"]["pass@1"],
                    r["metrics"]["partial_fraction"],
                    -r["step"],
                ),
            )
            atomic_json(
                root / "selection.json",
                {
                    "state": "training",
                    "selected": best,
                    "selection_set": "fixed 60-problem dev; never held-out",
                },
            )
            print("DEV", json.dumps(record), flush=True)

        def on_train_begin(self, args, state, control, model=None, **kw):
            self.score(state, model)

        def on_save(self, args, state, control, model=None, **kw):
            self.score(state, model)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            **trainer_options(cfg),
            max_length=cfg["max_length"],
            completion_only_loss=True,
            packing=False,
            dataset_kwargs={"add_special_tokens": False},
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
        callbacks=[Monitor()],
    )
    batch = next(iter(trainer.get_train_dataloader()))
    if not (batch["labels"] != -100).any() or not (batch["labels"] == -100).any():
        raise ValueError("Response-only mask failed")
    from transformers.trainer_utils import get_last_checkpoint

    checkpoint = get_last_checkpoint(str(root))
    if checkpoint:
        original = trainer.train

        def resume(*args, **kwargs):
            kwargs["resume_from_checkpoint"] = checkpoint
            return original(*args, **kwargs)

        trainer.train = resume
    run_and_save(trainer, tokenizer, cfg)
    selection = json.loads((root / "selection.json").read_text())
    selection["state"] = "completed"
    atomic_json(root / "selection.json", selection)


if __name__ == "__main__":
    main()
