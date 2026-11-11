"""Resume original round-1 trainers from complete, trusted HF trainer states.

Never unpickle arbitrary downloads. This entry point only validates our inventory
and delegates restoration to the pinned Transformers Trainer implementation.
"""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from repairrl.common import digest
from scripts.preserve_project import checkpoint_state


def prepare(metadata, checkpoint, output, max_steps=None):
    meta = json.loads(Path(metadata).read_text())
    cfg = dict(meta["config"])
    audit = checkpoint_state(Path(checkpoint))
    if audit["classification"] != "resumable-trainer-state":
        raise ValueError(f"Incomplete trainer checkpoint: {audit}")
    state = json.loads((Path(checkpoint)/"trainer_state.json").read_text())
    target = max_steps if max_steps is not None else state["max_steps"]
    if state["global_step"] >= target:
        raise ValueError("This state has already finished its budget; use an earlier checkpoint or explicitly extend --max-steps")
    for key, sha in meta["dataset_hashes"].items():
        if digest(cfg[key]) != sha:
            raise ValueError(f"Dataset hash mismatch: {key}")
    if meta["arm"] != "M1" and digest(Path(cfg["model"])/"repairrl-merge.json") != meta["parent_sha256"]:
        raise ValueError("Parent differs from the recorded M1 merge; restore the original parent")
    cfg.update(output_dir=str(output), max_steps=target, resume_from_checkpoint=str(Path(checkpoint).resolve()),
               resume_source_sha256=digest(metadata), resume_kind="extension" if target != state["max_steps"] else "interrupted-budget")
    if Path(output).exists() and any(Path(output).iterdir()):
        raise ValueError("Choose a fresh output directory; never overwrite a historical run")
    return meta, cfg, audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    meta, cfg, audit = prepare(a.metadata,a.checkpoint,a.output,a.max_steps)
    from training import common
    rows, _ = common.check_data(cfg,cfg.get("row_limit"))
    summary = {"arm":meta["arm"],"checkpoint":a.checkpoint,"state":audit,"rows":len(rows),
               "target_steps":cfg["max_steps"],"resume_kind":cfg["resume_kind"],
               "gpu_resume_executed":False,"note":"Use original pinned environment, one visible GPU, and trusted states only"}
    print(json.dumps(summary,indent=2))
    if a.validate_only:
        return
    from transformers import Trainer
    original = Trainer.train
    def train_resuming(self,*args,**kwargs):
        kwargs["resume_from_checkpoint"] = cfg["resume_from_checkpoint"]
        return original(self,*args,**kwargs)
    Trainer.train = train_resuming
    common.arguments = lambda arm: (SimpleNamespace(validate_only=False,limit=cfg.get("row_limit")),cfg)
    module = {"M1":"training.train_sft","M2":"training.train_dpo","M3":"training.train_grpo"}[meta["arm"]]
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
