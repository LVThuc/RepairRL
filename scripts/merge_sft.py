"""Merge M1 once at full precision on CPU; M2 and M3 independently load this root."""
import argparse
import json
from pathlib import Path
from repairrl.common import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="checkpoints/M1")
    p.add_argument("--output", default="checkpoints/M1-merged")
    a = p.parse_args()
    meta_path = Path(a.adapter) / "repairrl-training.json"
    meta = json.loads(meta_path.read_text())
    if meta["arm"] != "M1":
        p.error("Only M1 can be the common branch root")
    out = Path(a.output)
    if out.exists():
        p.error("Destination exists; use a fresh output directory")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    cfg = meta["config"]
    revision = meta["model_revision"] or cfg["revision"]
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=revision,
                  torch_dtype=torch.float16, device_map="cpu", trust_remote_code=False)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload(safe_merge=True)
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(out)
    (out / "repairrl-merge.json").write_text(json.dumps({"arm": "M1", "merged": True,
        "adapter": str(Path(a.adapter).resolve()), "adapter_sha256": digest(Path(a.adapter) / "adapter_model.safetensors"),
        "training_sha256": digest(meta_path), "base": cfg["model"], "revision": revision}, indent=2))
    print(f"M1 merged to {out}; fresh adapters will branch from this checkpoint")


if __name__ == "__main__":
    main()
