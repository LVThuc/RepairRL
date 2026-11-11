"""Existing DPO entry point plus lineage-keyed persistent reference-logprob cache."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import trl
from repairrl.common import digest


class CachedDPOTrainer(trl.DPOTrainer):
    def get_train_dataloader(self):
        root = os.environ.get("REPAIRRL_DPO_REF_CACHE")
        if not root:
            return super().get_train_dataloader()
        if not hasattr(self, "_repairrl_reference_key"):
            parent = Path(os.environ["REPAIRRL_DPO_PARENT_MANIFEST"])
            identity = {"dataset": self.train_dataset._fingerprint, "rows": len(self.train_dataset),
                        "parent": digest(parent), "max_length": self.args.max_length,
                        "trl": importlib.metadata.version("trl"), "torch": importlib.metadata.version("torch")}
            self._repairrl_reference_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(root) / self._repairrl_reference_key
        required = {"ref_chosen_logps", "ref_rejected_logps"}
        if not self._precomputed_train_ref_log_probs and (cache / "dataset_info.json").exists():
            from datasets import load_from_disk
            loaded = load_from_disk(str(cache))
            if len(loaded) != len(self.train_dataset) or not required <= set(loaded.column_names):
                raise ValueError("Reference cache shape/columns mismatch")
            self.train_dataset = loaded
            self._precomputed_train_ref_log_probs = True
            print(f"Restored reference log probabilities: {cache}", flush=True)
        loader = super().get_train_dataloader()
        if self._precomputed_train_ref_log_probs and not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(cache.name + ".partial")
            self.train_dataset.save_to_disk(str(temporary))
            temporary.rename(cache)
            print(f"Saved reference log probabilities: {cache}", flush=True)
        return loader


def main():
    trl.DPOTrainer = CachedDPOTrainer
    from training.train_dpo import main as train
    train()


if __name__ == "__main__":
    main()
