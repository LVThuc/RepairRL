from types import SimpleNamespace
import trl
from datasets import Dataset
from scripts.train_dpo_cached import CachedDPOTrainer


def test_reference_cache_reuses_only_matching_parent_and_dataset(tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    parent.write_text('{"adapter":"first"}')
    monkeypatch.setenv("REPAIRRL_DPO_REF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("REPAIRRL_DPO_PARENT_MANIFEST", str(parent))
    calls = []
    def upstream(self):
        if not self._precomputed_train_ref_log_probs:
            calls.append(True)
            self.train_dataset = self.train_dataset.add_column("ref_chosen_logps", [-1.0] * len(self.train_dataset))
            self.train_dataset = self.train_dataset.add_column("ref_rejected_logps", [-2.0] * len(self.train_dataset))
            self._precomputed_train_ref_log_probs = True
        return "loader"
    monkeypatch.setattr(trl.DPOTrainer, "get_train_dataloader", upstream)
    original = Dataset.from_list([{"prompt_input_ids": [1], "chosen_input_ids": [2], "rejected_input_ids": [3]}])
    def trainer():
        value = object.__new__(CachedDPOTrainer)
        value.train_dataset = original
        value.args = SimpleNamespace(max_length=2048)
        value._precomputed_train_ref_log_probs = False
        return value
    first = trainer()
    assert first.get_train_dataloader() == "loader"
    second = trainer()
    assert second.get_train_dataloader() == "loader"
    assert len(calls) == 1
    assert second.train_dataset[0]["ref_chosen_logps"] == -1.0
    parent.write_text('{"adapter":"different"}')
    third = trainer()
    third.get_train_dataloader()
    assert len(calls) == 2
