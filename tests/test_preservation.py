import json
from pathlib import Path
import zipfile
from scripts.preserve_project import checkpoint_state


def create_state(root: Path):
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        (root / name).write_text("owned test fixture")
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]:
        with zipfile.ZipFile(root / name, "w") as file:
            file.writestr("data", "state payload")
    (root / "trainer_state.json").write_text(
        json.dumps({"global_step": 10, "max_steps": 20})
    )


def test_in_progress_binary_cannot_be_called_resumable(tmp_path):
    create_state(tmp_path)
    assert checkpoint_state(tmp_path)["classification"] == "resumable-trainer-state"
    (tmp_path / "optimizer.pt").write_bytes(b"PK truncated write")
    assert checkpoint_state(tmp_path)["classification"] == "incomplete-trainer-state"


def test_invalid_trainer_json_cannot_be_called_resumable(tmp_path):
    create_state(tmp_path)
    (tmp_path / "trainer_state.json").write_text("{partial")
    assert checkpoint_state(tmp_path)["classification"] == "incomplete-trainer-state"
