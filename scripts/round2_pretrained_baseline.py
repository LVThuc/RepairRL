"""Download an untouched pretrained comparator, then evaluate after local M1 jobs."""
import json
from pathlib import Path
import time

from huggingface_hub import snapshot_download

from repairrl.common import digest, read_jsonl
from scripts.round2_train import atomic_json

MODEL = "Qwen/Qwen2.5-Coder-1.5B"
REVISION = "df3ce67c0e24480f20468b6ef2894622d69eb73b"
ROOT = Path("results/round2/pretrained-base")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    def status(state, **details):
        atomic_json(ROOT / "status.json", dict(state=state, updated_at=time.time(), **details))
    status("downloading", model=MODEL, revision=REVISION)
    snapshot = snapshot_download(MODEL, revision=REVISION,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    atomic_json(ROOT / "model-inventory.json", {
        "model": MODEL, "revision": REVISION, "training_stage": "pretraining",
        "files": [{"name": f.name, "bytes": f.stat().st_size, "sha256": digest(f)}
                  for f in sorted(Path(snapshot).iterdir()) if f.is_file()],
    })
    pending = [Path("results/round2/baselines-common-parent") / f"M1-{c}-{b}.summary.json"
               for c in ("v1",) for b in ("runbugrun", "humanevalpack")]
    while not all(p.exists() for p in pending):
        status("waiting_for_local_M1_evaluation", pending=[str(p) for p in pending if not p.exists()])
        time.sleep(30)
    # Keep the existing round-2 evaluator/generator unchanged: their hashes are frozen.
    from evaluation.round2 import evaluate
    from evaluation.grading_service import GradingClient
    from repairrl.model import Generator
    import torch
    torch.set_num_threads(4)
    frozen = json.loads(Path("results/round2/frozen-protocol.json").read_text())
    for controller in ("v1",):
        for benchmark in ("runbugrun", "humanevalpack"):
            output = ROOT / f"PT-base-{controller}-{benchmark}.jsonl"
            if output.with_suffix(".summary.json").exists():
                continue
            for source, sha in frozen["source_hashes"].items():
                if digest(source) != sha:
                    raise ValueError(f"Frozen source changed: {source}")
            source = ("data/prepared/scale-verified/test.jsonl" if benchmark == "runbugrun"
                      else "data/external/humanevalpack.jsonl")
            tasks = read_jsonl(source)
            reference = Path("results/round2/baselines") / f"M0-{controller}-{benchmark}.protocol.json"
            protocol = json.loads(reference.read_text())
            protocol.update(arm="PT-base", model=MODEL, model_revision=REVISION,
                            adapter=None, adapter_sha256=None, task_ids=[t["id"] for t in tasks],
                            data_sha256=digest(source), grader=GradingClient().health(),
                            training_stage="pretraining", comparison="same frozen chat/repair protocol",
                            interpretation="Measures repair/chat behavior; pretrained model has no instruction tuning.")
            manifest = output.with_suffix(".protocol.json")
            if manifest.exists() and json.loads(manifest.read_text()) != protocol:
                raise ValueError("Pretrained evaluation protocol changed")
            atomic_json(manifest, protocol)
            status("evaluating", controller=controller, benchmark=benchmark)
            generator = Generator(MODEL, revision=REVISION, max_new_tokens=768,
                                  max_context=4096, seed=42, quantized=False)
            start = time.monotonic()
            metrics = evaluate(tasks, generator, controller, benchmark, output)
            atomic_json(output.with_suffix(".summary.json"), {
                "protocol": protocol, "metrics": metrics, "wall_seconds": time.monotonic()-start})
            del generator
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    status("completed")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        atomic_json(ROOT / "status.json", {"state": "failed", "error": repr(error), "updated_at": time.time()})
        raise
