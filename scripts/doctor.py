import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from agent.sandbox import Sandbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/doctor.json")
    a = p.parse_args()
    report = {"python": platform.python_version(), "platform": platform.platform(), "bwrap": shutil.which("bwrap"),
              "candidate_python": subprocess.check_output(["/usr/bin/python3", "--version"], text=True).strip()}
    report["sandbox"] = vars(Sandbox().check())
    import torch
    import bitsandbytes as bnb
    from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
    report["packages"] = {n: importlib.metadata.version(n) for n in (
        "torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate", "gradio")}
    report["cuda"] = torch.cuda.is_available()
    if not report["cuda"]:
        raise SystemExit("CUDA unavailable")
    report["gpu"] = torch.cuda.get_device_name(0)
    report["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Real NF4 CUDA forward/backward, independent of a model download.
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4").cuda()
    inputs = torch.randn(2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    layer(inputs).float().square().mean().backward()
    report["nf4_cuda_backward"] = bool(torch.isfinite(inputs.grad).all().item())
    common = dict(output_dir="/tmp/repairrl-config-check", report_to="none", fp16=True, bf16=False)
    SFTConfig(**common, max_length=2048, completion_only_loss=True)
    DPOConfig(**common, beta=0.1, max_prompt_length=None)
    g = GRPOConfig(**common, per_device_train_batch_size=1, gradient_accumulation_steps=4, num_generations=4)
    report["trainer_imports_and_configs"] = True
    report["trainer_classes"] = [cls.__name__ for cls in (SFTTrainer, DPOTrainer, GRPOTrainer)]
    report["grpo_generation_batch_size"] = g.generation_batch_size
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
