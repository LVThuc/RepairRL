"""Multi-prompt FP16 M1 rollouts; resumes and preserves the first rollout wave."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from agent.repair_v2 import retry_seed
from evaluation.grading_service import GradingClient
from evaluation.runner import parse_program
from repairrl.common import BASE_REVISION, read_jsonl, digest, messages
from scripts.round2_rollouts import select_negative
from scripts.round2_train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-prompts", type=int, default=8)
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    from transformers import set_seed
    from repairrl.model import Generator
    source = Path("data/prepared/scale-verified/train.jsonl")
    rows = read_jsonl(source)
    random.Random(2026).shuffle(rows)
    lookup = {str(r["id"]): r for r in rows}
    a.output.mkdir(parents=True, exist_ok=True)
    grader = GradingClient()
    health = grader.health()
    if health["tests_sha256"] != digest("data/prepared/scale-verified/tests.json"):
        raise ValueError("Grader identity mismatch")
    generate = Generator("Qwen/Qwen2.5-Coder-1.5B-Instruct", "checkpoints/M1", BASE_REVISION,
                         max_new_tokens=768, max_context=2304, quantized=False, seed=2026)
    manifest = a.output / "manifest.json"
    identity = {"source_sha256": digest(source), "adapter_sha256": digest("checkpoints/M1/adapter_model.safetensors"),
                "tests_sha256": health["tests_sha256"], "samples": 4, "seed": 2026,
                "max_context": 2304, "max_new_tokens": 768,
                "sampler": "fp16-multiprompt-v2", "prompt_batch": a.batch_prompts,
                "source_sha256_script": digest(__file__), "top_p": .95, "top_k": 50, "temperature": .8}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(old[k] != identity[k] for k in ("source_sha256", "adapter_sha256", "tests_sha256", "samples")):
            raise ValueError("Cannot resume a changed source/model")
        previous = a.output / "manifest-initial.json"
        if not previous.exists():
            previous.write_text(json.dumps(old, indent=2))
    atomic_json(manifest, identity)
    path = a.output / "rollouts.jsonl"
    done = {str(r["id"]): r for r in read_jsonl(path)} if path.exists() else {}
    remaining = [r for r in rows if str(r["id"]) not in done]
    started, cursor, batch_size = time.monotonic(), 0, a.batch_prompts
    with path.open("a") as log:
        while cursor < len(remaining):
            group = remaining[cursor:cursor+batch_size]
            prompts = [generate.tokenizer.apply_chat_template(messages(r["statement"], r["buggy_code"]),
                        tokenize=False, add_generation_prompt=True) for r in group]
            inputs = generate.tokenizer(prompts, return_tensors="pt", padding=True,
                                        add_special_tokens=False).to(generate.model.device)
            length = inputs["input_ids"].shape[-1]
            seed = retry_seed(group[0]["id"], len(group), 2026)
            set_seed(seed)
            try:
                began = time.monotonic()
                with torch.inference_mode():
                    greedy = generate.model.generate(**inputs, do_sample=False, max_new_tokens=768,
                        pad_token_id=generate.tokenizer.pad_token_id, use_cache=True)
                    sampled = generate.model.generate(**inputs, do_sample=True, num_return_sequences=3,
                        temperature=.8, top_p=.95, top_k=50, repetition_penalty=1.1, max_new_tokens=768,
                        pad_token_id=generate.tokenizer.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                del inputs
                torch.cuda.empty_cache()
                print("OOM: retrying unsaved prompt batch with", batch_size, flush=True)
                continue
            elapsed = time.monotonic() - began
            all_samples = []
            for i, row in enumerate(group):
                codes = [greedy[i, length:]] + [sampled[i*3+j, length:] for j in range(3)]
                samples = [{"sample_index": j, "seed": seed, "temperature": 0 if j == 0 else .8,
                    "batch_row_ids": [r["id"] for r in group], "batch_row_index": i,
                    "sampler": identity["sampler"], "generation": {
                        "text": generate.tokenizer.decode(ids, skip_special_tokens=True),
                        "tokens": int((ids != generate.tokenizer.pad_token_id).sum()),
                        "seconds": elapsed/(4*len(group))}} for j, ids in enumerate(codes)]
                all_samples.append(samples)
            grades = grader.batch([(s["generation"]["text"], r["problem_id"])
                                   for r, ss in zip(group, all_samples, strict=True) for s in ss])
            for i, (row, samples) in enumerate(zip(group, all_samples, strict=True)):
                for sample, grade in zip(samples, grades[i*4:(i+1)*4], strict=True):
                    sample["grade"] = grade.to_dict()
                negatives = select_negative(row, samples)
                record = {"id": row["id"], "problem_id": str(row["problem_id"]), "samples": samples,
                    "gold_grade": grader.grade(row["fixed_code"], row["problem_id"]).to_dict() if negatives else None}
                log.write(json.dumps(record) + "\n")
                log.flush()
                done[str(row["id"])] = record
            cursor += len(group)
            pairs = []
            for rid, record in done.items():
                if not record["gold_grade"] or not record["gold_grade"]["success"]:
                    continue
                for negative in select_negative(lookup[rid], record["samples"]):
                    pairs.append({"id": lookup[rid]["id"], "problem_id": record["problem_id"],
                        "rejected_code": parse_program(negative["generation"]["text"]),
                        "sample_index": negative["sample_index"], "fraction": negative["grade"]["fraction"]})
            temporary = a.output / "pairs.tmp"
            temporary.write_text("".join(json.dumps(r) + "\n" for r in pairs))
            temporary.replace(a.output / "pairs.jsonl")
            progress = {"state": "running", "rows_completed": len(done), "rows_total": len(rows), "pairs": len(pairs),
                "distinct_problems": len({p["problem_id"] for p in pairs}), "wave_seconds": time.monotonic()-started,
                "batch_prompts": batch_size, "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                "categories": dict(Counter(s["grade"]["category"] for r in done.values() for s in r["samples"]))}
            atomic_json(a.output / "status.json", progress)
            print(json.dumps(progress), flush=True)
    progress = json.loads((a.output / "status.json").read_text())
    progress.update(state="completed", pairs_sha256=digest(a.output / "pairs.jsonl"))
    atomic_json(a.output / "status.json", progress)


if __name__ == "__main__":
    main()
