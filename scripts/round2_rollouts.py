"""M1 train-only rollout bank: every source row and every sample has provenance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from agent.repair_v2 import fingerprint, retry_seed
from evaluation.grading_service import GradingClient
from evaluation.runner import parse_program
from repairrl.common import read_jsonl, digest, messages, BASE_REVISION


def select_negative(row, samples):
    keys = {fingerprint(row[k]) for k in ("buggy_code", "fixed_code")}
    eligible = []
    for sample in samples:
        grade = sample["grade"]
        raw = sample["generation"]["text"]
        key = fingerprint(raw)
        if not grade["valid"] or grade["category"] != "incorrect_output" or key in keys:
            continue
        keys.add(key)
        eligible.append(sample)
    return sorted(eligible, key=lambda s: (-s["grade"]["fraction"], s["sample_index"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--fp16-batched", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = Path("data/prepared/scale-verified/train.jsonl")
    rows = read_jsonl(source)
    random.Random(2026).shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    grader = GradingClient()
    health = grader.health()
    if health["tests_sha256"] != digest("data/prepared/scale-verified/tests.json"):
        raise ValueError("Grader tests do not match training source")
    import torch
    torch.set_num_threads(4)
    from repairrl.model import Generator
    from transformers import set_seed
    generate = Generator("Qwen/Qwen2.5-Coder-1.5B-Instruct", "checkpoints/M1", BASE_REVISION,
                         max_new_tokens=768, max_context=2304, temperature=0.7, seed=2026,
                         quantized=not a.fp16_batched)
    identity = {"source_sha256": digest(source), "adapter_sha256": digest("checkpoints/M1/adapter_model.safetensors"),
                "tests_sha256": health["tests_sha256"], "samples": a.samples, "seed": 2026,
                "max_context": 2304, "max_new_tokens": 768,
                "sampler": "fp16-greedy-plus-batched-stochastic-v1" if a.fp16_batched else "nf4-sequential-v1",
                "generation_defaults": generate.model.generation_config.to_dict()}
    manifest = a.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Cannot resume a changed rollout configuration")
    manifest.write_text(json.dumps(identity, indent=2))
    path = a.output / "rollouts.jsonl"
    done = {str(row["id"]): row for row in read_jsonl(path)} if path.exists() else {}
    started = time.monotonic()
    with path.open("a") as log:
        for row in rows:
            if str(row["id"]) in done:
                continue
            samples = []
            for i in range(1 if a.fp16_batched else a.samples):
                seed = retry_seed(row["id"], i + 1, 2026)
                set_seed(seed)
                generate.temperature = [0, 0.7, 0.9, 0.5][i % 4]
                result = generate(messages(row["statement"], row["buggy_code"]))
                samples.append({"sample_index": i, "seed": seed, "temperature": generate.temperature,
                                "generation": result})
            if a.fp16_batched and a.samples > 1:
                seed = retry_seed(row["id"], 2, 2026)
                set_seed(seed)
                inputs = generate.tokenizer.apply_chat_template(messages(row["statement"], row["buggy_code"]),
                    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(generate.model.device)
                began = time.monotonic()
                with torch.inference_mode():
                    batch = generate.model.generate(**inputs, num_return_sequences=a.samples-1, do_sample=True,
                        temperature=.8, top_p=.95, top_k=50, repetition_penalty=1.1, max_new_tokens=768,
                        pad_token_id=generate.tokenizer.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
                for index, ids in enumerate(batch[:, inputs["input_ids"].shape[-1]:], 1):
                    samples.append({"sample_index": index, "seed": seed, "temperature": .8,
                        "top_p": .95, "top_k": 50, "batch_index": index-1,
                        "generation": {"text": generate.tokenizer.decode(ids, skip_special_tokens=True),
                            "tokens": int((ids != generate.tokenizer.pad_token_id).sum()),
                            "seconds": (time.monotonic()-began)/(a.samples-1)}})
            grades = grader.batch([(s["generation"]["text"], row["problem_id"]) for s in samples])
            for sample, grade in zip(samples, grades, strict=True):
                sample["grade"] = grade.to_dict()
            record = {"id": row["id"], "problem_id": str(row["problem_id"]), "samples": samples}
            # Verify the actual chosen on this same grader when a negative exists.
            negatives = select_negative(row, samples)
            record["gold_grade"] = grader.grade(row["fixed_code"], row["problem_id"]).to_dict() if negatives else None
            log.write(json.dumps(record) + "\n")
            log.flush()
            done[str(row["id"])] = record
            pairs = []
            lookup = {str(r["id"]): r for r in rows}
            for rid, r in done.items():
                if not r["gold_grade"] or not r["gold_grade"]["success"]:
                    continue
                for negative in select_negative(lookup[rid], r["samples"]):
                    pairs.append({"id": lookup[rid]["id"], "problem_id": r["problem_id"],
                                  "rejected_code": parse_program(negative["generation"]["text"]),
                                  "sample_index": negative["sample_index"], "fraction": negative["grade"]["fraction"]})
            # Atomic snapshots permit training to start on a complete wave.
            temporary = a.output / "pairs.tmp"
            temporary.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
            temporary.replace(a.output / "pairs.jsonl")
            progress = {"state": "running", "rows_completed": len(done), "rows_total": len(rows),
                        "pairs": len(pairs), "distinct_problems": len({p["problem_id"] for p in pairs}),
                        "seconds": time.monotonic() - started,
                        "categories": dict(Counter(s["grade"]["category"] for r in done.values() for s in r["samples"]))}
            (a.output / "status.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
    progress["state"] = "completed"
    progress["pairs_sha256"] = digest(a.output / "pairs.jsonl")
    (a.output / "status.json").write_text(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
