"""Frozen, resumable controller/model evaluation; no test-based model selection."""
import argparse
import json
from pathlib import Path
import random
import time
from agent.repair import repair_task
from agent.repair_v31 import repair_task_v31, DiagnosisGenerator
from evaluation.benchmark import summarize
from evaluation.grading_service import GradingClient
from repairrl.common import BASE_REVISION, digest, read_jsonl


def dev_rows(limit=60):
    unique = {}
    for row in read_jsonl("data/prepared/scale-verified/dev.jsonl"):
        unique.setdefault(str(row["problem_id"]), row)
    rows = list(unique.values())
    random.Random(777).shuffle(rows)
    return rows[:limit]


def attached_generator(model, tokenizer, max_new_tokens=768):
    from repairrl.model import Generator
    g = object.__new__(Generator)
    g.model, g.tokenizer = model, tokenizer
    g.max_new_tokens, g.max_context, g.temperature = max_new_tokens, 4096, 0.0
    return g


def evaluate(tasks, generate, controller="v31", benchmark="runbugrun", path=None, batch_size=1):
    grader = GradingClient()
    wrapper = DiagnosisGenerator(generate)
    results = read_jsonl(path) if path and Path(path).exists() else []
    if [str(r["id"]) for r in results] != [str(t["id"]) for t in tasks[:len(results)]]:
        raise ValueError("Resumed evaluation task IDs changed")
    log = Path(path).open("a") if path else None
    try:
        if controller == "single" and batch_size > 1:
            import torch
            from repairrl.common import messages
            while len(results) < len(tasks):
                group = tasks[len(results):len(results)+batch_size]
                prompts = [generate.tokenizer.apply_chat_template(messages(t["statement"], t["buggy_code"]),
                           tokenize=False, add_generation_prompt=True) for t in group]
                inputs = generate.tokenizer(prompts, return_tensors="pt", padding=True,
                                             add_special_tokens=False).to(generate.model.device)
                start = time.monotonic()
                with torch.inference_mode():
                    output = generate.model.generate(**inputs, do_sample=False, max_new_tokens=generate.max_new_tokens,
                        pad_token_id=generate.tokenizer.pad_token_id, use_cache=True)
                torch.cuda.synchronize()
                duration = time.monotonic() - start
                completions = output[:, inputs["input_ids"].shape[-1]:]
                texts = generate.tokenizer.batch_decode(completions, skip_special_tokens=True)
                grades = grader.batch([(text, task["problem_id"]) for text, task in zip(texts, group, strict=True)])
                for task, text, ids, grade in zip(group, texts, completions, grades, strict=True):
                    generation = {"text": text, "tokens": int((ids != generate.tokenizer.pad_token_id).sum()),
                                  "seconds": duration/len(group), "amortized_batch_size": len(group)}
                    result = {"id": task["id"], "problem_id": task["problem_id"], "attempts": 1,
                              "pass_at_1": grade.success, "agent_at_3": grade.success, "candidate": text,
                              "trace": [{"attempt": 1, "generation": generation, "grade": grade.to_dict()}]}
                    results.append(result)
                    if log:
                        log.write(json.dumps(result) + "\n")
                        log.flush()
                print(f"dev batch {len(results)}/{len(tasks)}", flush=True)
        for task in tasks[len(results):]:
            def grade(raw):
                return grader.grade(raw, task["problem_id"], task["id"], benchmark)
            if controller == "v1":
                result = repair_task(task, wrapper, grade)
            elif controller == "single":
                result = repair_task(task, wrapper, grade, max_attempts=1)
            else:
                result = repair_task_v31(task, wrapper, grade, max_diagnoses=0 if controller == "no-diagnosis" else 2)
            results.append(result)
            if log:
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(f"{controller} {len(results)}/{len(tasks)} {task['id']} {result['agent_at_3']}", flush=True)
    finally:
        if log:
            log.close()
    metrics = summarize(results)
    metrics.update(diagnosis_calls=sum(r.get("diagnosis_calls", 0) for r in results),
                   diagnosis_tokens=sum(d["generation"]["tokens"] for r in results for d in r.get("diagnoses", [])),
                   diagnosis_seconds=sum(d["generation"]["seconds"] for r in results for d in r.get("diagnoses", [])),
                   duplicate_generations=sum(t.get("duplicate_patch", False) for r in results for t in r["trace"]),
                   extra_repairs=sum(r["agent_at_3"] and not r["pass_at_1"] for r in results))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--adapter")
    p.add_argument("--arm", default="M1")
    p.add_argument("--controller", choices=["v1", "v31", "no-diagnosis", "single"], default="v31")
    p.add_argument("--benchmark", choices=["runbugrun", "humanevalpack"], default="runbugrun")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    if a.batch_size != 1 and (a.controller != "single" or not a.dev or a.benchmark != "runbugrun"):
        p.error("Batched scoring is dev-only single-shot; headline agent evaluations remain sequential")
    source = ("data/prepared/scale-verified/dev.jsonl" if a.dev else
              "data/prepared/scale-verified/test.jsonl" if a.benchmark == "runbugrun" else "data/external/humanevalpack.jsonl")
    if a.dev and a.benchmark != "runbugrun":
        p.error("HumanEvalPack is evaluation only")
    tasks = dev_rows(a.limit) if a.dev else read_jsonl(source)
    protocol = {"arm": a.arm, "controller": a.controller, "benchmark": a.benchmark, "data_sha256": digest(source),
                "tests_sha256": digest("data/prepared/scale-verified/tests.json") if a.benchmark == "runbugrun" else digest(source),
                "task_ids": [t["id"] for t in tasks],
                "max_context": 4096, "max_new_tokens": 768, "max_attempts": 1 if a.controller == "single" else 3,
                "max_diagnoses": 2 if a.controller == "v31" else 0, "diagnosis_tokens": 128,
                "retry_temperature": 0.4 if a.controller in {"v31", "no-diagnosis"} else 0,
                "seed": 42, "model": a.model, "adapter": a.adapter, "selection_set": "dev" if a.dev else "held-out",
                "quantized": not a.fp16,
                "batch_size": a.batch_size,
                "controller_sha256": digest("agent/repair_v31.py"), "generator_sha256": digest("repairrl/model.py")}
    protocol["adapter_sha256"] = digest(Path(a.adapter) / "adapter_model.safetensors") if a.adapter else None
    protocol["model_revision"] = BASE_REVISION
    protocol["grader"] = GradingClient().health()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = a.output.with_suffix(".protocol.json")
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Evaluation protocol changed; use a new output")
    manifest.write_text(json.dumps(protocol, indent=2))
    from repairrl.model import Generator
    import torch
    torch.set_num_threads(4)
    generate = Generator(a.model, a.adapter, BASE_REVISION, max_new_tokens=768, max_context=4096, seed=42,
                         quantized=not a.fp16)
    started = time.monotonic()
    metrics = evaluate(tasks, generate, a.controller, a.benchmark, a.output, a.batch_size)
    report = {"protocol": protocol, "metrics": metrics, "wall_seconds": time.monotonic() - started}
    a.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
