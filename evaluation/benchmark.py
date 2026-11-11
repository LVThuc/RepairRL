import argparse
from collections import Counter
import json
from pathlib import Path
import time
from agent.repair import repair_task
from agent.sandbox import Sandbox
from evaluation.runner import grade_program
from repairrl.common import read_jsonl, digest, PROMPT_VERSION, BASE_REVISION


def summarize(rows):
    if not rows:
        raise ValueError("Empty evaluation")
    attempts = [a for r in rows for a in r["trace"]]
    return {"n": len(rows), "pass@1": sum(r["pass_at_1"] for r in rows) / len(rows),
            "agent@3": sum(r["agent_at_3"] for r in rows) / len(rows),
            "avg_attempts": sum(r["attempts"] for r in rows) / len(rows),
            "validity": sum(a["grade"]["valid"] for a in attempts) / len(attempts),
            "partial_fraction": sum(a["grade"]["fraction"] for a in attempts) / len(attempts),
            "avg_generation_tokens": sum(a["generation"]["tokens"] for a in attempts) / len(attempts),
            "avg_generation_seconds": sum(a["generation"]["seconds"] for a in attempts) / len(attempts),
            "avg_patch_seconds": sum(a["generation"]["seconds"] + sum(t["seconds"] for t in a["grade"]["tests"]) for a in attempts) / len(attempts),
            "tests_run": sum(len(a["grade"]["tests"]) for a in attempts),
            "categories": dict(Counter(a["grade"]["category"] for a in attempts))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--revision", default=BASE_REVISION)
    p.add_argument("--adapter")
    p.add_argument("--data", default="data/prepared/verified/test.jsonl")
    p.add_argument("--tests", default="data/prepared/verified/tests.json")
    p.add_argument("--output", required=True)
    p.add_argument("--arm", required=True, choices=["M0", "M1", "M2", "M3"])
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-context", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--benchmark", choices=["runbugrun", "humanevalpack"], default="runbugrun")
    a = p.parse_args()
    if a.arm == "M0" and a.adapter:
        p.error("M0 must be evaluated without an adapter")
    if a.arm != "M0" and not a.adapter:
        p.error("M1/M2/M3 require their own saved adapter")
    if a.benchmark == "humanevalpack":
        gate_path = Path("results/humanevalpack-gate.json")
        if not gate_path.exists():
            p.error("Run python -m evaluation.humanevalpack_adapter --sanity first")
        gate = json.loads(gate_path.read_text())
        if not gate.get("passed") or gate.get("data_sha256") != digest(a.data):
            p.error("HumanEvalPack gold gate failed or data changed")
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        p.error("Output already exists; use a new run path")
    tasks = read_jsonl(a.data)
    if a.limit:
        tasks = tasks[:a.limit]
    if not tasks:
        p.error("No tasks")
    sandbox = Sandbox(timeout=a.timeout)
    sandbox.check()
    suites = json.loads(Path(a.tests).read_text()) if a.benchmark == "runbugrun" else None
    from repairrl.model import Generator
    generate = Generator(a.model, a.adapter, a.revision, a.max_new_tokens, a.max_context, seed=a.seed)
    results, start = [], time.monotonic()
    with path.open("x") as f:
        for task in tasks:
            if a.benchmark == "runbugrun":
                def grader(raw, task=task):
                    return grade_program(raw, suites[str(task["problem_id"])], task["problem_id"], sandbox)
            else:
                from evaluation.humanevalpack_adapter import grade_humaneval
                def grader(raw, task=task):
                    return grade_humaneval(raw, task, sandbox)
            result = repair_task(task, generate, grader)
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(f"{len(results)}/{len(tasks)} {task['id']}: {result['agent_at_3']}", flush=True)
    report = {"metrics": summarize(results), "wall_seconds": time.monotonic() - start,
              "protocol": vars(a) | {"resolved_revision": generate.revision, "prompt_version": PROMPT_VERSION,
              "data_sha256": digest(a.data), "tests_sha256": digest(a.tests) if suites else None,
              "task_ids": [t["id"] for t in tasks], "max_attempts": 3, "temperature": 0.0}}
    path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
