"""Small dev-only v1/v2/v3 ablation. Run when the GPU training/evaluation queue is idle."""
import argparse
import json
from pathlib import Path
import random
import os
import time
from agent.repair import repair_task
from agent.repair_v2 import repair_task_v2, fingerprint
from agent.repair_v3 import repair_task_v3, DiagnosisGenerator
from agent.sandbox import Sandbox
from evaluation.benchmark import summarize
from evaluation.runner import grade_program
from repairrl.common import BASE_REVISION, digest, read_jsonl
from scripts.run_grpo_job import process_alive


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="checkpoints/M1")
    p.add_argument("--data", default="data/prepared/scale-verified/dev.jsonl")
    p.add_argument("--tests", default="data/prepared/scale-verified/tests.json")
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--after-status", type=Path, help="Wait for a successful upstream job before using the GPU")
    a = p.parse_args()
    if not 1 <= a.tasks <= 50 or Path(a.data).name != "dev.jsonl":
        p.error("Use 1–50 tasks from a dev.jsonl manifest")
    if a.output.exists():
        p.error("Output exists; select a new run directory")
    queue_path = a.output.with_suffix(".queue.json")
    def queue_state(state, **extra):
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(json.dumps({"state": state, "pid": os.getpid(),
                              "output": str(a.output.resolve()), **extra}, indent=2))
    if a.after_status:
        queue_state("waiting", after_status=str(a.after_status.resolve()))
        deadline = time.monotonic() + 24 * 3600
        while True:
            upstream = json.loads(a.after_status.read_text())
            if upstream["state"] in {"completed", "gate_failed", "failed"} and not process_alive(upstream.get("child_pid")):
                break
            if time.monotonic() > deadline:
                queue_state("failed", reason="Upstream failed or exceeded wait budget")
                raise SystemExit("Upstream did not complete successfully; dev check not started")
            if not Path(f"/proc/{upstream['supervisor_pid']}").exists():
                queue_state("failed", reason="Upstream supervisor stopped")
                raise SystemExit("Upstream supervisor stopped; dev check not started")
            time.sleep(15)
    queue_state("running")
    # One case per problem, selected before viewing model outputs.
    unique = {}
    for row in read_jsonl(a.data):
        unique.setdefault(str(row["problem_id"]), row)
    tasks = list(unique.values())
    random.Random(42).shuffle(tasks)
    tasks = tasks[:a.tasks]
    if len(tasks) != a.tasks:
        p.error("Insufficient distinct dev problems")
    suites = json.loads(Path(a.tests).read_text())
    box = Sandbox()
    box.check()
    a.output.mkdir(parents=True)
    from repairrl.model import Generator
    generate = DiagnosisGenerator(Generator("Qwen/Qwen2.5-Coder-1.5B-Instruct", a.adapter,
                                       BASE_REVISION, seed=42))
    results = {"v1": [], "v2": [], "v3": []}
    def visible(record):
        repeated = record.get("duplicate_patch", record.get("repeated_failed_program", False))
        print(f"  attempt={record['attempt']} score={record['grade']['passed']}/{record['grade']['total']} "
              f"repeated={repeated} sampled={record['sampled_retry']} best={record.get('best_attempt')}", flush=True)
        if record.get("diagnosis"):
            print("  diagnosis:", json.dumps(record["diagnosis"]), flush=True)
        if record["feedback_sent"]:
            print("  feedback:", record["feedback_sent"], flush=True)
    for version in results:
        with (a.output / f"{version}.jsonl").open("x") as f:
            for task in tasks:
                print(version, task["id"], flush=True)
                def grader(raw):
                    return grade_program(raw, suites[str(task["problem_id"])], task["problem_id"], box)
                if version == "v1":
                    result = repair_task(task, generate, grader)
                elif version == "v2":
                    result = repair_task_v2(task, generate, grader, on_event=visible)
                else:
                    result = repair_task_v3(task, generate, grader, on_event=visible)
                results[version].append(result)
                f.write(json.dumps(result) + "\n")
                f.flush()
    metrics = {k: summarize(v) for k, v in results.items()}
    for version, rows in results.items():
        code_tokens = sum(t["generation"]["tokens"] for row in rows for t in row["trace"])
        diagnostic_tokens = sum(d["generation"]["tokens"] for row in rows for d in row.get("diagnoses", []))
        metrics[version].update(total_code_tokens=code_tokens, total_diagnosis_tokens=diagnostic_tokens,
            diagnosis_calls=sum(row.get("diagnosis_calls", 0) for row in rows),
            final_best_fraction=sum(row.get("best_grade", row["trace"][-1]["grade"])["fraction"] for row in rows) / len(rows),
            extra_repairs=sum(row["agent_at_3"] and not row["pass_at_1"] for row in rows),
            total_model_seconds=sum(t["generation"]["seconds"] for row in rows for t in row["trace"]) +
                                sum(d["generation"]["seconds"] for row in rows for d in row.get("diagnoses", [])))
        repeats = 0
        for row in rows:
            seen = set()
            for t in row["trace"]:
                key = fingerprint(t["generation"]["text"])
                repeats += key in seen
                seen.add(key)
        metrics[version]["repeated_code_generations"] = repeats
    report = {"metrics": metrics, "task_ids": [r["id"] for r in tasks], "data_sha256": digest(a.data),
              "tests_sha256": digest(a.tests), "adapter": a.adapter, "seed": 42,
              "note": "Dev ablation only. Controller policies differ. V3 costs include bounded diagnosis calls; duplicate patches are rejected before execution."}
    (a.output / "summary.json").write_text(json.dumps(report, indent=2))
    queue_state("completed", summary=str((a.output / "summary.json").resolve()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
