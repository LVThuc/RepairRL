"""Run public gold/bug sanity checks on the staged server-local grader."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
from evaluation.grading_service import GradingClient
from repairrl.common import read_jsonl
from scripts.round2_train import atomic_json


def main():
    os.environ["REPAIRRL_GRADER_URL"] = "http://127.0.0.1:18592"
    client = GradingClient()
    print(client.health(), flush=True)
    tasks = [
        ("runbugrun", r)
        for r in read_jsonl("data/prepared/scale-verified/test.jsonl")[:20]
    ]
    tasks += [
        ("humanevalpack", r) for r in read_jsonl("data/external/humanevalpack.jsonl")
    ]

    def check(item):
        benchmark, row = item
        gold = client.grade(row["fixed_code"], row["problem_id"], row["id"], benchmark)
        bug = client.grade(row["buggy_code"], row["problem_id"], row["id"], benchmark)
        return {
            "benchmark": benchmark,
            "id": row["id"],
            "gold_pass": gold.success,
            "bug_fail": not bug.success,
        }

    start = time.monotonic()
    reports = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(check, tasks):
            reports.append(r)
            if len(reports) % 20 == 0:
                print("validated", len(reports), flush=True)
    result = {
        "passed": all(r["gold_pass"] and r["bug_fail"] for r in reports),
        "tasks": len(reports),
        "seconds": time.monotonic() - start,
        "reports": reports,
        "health": client.health(),
    }
    atomic_json(Path("results/round2/server-grader-validation.json"), result)
    print({k: v for k, v in result.items() if k != "reports"}, flush=True)
    if not result["passed"]:
        raise ValueError("Server grader sanity failed")


if __name__ == "__main__":
    main()
