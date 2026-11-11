"""Check a saved adapter on real dev data without touching held-out results."""
import argparse
import json
from pathlib import Path
from repairrl.common import messages, read_jsonl
from repairrl.model import Generator
from evaluation.runner import grade_program
from agent.sandbox import Sandbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--data", default="data/prepared/verified/dev.jsonl")
    p.add_argument("--tests", default="data/prepared/verified/tests.json")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    meta = json.loads((Path(a.adapter) / "repairrl-training.json").read_text())
    cfg = meta["config"]
    generate = Generator(cfg["model"], a.adapter, cfg["revision"], max_new_tokens=256)
    task = read_jsonl(a.data)[0]
    suites = json.loads(Path(a.tests).read_text())
    box = Sandbox()
    box.check()
    result = generate(messages(task["statement"], task["buggy_code"]))
    grade = grade_program(result["text"], suites[str(task["problem_id"])], task["problem_id"], box)
    report = {"adapter": a.adapter, "id": task["id"], "generation": result, "grade": grade.to_dict()}
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(f"Reloaded {meta['arm']}; valid={grade.valid}; result={grade.category}; output={a.output}")
    if not grade.valid:
        raise SystemExit("NO-GO: reload smoke did not generate valid Python")


if __name__ == "__main__":
    main()
