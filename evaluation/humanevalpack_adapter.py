"""Evaluation-only HumanEvalPack adapter, with hidden assertions outside sandbox.

Published tests run in the trusted grader. Each candidate function call is an
isolated JSON RPC: candidate code receives arguments, never assertion source or
expected values. Gold sanity across all 164 tasks gates use of this adapter.
"""
import argparse
import ast
from dataclasses import asdict
import json
import time
from pathlib import Path
from agent.sandbox import Sandbox, SandboxUnavailable
from evaluation.runner import Grade, parse_program
from repairrl.codec import encode, decode
from repairrl.common import write_jsonl, read_jsonl, digest

CODEC_SOURCE = Path(__file__).parents[1].joinpath("repairrl/codec.py").read_text()


class CandidateFailure(Exception):
    pass


def function_source(row, variant):
    code = row[variant]
    # HumanEvalPack stores function bodies separately from the declaration.
    return (row.get("import", "") + "\n" + row["declaration"] + code)


def grade_humaneval(raw, task, sandbox=None):
    try:
        code = parse_program(raw)
    except ValueError:
        return Grade(0, 1, "invalid_code", False, [])
    box = sandbox or Sandbox()
    runs, failure = [], None
    started = time.monotonic()
    entry = task["entry_point"]
    if not entry.isidentifier():
        raise ValueError("Unexpected entry point")
    # Keep transport bindings out of candidate module globals and don't mount tests.
    wrapper = f'''import json,sys,io,contextlib
{CODEC_SOURCE}
_request=json.loads(sys.stdin.read())
_ns={{}}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile({code!r}, '/work/solution.py', 'exec'), _ns)
    _result=_ns[{entry!r}](*decode(_request['args']), **decode(_request['kwargs']))
print(json.dumps(encode(_result), allow_nan=False))
'''
    def candidate(*args, **kwargs):
        nonlocal failure
        if len(runs) >= 5000 or time.monotonic() - started >= 30:
            failure = "timeout"
            raise CandidateFailure("Function call/time budget exceeded")
        result = box.run(wrapper, json.dumps({"args": encode(args), "kwargs": encode(kwargs)}, allow_nan=False))
        runs.append(asdict(result) | {"test_id": len(runs), "category": result.status})
        if result.status != "ok":
            failure = result.status
            raise CandidateFailure("Candidate execution failed")
        try:
            return decode(json.loads(result.stdout))
        except (ValueError, KeyError, TypeError, RecursionError):
            failure = "runtime_error"
            raise CandidateFailure("Invalid function result") from None
    # Only externally maintained, downloaded benchmark tests are executed here.
    # Generated program text is exclusively compiled/executed inside Bubblewrap.
    namespace = {entry: candidate}
    # Some benchmark assertions use published declaration helpers (e.g. poly).
    # Only trusted source definitions are used here; the repaired entry point is RPC.
    if task.get("fixed_code"):
        helpers = ast.parse(task["fixed_code"])
        helpers.body = [n for n in helpers.body if not (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry)]
        exec(compile(helpers, "<trusted-benchmark-helpers>", "exec"), namespace)
    try:
        tree = ast.parse(task.get("test_setup", "") + "\n" + task["test"])
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "check")]
        exec(compile(tree, "<trusted-benchmark-tests>", "exec"), namespace)
        namespace["check"](candidate)
    except AssertionError:
        failure = "incorrect_output"
    except CandidateFailure:
        failure = failure or "runtime_error"
    except SandboxUnavailable:
        raise
    except Exception:
        # Gold-validated assertions can raise on a wrong result type, length,
        # or value. These are candidate failures, not harness incompatibility.
        failure = "incorrect_output"
    good = failure is None
    for run in runs:
        if run["category"] == "ok":
            run["category"] = "pass" if good else "executed"
    return Grade(int(good), 1, "pass" if good else failure, True, runs)


def prepare(output, revision="main"):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    resolved = HfApi().dataset_info("bigcode/humanevalpack", revision=revision).sha
    ds = load_dataset("bigcode/humanevalpack", "python", split="test", revision=resolved)
    if len(ds) != 164:
        raise ValueError("Expected exactly 164 HumanEvalPack Python tasks")
    rows = []
    for r in ds:
        rows.append({"id": r["task_id"], "problem_id": r["task_id"],
                     "statement": r["instruction"] + "\n" + r["docstring"],
                     "buggy_code": function_source(r, "buggy_solution"),
                     "fixed_code": function_source(r, "canonical_solution"),
                     "entry_point": r["entry_point"], "test": r["test"], "test_setup": r["test_setup"],
                     "source": "bigcode/humanevalpack", "source_revision": resolved, "evaluation_only": True})
    write_jsonl(output, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/external/humanevalpack.jsonl")
    p.add_argument("--revision", default="main")
    p.add_argument("--sanity", action="store_true")
    a = p.parse_args()
    if not Path(a.output).exists():
        prepare(a.output, a.revision)
    if a.sanity:
        box = Sandbox()
        box.check()
        reports = []
        for task in read_jsonl(a.output):
            gold = grade_humaneval(task["fixed_code"], task, box)
            buggy = grade_humaneval(task["buggy_code"], task, box)
            reports.append({"id": task["id"], "gold": gold.to_dict(), "buggy": buggy.to_dict()})
            print(task["id"], "gold:", gold.success, "buggy:", buggy.success, flush=True)
        write_jsonl("results/humanevalpack-sanity.jsonl", reports)
        passed = len(reports) == 164 and all(r["gold"]["success"] for r in reports)
        Path("results/humanevalpack-gate.json").write_text(json.dumps({
            "passed": passed, "n": len(reports), "gold_pass": sum(r["gold"]["success"] for r in reports),
            "bug_fail": sum(not r["buggy"]["success"] for r in reports), "data_sha256": digest(a.output)}, indent=2))
        if not passed:
            raise SystemExit("NO-GO: HumanEvalPack gold sanity failed; do not report benchmark metrics")
        print("HumanEvalPack gold sanity: 164/164 passed")


if __name__ == "__main__":
    main()
