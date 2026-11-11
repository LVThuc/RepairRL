"""Refuse headline comparisons when evaluation protocol or task sets differ."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summaries", nargs="+")
    a = p.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in a.summaries]
    keys = ("benchmark", "data_sha256", "tests_sha256", "task_ids", "max_attempts", "temperature",
            "max_new_tokens", "max_context", "seed", "timeout", "prompt_version")
    reference = reports[0]["protocol"]
    for report in reports[1:]:
        mismatches = [k for k in keys if report["protocol"].get(k) != reference.get(k)]
        if mismatches:
            p.error(f"Protocol mismatch: {mismatches}; compare separate tables")
    print("| Checkpoint | pass@1 | agent@3 | avg patch seconds | validity | n |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in reports:
        m = r["metrics"]
        print(f"| {r['protocol']['arm']} | {m['pass@1']:.3f} | {m['agent@3']:.3f} | {m['avg_patch_seconds']:.2f} | {m['validity']:.3f} | {m['n']} |")


if __name__ == "__main__":
    main()
