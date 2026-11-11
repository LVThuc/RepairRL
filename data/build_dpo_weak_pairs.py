"""Derive weak preferences only; original buggy program is the rejected answer."""
import argparse
from repairrl.common import messages, read_jsonl, write_jsonl


def weak_pair(row):
    if row.get("verified") is not True or row["fixed_code"].strip() == row["buggy_code"].strip():
        raise ValueError("Require a verified, nondegenerate source pair")
    return {"id": row["id"], "problem_id": row["problem_id"],
            "prompt": messages(row["statement"], row["buggy_code"]),
            "chosen": [{"role": "assistant", "content": row["fixed_code"]}],
            "rejected": [{"role": "assistant", "content": row["buggy_code"]}],
            "preference_kind": "weak-gold-vs-original-bug"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    write_jsonl(a.output, [weak_pair(r) for r in read_jsonl(a.input)])


if __name__ == "__main__":
    main()
