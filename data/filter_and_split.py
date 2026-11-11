"""Verify real source pairs in the sandbox, deduplicate, then split by problem."""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import tokenize
from agent.sandbox import Sandbox
from evaluation.runner import grade_program
from repairrl.common import read_jsonl, write_jsonl, digest


def shingles(code):
    try:
        tokens = [t.string for t in tokenize.generate_tokens(io.StringIO(code).readline)
                  if t.type not in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                                    tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}]
    except (tokenize.TokenError, IndentationError):
        tokens = code.split()
    return {tuple(tokens[i:i + 5]) for i in range(max(1, len(tokens) - 4))}


def code_key(code):
    try:
        canonical = ast.dump(ast.parse(code), include_attributes=False)
    except (SyntaxError, ValueError):
        canonical = " ".join(code.split())
    return hashlib.sha256(canonical.encode()).hexdigest()


def split_rows(rows, seed=42, preserved=None):
    groups = sorted({str(r["problem_id"]) for r in rows})
    if len(groups) < 3:
        raise ValueError("Need at least three verified problem groups for train/dev/test")
    preserved = preserved or {}
    unknown = set(preserved.values()) - {"train", "dev", "test"}
    if unknown:
        raise ValueError(f"Invalid preserved split names: {unknown}")
    random.Random(seed).shuffle(groups)
    n_dev = max(1, round(len(groups) * 0.1))
    n_test = max(1, round(len(groups) * 0.1))
    names = {pid: ("dev" if i < n_dev else "test" if i < n_dev + n_test else "train")
             for i, pid in enumerate(groups)}
    names.update({pid: name for pid, name in preserved.items() if pid in names})
    return {name: [r for r in rows if names[str(r["problem_id"])] == name]
            for name in ("train", "dev", "test")}


def verify_pair(item):
    row, suite, timeout = item
    sandbox = Sandbox(timeout=timeout)
    fixed = grade_program(row["fixed_code"], suite, row["problem_id"], sandbox)
    buggy = grade_program(row["buggy_code"], suite, row["problem_id"], sandbox)
    return fixed, buggy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Extraction directory")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--min-verified", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--preserve-splits", type=Path)
    p.add_argument("--max-train", type=int)
    p.add_argument("--max-dev", type=int)
    p.add_argument("--max-test", type=int)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error("workers must be between 1 and 8")
    if any(n is not None and n < 1 for n in (a.max_train, a.max_dev, a.max_test)):
        p.error("Split caps must be positive")
    source, output = Path(a.input), Path(a.output)
    if output.resolve() == source.resolve():
        p.error("Use separate extraction and verified output directories")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(source / "candidates.jsonl")
    suites = json.loads((source / "tests.json").read_text())
    sandbox = Sandbox(timeout=a.timeout)
    sandbox.check()
    accepted, audit, counts = [], [], Counter()
    seen, near, index = set(), [], defaultdict(set)
    # executor.map yields in source order, so dedup/splits stay deterministic.
    work = ((row, suites[str(row["problem_id"])], a.timeout) for row in rows)
    pool = ThreadPoolExecutor(max_workers=a.workers)
    verification = pool.map(verify_pair, work)
    for row, (fixed, buggy) in zip(rows, verification, strict=True):
        pid = str(row["problem_id"])
        reason = "accepted"
        keys = [code_key(row[k]) for k in ("buggy_code", "fixed_code")]
        prints = [shingles(row[k]) for k in ("buggy_code", "fixed_code")]
        if not fixed.success:
            reason = "gold_failed"
        elif buggy.success:
            reason = "bug_passed"
        elif any(k in seen for k in keys):
            reason = "duplicate_code"
        else:
            for sh in prints:
                candidates = set().union(*(index[s] for s in sh))
                if any(len(sh & near[j]) / max(1, len(sh | near[j])) >= 0.9 for j in candidates):
                    reason = "near_duplicate_code"
                    break
        counts[reason] += 1
        audit.append({"id": row["id"], "problem_id": pid, "reason": reason,
                      "gold": fixed.to_dict(), "buggy": buggy.to_dict()})
        if reason == "accepted":
            accepted.append(row | {"verified": True})
            seen.update(keys)
            for sh in prints:
                idx = len(near)
                near.append(sh)
                for s in sh:
                    index[s].add(idx)
        if len(audit) % 10 == 0:
            print(f"Checked {len(audit)}/{len(rows)}: {dict(counts)}", flush=True)
    pool.shutdown(wait=True)
    write_jsonl(output / "verification.jsonl", audit)
    if len(accepted) < a.min_verified:
        (output / "gate.json").write_text(json.dumps({"passed": False, "counts": dict(counts)}, indent=2))
        raise SystemExit(f"NO-GO: only {len(accepted)} verified rows, require {a.min_verified}")
    preserved = {}
    if a.preserve_splits:
        for name in ("train", "dev", "test"):
            for record in read_jsonl(a.preserve_splits / f"{name}.jsonl"):
                pid = str(record["problem_id"])
                if pid in preserved and preserved[pid] != name:
                    raise ValueError("Preserved manifests leak problem IDs")
                preserved[pid] = name
    parts = split_rows(accepted, a.seed, preserved)
    for name in parts:
        cap = getattr(a, f"max_{name}")
        if cap:
            parts[name] = parts[name][:cap]
    for name, records in parts.items():
        write_jsonl(output / f"{name}.jsonl", records)
    used = {str(r["problem_id"]) for part in parts.values() for r in part}
    (output / "tests.json").write_text(json.dumps({pid: suites[pid] for pid in sorted(used)}))
    hashes = {f.name: digest(f) for f in output.glob("*.jsonl")}
    hashes["tests.json"] = digest(output / "tests.json")
    report = {"passed": True, "seed": a.seed, "counts": dict(counts),
              "workers": a.workers, "preserved_splits": str(a.preserve_splits) if a.preserve_splits else None,
              "split_caps": {name: getattr(a, f"max_{name}") for name in parts},
              "splits": {k: {"rows": len(v), "problems": len({r["problem_id"] for r in v})} for k, v in parts.items()},
              "hashes": hashes, "source_candidates_sha256": digest(source / "candidates.jsonl"),
              "sandbox_python": "/usr/bin/python3", "timeout": a.timeout,
              "near_duplicate": "Python token 5-gram Jaccard >=0.9; AST exact hash; both code variants"}
    (output / "gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
