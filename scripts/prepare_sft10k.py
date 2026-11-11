"""Expand the verified training split while freezing existing evaluation groups."""

import json
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer
from data.extract_runbugrun import extract
from data.filter_and_split import code_key, shingles, verify_pair
from repairrl.common import read_jsonl, write_jsonl, digest, messages, BASE_REVISION
from scripts.round2_train import atomic_json

OUT = Path("data/prepared/sft10k")
OLD = Path("data/prepared/scale-verified")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    def status(state, **kw):
        atomic_json(OUT / "status.json", dict(state=state, **kw))

    if (OUT / "gate.json").exists() and json.loads((OUT / "gate.json").read_text()).get(
        "passed"
    ):
        return
    source = Path("data/prepared/sft10k-source")
    status("extracting")
    if not (source / "extraction.json").exists():
        extract(
            "data/raw/runbugrun.db", source, limit=60000, token_limit=512, max_tests=10
        )
    from agent.sandbox import Sandbox

    Sandbox(timeout=3.0).check()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", revision=BASE_REVISION
    )
    original = {n: read_jsonl(OLD / f"{n}.jsonl") for n in ("train", "dev", "test")}
    excluded = {str(r["problem_id"]) for n in ("dev", "test") for r in original[n]}
    accepted_file = OUT / "accepted.jsonl"
    accepted = (
        read_jsonl(accepted_file) if accepted_file.exists() else list(original["train"])
    )
    if not accepted_file.exists():
        write_jsonl(accepted_file, accepted)
    ids = {str(r["id"]) for r in accepted}
    audit = OUT / "verification.jsonl"
    done = {str(r["id"]) for r in read_jsonl(audit)} if audit.exists() else set()
    old_suites = json.loads((OLD / "tests.json").read_text())
    suites = json.loads((source / "tests.json").read_text()) | old_suites
    seen = set()
    near = []
    index = defaultdict(set)

    def remember(row):
        for key in ("buggy_code", "fixed_code"):
            if key not in row:
                continue
            seen.add(code_key(row[key]))
            sh = shingles(row[key])
            i = len(near)
            near.append(sh)
            for s in sh:
                index[s].add(i)

    for r in (
        accepted
        + original["dev"]
        + original["test"]
        + read_jsonl("data/external/humanevalpack.jsonl")
    ):
        remember(r)
    counts = Counter()
    with accepted_file.open("a") as log, audit.open("a") as audits:
        for row in read_jsonl(source / "candidates.jsonl"):
            if len(accepted) >= 10000:
                break
            rid = str(row["id"])
            pid = str(row["problem_id"])
            if rid in ids or rid in done:
                continue
            reason = None
            if pid in excluded:
                reason = "heldout_problem"
            prompt = tokenizer.apply_chat_template(
                messages(row["statement"], row["buggy_code"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            if (
                reason is None
                and len(
                    tokenizer(
                        prompt + row["fixed_code"] + tokenizer.eos_token,
                        add_special_tokens=False,
                    )["input_ids"]
                )
                > 2048
            ):
                reason = "context"
            if reason is None:
                for key in ("buggy_code", "fixed_code"):
                    sh = shingles(row[key])
                    candidates = set().union(*(index[s] for s in sh))
                    if code_key(row[key]) in seen or any(
                        len(sh & near[j]) / max(1, len(sh | near[j])) >= 0.9
                        for j in candidates
                    ):
                        reason = "duplicate"
                        break
            fixed = buggy = None
            if reason is None:
                fixed, buggy = verify_pair((row, suites[pid], 3.0))
                if not fixed.success:
                    reason = "gold_failed"
                elif buggy.success:
                    reason = "bug_passed"
                else:
                    reason = "accepted"
                    row = row | {"verified": True}
                    accepted.append(row)
                    ids.add(rid)
                    remember(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
            counts[reason] += 1
            audits.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "reason": reason,
                        "gold": fixed.to_dict() if fixed else None,
                        "buggy": buggy.to_dict() if buggy else None,
                    }
                )
                + "\n"
            )
            audits.flush()
            if sum(counts.values()) % 50 == 0:
                status("verifying", accepted=len(accepted), counts=dict(counts))
    if len(accepted) != 10000:
        status(
            "failed",
            accepted=len(accepted),
            error="Insufficient verified examples; training stays blocked",
        )
        raise ValueError("Did not reach 10000 verified training pairs")
    write_jsonl(OUT / "train.jsonl", accepted)
    for n in ("dev", "test"):
        (OUT / f"{n}.jsonl").write_bytes((OLD / f"{n}.jsonl").read_bytes())
    used = {str(r["problem_id"]) for r in accepted + original["dev"] + original["test"]}
    atomic_json(OUT / "tests.json", {pid: suites[pid] for pid in sorted(used)})
    atomic_json(
        OUT / "gate.json",
        {
            "passed": True,
            "target": 10000,
            "counts": dict(counts),
            "splits": {
                n: {"rows": len(accepted if n == "train" else original[n])}
                for n in ("train", "dev", "test")
            },
            "hashes": {
                n: digest(OUT / n)
                for n in ("train.jsonl", "dev.jsonl", "test.jsonl", "tests.json")
            },
            "original_train_sha256": digest(OLD / "train.jsonl"),
            "source_sha256": digest(source / "candidates.jsonl"),
            "policy": "Keep original 2000 train; exclude all fixed dev/test problem IDs and exact/near code duplicates including HumanEvalPack; gold pass and bug fail on 2-10 tests; response+prompt <=2048.",
        },
    )
    status("completed", accepted=len(accepted))


if __name__ == "__main__":
    main()
