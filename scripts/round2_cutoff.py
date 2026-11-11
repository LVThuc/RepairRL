"""Stop rollout generation at a completed batch and freeze an exact row prefix."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import time
from repairrl.common import digest, read_jsonl
from evaluation.runner import parse_program
from scripts.round2_rollouts import select_negative
from scripts.round2_train import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    root = Path("results/round2")
    bank = root / "rollouts-fp16"
    control = root / "rollout-cutoff.json"
    request = {"target_source_examples": args.target, "state": "waiting_for_completed_batch",
               "requested_at": time.time(), "policy": "first N completed source rows; preserve batch overshoot separately"}
    atomic_json(control, request)
    while True:
        progress = json.loads((bank / "status.json").read_text())
        if progress.get("cutoff_target") == args.target and progress["state"] == "completed":
            request.update(state="completed", bank_status=progress)
            atomic_json(control, request)
            return
        if progress["rows_completed"] >= args.target:
            break
        time.sleep(.5)
    # Supervisor stop explicitly prevents autorestart. The last committed status
    # follows JSONL flush and atomic pairs publication, so no saved batch is lost.
    subprocess.run(["supervisorctl", "stop", "repairrl-rollouts"], check=True)
    records = read_jsonl(bank / "rollouts.jsonl")
    if len(records) < args.target:
        raise ValueError("Committed record count below requested cutoff")
    preserved = root / f"rollouts-before-cutoff-{args.target}"
    if not preserved.exists():
        shutil.copytree(bank, preserved)
    selected = records[:args.target]
    source = {str(r["id"]): r for r in read_jsonl("data/prepared/scale-verified/train.jsonl")}
    pairs = []
    for record in selected:
        row = source[str(record["id"])]
        if not record["gold_grade"] or not record["gold_grade"]["success"]:
            continue
        for negative in select_negative(row, record["samples"]):
            pairs.append({"id": row["id"], "problem_id": str(row["problem_id"]),
                          "rejected_code": parse_program(negative["generation"]["text"]),
                          "sample_index": negative["sample_index"], "fraction": negative["grade"]["fraction"]})
    for name, rows in [("rollouts.jsonl", selected), ("pairs.jsonl", pairs)]:
        temporary = bank / (name + ".cutoff-tmp")
        temporary.write_text("".join(json.dumps(r) + "\n" for r in rows))
        temporary.replace(bank / name)
    manifest = json.loads((bank / "manifest.json").read_text())
    manifest.update(cutoff_target=args.target, selected_row_ids=[r["id"] for r in selected],
                    finalizer_sha256=digest(__file__), full_batch_archive=str(preserved))
    atomic_json(bank / "manifest.json", manifest)
    (bank / "CUTOFF_COMPLETE").write_text(f"Frozen at {args.target} source examples; use a new bank to expand.\n")
    final = progress | {"state": "completed", "rows_completed": len(selected), "rows_total": args.target,
        "original_source_budget": 2000, "cutoff_target": args.target,
        "pairs": len(pairs), "distinct_problems": len({r["problem_id"] for r in pairs}),
        "pairs_sha256": digest(bank / "pairs.jsonl"), "rollouts_sha256": digest(bank / "rollouts.jsonl"),
        "physical_rows_completed": len(records), "overshoot_preserved": len(records)-len(selected),
        "categories": dict(Counter(s["grade"]["category"] for r in selected for s in r["samples"]))}
    # Publish completed only after all bank files are frozen. Existing DPO queue
    # then observes this state and starts normally.
    atomic_json(bank / "status.json", final)
    request.update(state="completed", bank_status=final, finished_at=time.time())
    atomic_json(control, request)
    print(json.dumps(request), flush=True)


if __name__ == "__main__":
    main()
