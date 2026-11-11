"""Audit exact/near code overlap before freezing full training manifests.

This cannot establish semantic problem disjointness or absence of pretraining
contamination. Matches require review and removal from training, not evaluation.
"""
import argparse
import json
from pathlib import Path
from data.filter_and_split import code_key, shingles
from repairrl.common import read_jsonl, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/prepared/verified/train.jsonl")
    p.add_argument("--external", default="data/external/humanevalpack.jsonl")
    p.add_argument("--output", default="results/overlap-audit.json")
    a = p.parse_args()
    train, external = read_jsonl(a.train), read_jsonl(a.external)
    reference = [(r["id"], code_key(r[k]), shingles(r[k])) for r in external
                 for k in ("buggy_code", "fixed_code")]
    matches = []
    for row in train:
        for key in ("buggy_code", "fixed_code"):
            exact, sh = code_key(row[key]), shingles(row[key])
            for eid, other, tokens in reference:
                score = len(sh & tokens) / max(1, len(sh | tokens))
                if exact == other or score >= 0.9:
                    matches.append({"train_id": row["id"], "external_id": eid,
                                    "variant": key, "exact": exact == other, "jaccard": score})
    report = {"passed": not matches, "train_sha256": digest(a.train), "external_sha256": digest(a.external),
              "train_rows": len(train), "external_rows": len(external), "matches": matches,
              "limit": "Code similarity only; semantic overlap and pretraining contamination unresolved"}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"Code overlap audit: {len(matches)} matches across {len(train)} train / {len(external)} external tasks")
    if matches:
        raise SystemExit("NO-GO: review/remove matched training records, then rebuild verification gates")


if __name__ == "__main__":
    main()
