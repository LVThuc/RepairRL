# Data preparation

Training data comes from [RunBugRun v2](https://github.com/giganticode/run_bug_run/releases/tag/v2). [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) Python is evaluation-only. Raw programs and scoring tests are not redistributed in this repository.

## Published split

[runbugrun-splits.json](manifests/runbugrun-splits.json) records exact source row IDs, problem IDs, seed and prepared-file hashes:

| Split | Examples | Problem groups |
|---|---:|---:|
| Train | 2,000 | 1,219 |
| Dev | 200 | 147 |
| Test | 200 | 144 |

All problem groups are disjoint. Frequent checkpoint screens use 60 distinct dev problems. HumanEvalPack has 164 evaluation-only tasks.

## Preparation procedure

The source database is an lrzip-compressed SQL dump. Restore it into a fresh SQLite file, extract Python examples, filter prompt/response length, verify execution, then deduplicate and split by problem ID.

```bash
python -m scripts.fetch_runbugrun
lrunzip -o data/raw/runbugrun.sql data/raw/runbugrun.sql.lrz
python -m scripts.restore_db data/raw/runbugrun.sql data/raw/runbugrun.db
python -m data.extract_runbugrun --db data/raw/runbugrun.db --output data/prepared/scale-source --limit 4000
python -m data.filter_context --input data/prepared/scale-source --output data/prepared/scale-context
python -m data.filter_and_split --input data/prepared/scale-context --output data/prepared/scale-verified --workers 4 --max-train 2000 --max-dev 200 --max-test 200
```

These commands describe the preparation pipeline. The published experiment also preserved earlier pilot problem assignments: reproducing its exact membership requires the recorded split IDs/frozen manifests, rather than assuming a newly sampled split is identical. Label new preparations as new experiments and compare hashes before training.

The recorded funnel is 4,000 extracted → 3,913 context/statement-filtered → 3,044 verified/deduplicated → 2,000 train, 200 dev, 200 test, 644 accepted but unused.

Checks include gold passing all selected tests, buggy code failing at least one, exact AST deduplication and token five-gram Jaccard similarity >=0.9. Each RunBugRun task has 2–10 selected active tests. The description/language filter is heuristic. Source token counts differ from Qwen tokenizer counts.

Use `data.audit_overlap` against HumanEvalPack after preparation. This checks code similarity; semantic overlap and foundation-model pretraining contamination remain unresolved. Do not use HumanEvalPack examples for training or selection.

The trainer requires a successful `gate.json`, matching hashes and verified, problem-disjoint split manifests. See the [technical report](../docs/seminar/TECHNICAL_REPORT.md) for source revisions, counts, exclusions and limitations.
