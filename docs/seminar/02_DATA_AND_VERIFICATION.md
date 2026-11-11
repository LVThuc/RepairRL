# 02 — Data acquisition, verification and splits

## Published source, not invented toy bugs

RunBugRun v2 is the source of the training examples. Its published SQLite dump
contains paired competitive-programming submissions, problem descriptions and
stdin/stdout tests. The local restored database contains 716,944 bug records
across languages and 329,379 tests; those are source-database counts, not the
number of Python examples used for training.

- [Official repository](https://github.com/giganticode/run_bug_run).
- [v2 release](https://github.com/giganticode/run_bug_run/releases/tag/v2).
- Asset: `runbugrun.sql.lrz`, **120,501,798 bytes**.
- Asset SHA256: `c46225277296add83a862e47c0bc0fb18c5d26ef8b6980bd41e5ef737f93896c`.
- Decompressed SQL: 2,099,563,917 bytes; restored DB: 2,298,048,512 bytes.
- Local provenance: `data/raw/download.json`, `release-v2.json`, and the archive inventory.

The original proposal warned that the dump might be too large or unavailable.
Actual acquisition/restore succeeded. Do not present the old readiness matrix as
the current data status.

## Actual acquisition and extraction code

[`scripts/fetch_runbugrun.py`](../../scripts/fetch_runbugrun.py) records the release
asset identity; [`scripts/restore_db.py`](../../scripts/restore_db.py) restores the
dump. The external `lrzip` and `sqlite3` utilities are required for the source
workflow. Archive recovery is faster than repeating these steps.

[`data/extract_runbugrun.py`](../../data/extract_runbugrun.py) joins `bugs` with
`problems`, selects active Python examples (`language=5`), excludes unfiltered
source split 3, and uses source `token_count <= 512` as a preliminary filter.
Candidates are spread across problem IDs with round-robin ranking rather than
simply taking thousands of variants of the first problem.

The essential query shape is:

```sql
SELECT b.id, b.problem_id, p.text AS statement,
       b.buggy_code, b.fixed_code, b.split
FROM bugs b JOIN problems p ON b.problem_id = p.problem_id
WHERE b.language = 5
  AND b.split IN (0, 1, 2)
  AND b.buggy_code <> b.fixed_code;
```

Read the actual extraction script for active flags, null checks, ordering,
token filters and test selection. This shortened snippet illustrates the join;
it is not a byte-for-byte replacement for the extraction command.

## Measured preparation funnel

| Stage | Pairs retained |
|---|---:|
| Published candidates extracted | 4,000 |
| Passed context/statement checks | 3,913 |
| Passed execution verification and deduplication | 3,044 |
| Train | 2,000 |
| Dev | 200 |
| Held-out | 200 |
| Verified but unused after split caps | 644 |

Context/statement exclusions: 50 oversized prompts, 5 oversized full sequences,
32 statements requiring language/completeness review. Execution/dedup exclusions:
243 gold failures, 564 bugs passing the selected tests, 55 exact duplicates and
7 near duplicates. These are local filtering outcomes, not proof of source-label
errors. Python-version differences, time limits and the selected test cap can
explain exclusions.

The statement heuristic requires at least 40 whitespace words and three English
cue words; six samples were manually inspected. This is a coarse heuristic,
not language identification or proof that a specification is complete.

## Execution gate

For each retained pair:

1. Choose at least two active tests, up to ten deterministic test IDs per problem.
2. Run the published fixed program through the same candidate executor.
3. Require the fix to pass every selected test.
4. Run the original bug and require at least one failing test.
5. Keep test sources and expected outputs outside the candidate filesystem.

The source may have additional tests. The project verified **selected suites**, not
all possible inputs or every test in the database. A bug passing the selected
suite is discarded rather than assigned a fabricated failing test.

[`data/filter_and_split.py`](../../data/filter_and_split.py) implements execution
verification, AST-exact deduplication and Python-token 5-gram near deduplication
with Jaccard threshold 0.9. Both buggy and fixed variants participate in duplicate
checks. [`data/filter_context.py`](../../data/filter_context.py) uses the pinned
Qwen tokenizer to enforce actual model-context limits.

## Grouped splits

226 source problems appeared across predefined source splits. RepairRL therefore
created new **problem-grouped** splits with seed 42. Previously viewed pilot
train/dev/test assignments were locked so those dev/test problems did not migrate
into the enlarged training set.

| Split | Pairs | Problem IDs | Unique selected tests | JSONL bytes |
|---|---:|---:|---:|---:|
| Train | 2,000 | 1,219 | 11,230 | 5,189,639 |
| Dev | 200 | 147 | 1,370 | 552,988 |
| Held-out | 200 | 144 | 1,323 | 492,760 |

The problem-ID sets are disjoint. Shared tests within a split are expected when
multiple repair pairs belong to one problem. The 200 held-out pairs require
1,861 test executions for one complete attempt on every pair. They are not 200
fully independent problems; uncertainty estimates should cluster by problem ID.

Actual manifests live in `data/prepared/scale-verified/`. Do not confuse them with
the older small pilot directories under `data/prepared/verified/`.

| File | SHA256 |
|---|---|
| `train.jsonl` | `a28523e1fbeda27c8bebfedccfb8d344f7068f9d56d01fa9edd2c524a83c62b5` |
| `dev.jsonl` | `047b8b0ada1bffd92bb202b9f49ff3a1967b0188e481e54bb8ee0b3f039c495a` |
| `test.jsonl` | `942c6a1431af900553a833ce6c7af461554387fd010dc876d3434d197216d795` |
| `tests.json` | `949b6c5849e074a303f61bfd46f45b9da36a03e817e579824221f24f5a14369c` |

`tests.json` contains 13,923 unique selected tests across the retained splits and
occupies 1,563,502 bytes. The gate file and verification audit preserve the
selection and acceptance evidence.

## Actual token lengths

| Component | Total tokens | Mean | Median | p95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Prompt | 1,246,960 | 623.48 | 587.5 | 1,099 | 1,535 |
| Response including EOS | 310,459 | 155.23 | 116 | 410 | 689 |
| Full sequence | 1,557,419 | 778.71 | 718 | 1,455 | 2,024 |

Response lengths are estimated by joint length minus prompt length. All SFT
examples fit 2,048 training tokens. However, 26 gold responses exceed the original
512-token evaluation generation cap. Round 2 raises the shared generation cap to
768 and context to 4,096; its results require new baselines and cannot be compared
directly with the old protocol to infer a training effect.

## HumanEvalPack and contamination

All 164 Python tasks are evaluation-only. Its function interface is adapted
separately from RunBugRun stdin/stdout execution. The harness sanity check passed
164/164 published gold programs and rejected 164/164 buggy programs.

The local AST/token overlap audit detected no exact/near code matches between
the 2,000 training pairs and HumanEvalPack. It does not establish absence of
semantic overlap or foundation-model pretraining contamination. The same held-out
benchmarks are reused across research iterations; they are not a fresh,
never-observed confirmatory benchmark at every round.

## Distribution and access

The public model and data sources used did not require an HF token. The project
did not use gated DebugBench-PNYX data. Do not infer redistribution permission
from public availability: raw datasets, hidden suites and full recorded examples
are kept in private preservation storage, not ordinary Git. Preserve upstream
provenance and review source terms before distributing a new dataset release.
