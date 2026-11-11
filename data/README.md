# Published data only

RunBugRun v2: https://github.com/giganticode/run_bug_run/releases/tag/v2

The release asset is 120,501,798 bytes. Its decompressed SQLite SQL dump is
2,099,563,917 bytes. Downloads, SQL, SQLite, extracted source examples and prepared
manifests are ignored by Git. Inspect upstream licensing/provenance before
redistributing examples; this repository does not republish dataset content.

`python -m scripts.fetch_runbugrun` records the official URL, byte size and a
local SHA256 (the release currently supplies no digest). Decompress with
`lrunzip -p 2 -o data/raw/runbugrun.sql data/raw/runbugrun.sql.lrz`, then use
`python -m scripts.restore_db data/raw/runbugrun.sql data/raw/runbugrun.db`.
A standard-library streaming restore avoids requiring the sqlite3 CLI.

Extraction audits schema, enums and original problem/split overlap, selecting
active Python (5), source splits 0/1/2 and active tests only. `filter_and_split`
executes both versions, deduplicates, groups by problem, and writes hashed gates.
No fake bugs or generated preference candidates are used. Tiny snippets in unit
tests are harness fixtures and are never used as experimental training data.

HumanEvalPack is evaluation only. `python -m evaluation.humanevalpack_adapter
--sanity` downloads its pinned revision, prepares 164 cases and checks gold and
buggy programs. Never mix `data/external` into training or dev selection.

Prepared files contain source and tests for the trusted grader, not candidate
mounts. Each training arm verifies dataset hashes against `gate.json` before
loading. Scaling extraction requires a new directory and a fresh verification
run; never overwrite manifests during a multi-arm comparison.

`python -m data.audit_overlap` checks both source-code variants in the training
manifest against the external benchmark for AST-exact and token-near matches.
Re-run it after scaling; a zero-match audit is not proof of semantic disjointness.

For larger runs, `data.filter_context` filters the published candidates using
Qwen's pinned tokenizer before execution. `filter_and_split --workers 4`
verifies independent pairs concurrently but consumes results in source order.
`--preserve-splits data/prepared/verified` locks existing pilot problem IDs to
their original partitions. Optional `--max-train`, `--max-dev`, and `--max-test`
cap rows only after group splitting; report actual retained counts separately
from total verification acceptance counts.
