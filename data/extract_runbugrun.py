"""Extract published Python source pairs and active test references, read-only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from repairrl.common import write_jsonl, digest


def extract(db, output, limit=100, token_limit=512, max_tests=10):
    if limit < 1 or token_limit < 1 or max_tests < 2:
        raise ValueError("Require a positive row/token limit and at least two tests")
    conn = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    required = {"bugs": {"id", "problem_id", "language", "split", "token_count", "active", "buggy_code", "fixed_code"},
                "problems": {"problem_id", "text"}, "tests": {"id", "problem_id", "active", "input", "output"}}
    for table, cols in required.items():
        actual = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not cols <= actual:
            raise ValueError(f"Unexpected {table} schema: missing {cols - actual}")
    # Verify source enum mappings against documented release, not inferred from code text.
    distribution = [dict(r) for r in conn.execute(
        "SELECT language,split,COUNT(*) AS n FROM bugs GROUP BY language,split")]
    overlap = conn.execute("SELECT COUNT(*) FROM (SELECT problem_id FROM bugs WHERE language=5 AND active=1 "
                           "AND split IN (0,1,2) GROUP BY problem_id HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
    query = """WITH candidates AS (SELECT b.id,b.problem_id,p.text AS statement,b.buggy_code,b.fixed_code,
                b.split AS source_split,b.token_count,b.change_count,
                ROW_NUMBER() OVER (PARTITION BY b.problem_id ORDER BY b.id) AS sample_rank
               FROM bugs b JOIN problems p ON p.problem_id=b.problem_id
               WHERE b.language=5 AND b.active=1 AND b.split IN (0,1,2)
               AND b.token_count<=? AND b.buggy_code<>b.fixed_code
               AND p.text IS NOT NULL AND LENGTH(p.text)>40
               ) SELECT * FROM candidates ORDER BY sample_rank,problem_id"""
    rows, suites, counts = [], {}, Counter()
    for r in conn.execute(query, (token_limit,)):
        r = dict(r)
        pid = r["problem_id"]
        if pid not in suites:
            tests = [dict(t) for t in conn.execute(
                "SELECT id,input,output FROM tests WHERE problem_id=? AND active=1 "
                "AND input IS NOT NULL AND output IS NOT NULL ORDER BY id LIMIT ?", (pid, max_tests))]
            suites[pid] = tests
        if len(suites[pid]) < 2:
            counts["insufficient_tests"] += 1
            continue
        # Prevent large/malformed rows overwhelming the pilot or prompts.
        if len(r["statement"]) > 16000 or max(len(r["buggy_code"]), len(r["fixed_code"])) > 12000:
            counts["oversized"] += 1
            continue
        r["source"] = "RunBugRun-v2"
        rows.append(r)
        if len(rows) >= limit:
            break
    conn.close()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "candidates.jsonl", rows)
    used = {r["problem_id"] for r in rows}
    (out / "tests.json").write_text(json.dumps({pid: suites[pid] for pid in sorted(used)}, ensure_ascii=False))
    report = {"source": "https://github.com/giganticode/run_bug_run/releases/tag/v2",
              "db_sha256": digest(db), "rows": len(rows), "problems": len(used),
              "language_enum": {"5": "python (official README)"}, "source_split_distribution": distribution,
              "source_problem_split_overlap": overlap, "max_tests": max_tests, "excluded": dict(counts)}
    (out / "extraction.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--output", default="data/prepared/pilot")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--token-limit", type=int, default=512)
    p.add_argument("--max-tests", type=int, default=10)
    a = p.parse_args()
    extract(a.db, a.output, a.limit, a.token_limit, a.max_tests)


if __name__ == "__main__":
    main()
