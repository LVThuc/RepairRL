"""Stream an official SQLite SQL dump without loading the whole dump into RAM."""
import argparse
import sqlite3
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sql", type=Path)
    p.add_argument("db", type=Path)
    args = p.parse_args()
    if args.db.exists():
        p.error("Destination exists; use a new filename")
    temp = args.db.with_suffix(".partial")
    if temp.exists():
        p.error("Partial restore exists; inspect/remove it before retrying")
    conn = sqlite3.connect(temp)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        statement, count = "", 0
        with args.sql.open(encoding="utf-8") as f:
            for line in f:
                statement += line
                if sqlite3.complete_statement(statement):
                    conn.execute(statement)
                    statement = ""
                    count += 1
                    if count % 100000 == 0:
                        print(f"Restored {count} statements", flush=True)
        if statement.strip():
            conn.execute(statement)
        conn.commit()
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite quick_check failed")
    finally:
        conn.close()
    temp.rename(args.db)
    print(f"Restored {args.db}")


if __name__ == "__main__":
    main()
