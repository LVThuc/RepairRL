"""Filter published pairs to the frozen prompt/response budget before execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from repairrl.common import BASE_REVISION, digest, messages, read_jsonl, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-prompt-length", type=int, default=1536)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", revision=BASE_REVISION)
    rows = read_jsonl(a.input / "candidates.jsonl")
    suites = json.loads((a.input / "tests.json").read_text())
    kept, counts = [], Counter()
    for row in rows:
        words = set(re.findall(r"[a-z]+", row["statement"].lower()))
        cues = words & {"the", "input", "output", "integer", "program", "given", "print", "number"}
        if len(cues) < 3 or len(row["statement"].split()) < 40:
            counts["statement_needs_language_or_completeness_review"] += 1
            continue
        prompt = tokenizer.apply_chat_template(messages(row["statement"], row["buggy_code"]),
                                                tokenize=False, add_generation_prompt=True)
        n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n > a.max_prompt_length:
            counts["prompt_overlength"] += 1
            continue
        if any(len(tokenizer(prompt + row[key] + tokenizer.eos_token,
                              add_special_tokens=False)["input_ids"]) > a.max_length
               for key in ("fixed_code", "buggy_code")):
            counts["response_overlength"] += 1
            continue
        kept.append(row)
    a.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.output / "candidates.jsonl", kept)
    used = {str(r["problem_id"]) for r in kept}
    (a.output / "tests.json").write_text(json.dumps({pid: suites[pid] for pid in sorted(used)}))
    report = {"input_rows": len(rows), "kept": len(kept), "problems": len(used),
              "excluded": dict(counts), "max_length": a.max_length,
              "max_prompt_length": a.max_prompt_length, "tokenizer_revision": BASE_REVISION,
              "input_sha256": digest(a.input / "candidates.jsonl"),
              "statement_filter": "English cue words and >=40 whitespace words; heuristic only"}
    (a.output / "context-filter.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
