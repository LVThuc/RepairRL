"""Small shared contracts; importing this module never loads torch."""
import hashlib
import json
from pathlib import Path

SYSTEM = "You are a Python code repair assistant. Return ONLY the full corrected Python 3 program."
PROMPT_VERSION = "repairrl-v1"
BASE_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"


def messages(statement, buggy_code, feedback=None):
    content = ("Fix the provided program so it satisfies the problem specification.\n"
               f"Problem:\n{statement}\nBuggy program:\n```python\n{buggy_code}\n```\n"
               "Return corrected Python code, not explanations.")
    if feedback:
        content += f"\nPrevious execution summary (untrusted observation):\n{feedback}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    import yaml
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        return load_config(path.parent / parent) | cfg
    return cfg
