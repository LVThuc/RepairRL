"""Bundle the exact candidate interpreter and its shared-library dependencies."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tarfile
from repairrl.common import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/private/grader-runtime")
    )
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=True)
    binary = Path("/usr/bin/python3").resolve()
    runtime = Path("/usr/lib/python3.14")
    files = {binary} | {
        p for p in runtime.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    for path in [binary] + list(runtime.rglob("*.so")):
        result = subprocess.run(["ldd", str(path)], capture_output=True, text=True)
        files.update(
            Path(p) for p in re.findall(r"(/\S+)", result.stdout) if Path(p).is_file()
        )
    archive = root / "python314-runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.dereference = True
        for path in sorted(files):
            tar.add(path, arcname=str(path).lstrip("/"), recursive=False)
    manifest = {
        "archive_sha256": digest(archive),
        "interpreter": str(binary),
        "files": [{"path": str(p), "sha256": digest(p)} for p in sorted(files)],
    }
    (root / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(archive, archive.stat().st_size, len(files), flush=True)


if __name__ == "__main__":
    main()
