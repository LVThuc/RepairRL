"""Inventory and checksum project artifacts without loading executable pickle files."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


def checksum(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_state(path):
    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    valid = not missing
    corrupt = []
    for name in [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "training_args.bin",
        "scaler.pt",
    ]:
        file = path / name
        if file.exists():
            if not zipfile.is_zipfile(file):
                corrupt.append(
                    name + ": non-ZIP or incomplete; not validated for pinned torch 2.8"
                )
            else:
                try:
                    with zipfile.ZipFile(file) as archive:
                        if archive.testzip() is not None:
                            corrupt.append(name)
                except (OSError, zipfile.BadZipFile):
                    corrupt.append(name)
    try:
        state = json.loads((path / "trainer_state.json").read_text())
    except (OSError, ValueError):
        state = {}
        corrupt.append("trainer_state.json")
    if not isinstance(state.get("global_step"), int) or not isinstance(
        state.get("max_steps"), int
    ):
        corrupt.append("trainer_state.json: missing progress/budget")
    return {
        "classification": "resumable-trainer-state"
        if valid and not corrupt
        else "incomplete-trainer-state",
        "step": state.get("global_step"),
        "max_steps": state.get("max_steps"),
        "epoch": state.get("epoch"),
        "missing": missing,
        "corrupt_zip": corrupt,
        "has_amp_scaler": (path / "scaler.pt").exists(),
        "validation": "required files + zip CRC; no pickle execution; no GPU resume performed",
    }


def inventory(root, output):
    root = Path(root).resolve()
    rows, checkpoints = [], []
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            any(
                part in excluded or part.endswith(".egg-info")
                for part in relative.parts
            )
            or not path.is_file()
            or path.is_symlink()
        ):
            continue
        before = path.stat()
        sha = checksum(path)
        after = path.stat()
        rows.append(
            {
                "path": str(relative),
                "bytes": before.st_size,
                "sha256": sha,
                "stable_during_hash": (before.st_size, before.st_mtime_ns)
                == (after.st_size, after.st_mtime_ns),
            }
        )
        if path.name == "trainer_state.json":
            checkpoints.append(
                {"path": str(relative.parent), **checkpoint_state(path.parent)}
            )
        elif (
            path.name == "adapter_config.json"
            and not (path.parent / "trainer_state.json").exists()
        ):
            checkpoints.append(
                {
                    "path": str(relative.parent),
                    "classification": "inference-only-adapter",
                    "reason": "adapter directory itself lacks optimizer/scheduler/RNG trainer state",
                }
            )
        elif path.name == "model.safetensors":
            checkpoints.append(
                {
                    "path": str(relative.parent),
                    "classification": "inference-only-merged-model",
                }
            )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_label": root.name,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "checkpoints": checkpoints,
        "live_files": [r["path"] for r in rows if not r["stable_during_hash"]],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: result[k] for k in ("file_count", "total_bytes", "live_files")})
    )


def remote_sync(destination, once=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    while True:
        # Never --delete: retain checkpoints after trainer save_total_limit prunes
        # the corresponding remote directory. In-progress files are marked by inventory.
        command = [
            "rsync",
            "-a",
            "--partial",
            "-e",
            "ssh -p 40401 -o BatchMode=yes -o ConnectTimeout=12",
            "root@125.240.239.50:/workspace/RepairRL/results/round2/",
            str(destination) + "/",
        ]
        result = subprocess.run(command)
        print("remote full-state sync", result.returncode, time.time(), flush=True)
        if once:
            return
        time.sleep(45)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("root")
    inv.add_argument("output")
    sync = sub.add_parser("remote-sync")
    sync.add_argument("destination")
    sync.add_argument("--once", action="store_true")
    args = p.parse_args()
    if args.command == "inventory":
        inventory(args.root, args.output)
    else:
        remote_sync(args.destination, args.once)


if __name__ == "__main__":
    main()
