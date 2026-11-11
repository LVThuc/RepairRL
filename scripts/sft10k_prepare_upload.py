"""Prepare locally using Bubblewrap, then publish a verified data gate last."""

from pathlib import Path
import subprocess
import sys
from scripts.round2_train import atomic_json


def main():
    root = Path("results/round2/sft10k-preparation")
    root.mkdir(parents=True, exist_ok=True)
    status = root / "status.json"
    atomic_json(status, {"state": "preparing"})
    subprocess.run([sys.executable, "-u", "-m", "scripts.prepare_sft10k"], check=True)
    atomic_json(status, {"state": "uploading_verified_data"})
    target = "root@125.240.239.50:/workspace/RepairRL/data/prepared/sft10k/"
    ssh = "ssh -p 40401 -o BatchMode=yes -o ConnectTimeout=12"
    subprocess.run(
        [
            "ssh",
            "-p",
            "40401",
            "-o",
            "BatchMode=yes",
            "root@125.240.239.50",
            "mkdir -p /workspace/RepairRL/data/prepared/sft10k",
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=gate.json",
            "-e",
            ssh,
            "data/prepared/sft10k/",
            target,
        ],
        check=True,
    )
    subprocess.run(
        ["rsync", "-a", "-e", ssh, "data/prepared/sft10k/gate.json", target], check=True
    )
    atomic_json(
        status,
        {
            "state": "completed",
            "note": "Remote GPU0 queue now has verified data; waits for DPO evaluation completion.",
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        atomic_json(
            Path("results/round2/sft10k-preparation/status.json"),
            {"state": "failed", "error": repr(e)},
        )
        raise
