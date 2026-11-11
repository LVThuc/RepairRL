"""Refresh evidence and full-state inventories; optionally publish authorized milestones."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.seminar_snapshot import ROOT, snapshot
from scripts.preserve_project import inventory

PRESERVE = Path("/home/thao/Thuc/RepairRL-preservation")


def main():
    last_states = None
    last_published = None
    while True:
        try:
            logs = PRESERVE / "remote-live/logs"
            logs.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "-e",
                    "ssh -p 40401 -o BatchMode=yes -o ConnectTimeout=12",
                    "root@125.240.239.50:/workspace/repairrl-logs/",
                    str(logs) + "/",
                ],
                timeout=90,
                check=False,
            )
            mirror = PRESERVE / "remote-live/results-round2"
            states = sorted(
                (str(p.relative_to(mirror)), p.stat().st_mtime_ns, p.stat().st_size)
                for p in mirror.rglob("trainer_state.json")
            )
            if states != last_states:
                private = PRESERVE / "manifests/live-remote-inventory.json"
                inventory(mirror, private)
                (ROOT / "artifacts/manifests/live-remote-inventory.json").write_bytes(
                    private.read_bytes()
                )
                last_states = states
            snapshot()
            status = json.loads(
                (ROOT / "docs/seminar/generated/LIVE_STATUS.json").read_text()
            )
            r = status["round2"]
            rollout = r.get("rollouts-fp16/status.json", {})
            milestone = {
                "rollout_hundreds": rollout.get("rows_completed", 0) // 100,
                "rollout_state": rollout.get("state"),
                "resumable_states": len(states),
                "grpo_dev_step": r.get("grpo-dense/selection.json", {}).get(
                    "latest_dev_step"
                ),
                "dpo_state": r.get("dpo-queue.json", {}).get("state"),
                "grpo_followup_state": r.get("grpo-followup-queue.json", {}).get(
                    "state"
                ),
            }
            signature = hashlib.sha256(
                json.dumps(milestone, sort_keys=True).encode()
            ).hexdigest()
            # Enabled only after the reviewed source/doc handoff commit is published.
            if (
                PRESERVE / "publication-enabled"
            ).exists() and signature != last_published:
                env = os.environ.copy()
                env["GIT_SSH_COMMAND"] = (
                    "ssh -i /home/thao/.ssh/repairrl_handoff -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=12"
                )
                subprocess.run(
                    [
                        "git",
                        "add",
                        "docs/seminar/generated",
                        "artifacts/manifests/live-remote-inventory.json",
                    ],
                    cwd=ROOT,
                    check=True,
                )
                changed = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"], cwd=ROOT
                ).returncode
                if changed:
                    subprocess.run(
                        [
                            "git",
                            "commit",
                            "-m",
                            "docs: refresh measured experiment status and retained trainer states",
                        ],
                        cwd=ROOT,
                        check=True,
                    )
                pushed = subprocess.run(
                    ["git", "push", "git@github.com:LVThuc/RepairRL.git", "HEAD:main"],
                    cwd=ROOT,
                    env=env,
                    timeout=90,
                )
                if pushed.returncode == 0:
                    last_published = signature
            (PRESERVE / "watch-status.json").write_text(
                json.dumps(
                    {
                        "updated_at": time.time(),
                        "milestone": milestone,
                        "github_milestone_published": last_published == signature,
                    },
                    indent=2,
                )
            )
        except Exception as error:
            print(type(error).__name__, str(error), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
