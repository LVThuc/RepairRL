"""Keep private grading tunnel alive and copy irreplaceable remote artifacts home."""
import json
from pathlib import Path
import subprocess
import time

HOST = "root@125.240.239.50"
SSH = ["ssh", "-p", "40401", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def main():
    out = Path("results/round2")
    out.mkdir(exist_ok=True)
    tunnel_log = (out / "grading-tunnel-watch.log").open("ab")
    tunnel = None
    while True:
        try:
            response = subprocess.run(SSH + [HOST, "curl -fsS --max-time 5 http://127.0.0.1:18591/health"],
                                      capture_output=True, timeout=25)
            healthy = response.returncode == 0 and json.loads(response.stdout).get("ready")
        except Exception:
            healthy = False
        if not healthy and not (out / "remote-grader.json").exists() and (tunnel is None or tunnel.poll() is not None):
            tunnel = subprocess.Popen(SSH[:-1] + ["ConnectTimeout=12", "-N", "-T", "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-R",
                "127.0.0.1:18591:127.0.0.1:18591", HOST], stdin=subprocess.DEVNULL,
                stdout=tunnel_log, stderr=subprocess.STDOUT)
        dest = out / "remote"
        dest.mkdir(exist_ok=True)
        with (out / "sync.log").open("ab") as log:
            try:
                subprocess.run(["rsync", "-az", "--exclude=checkpoint-*/", "-e",
                    "ssh -p 40401 -o BatchMode=yes -o ConnectTimeout=12",
                    HOST + ":/workspace/RepairRL/results/round2/", str(dest) + "/"],
                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
                freeze = out / "frozen-protocol.json"
                if freeze.exists():
                    subprocess.run(["scp", "-P", "40401", "-o", "BatchMode=yes", str(freeze),
                        HOST + ":/workspace/RepairRL/results/round2/frozen-protocol.json"],
                        stdout=log, stderr=subprocess.STDOUT, timeout=30)
            except subprocess.TimeoutExpired:
                log.write(b"Artifact sync timed out; retry next cycle\n")
        (out / "watch-status.json").write_text(json.dumps({"time": time.time(), "grader_reachable": healthy,
                                                          "remote": "125.240.239.50:40401"}, indent=2))
        time.sleep(45)


if __name__ == "__main__":
    main()
