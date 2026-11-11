"""Download the named official release asset and record its transport metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="data/raw")
    a = p.parse_args()
    root = Path(a.directory)
    root.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/giganticode/run_bug_run/releases/tags/v2"
    request = urllib.request.Request(api, headers={"User-Agent": "RepairRL-setup"})
    with urllib.request.urlopen(request, timeout=60) as r:
        release = json.load(r)
    asset = next(r for r in release["assets"] if r["name"] == "runbugrun.sql.lrz")
    target = root / asset["name"]
    if not target.exists():
        temp = target.with_suffix(".download")
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as src, temp.open("wb") as dst:
            while block := src.read(1048576):
                dst.write(block)
        temp.rename(target)
    if target.stat().st_size != asset["size"]:
        raise ValueError("Download size mismatch")
    h = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    sha = h.hexdigest()
    if asset.get("digest") and asset["digest"] != f"sha256:{sha}":
        raise ValueError("Published checksum mismatch")
    report = {"release": api, "url": asset["browser_download_url"], "bytes": asset["size"],
              "sha256": sha, "published_digest": asset.get("digest")}
    (root / "download.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
