"""Bounded CPU-only benchmark; remote execution uses a minimal chroot and seccomp."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import errno
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time
import uuid
from agent.sandbox import Execution, Sandbox
from evaluation.runner import grade_program

JAIL = Path("/workspace/repairrl-cpu-benchmark-jail")


def setup():
    if JAIL.exists():
        return
    binary = Path("/usr/bin/python3").resolve()
    files = [binary]
    import re

    files += [
        Path(x)
        for x in re.findall(
            r"(/\S+)", subprocess.check_output(["ldd", str(binary)], text=True)
        )
    ]
    for f in files:
        target = JAIL / f.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    runtime = (
        Path("/usr/lib")
        / f"python{os.sys.version_info.major}.{os.sys.version_info.minor}"
    )
    shutil.copytree(
        runtime,
        JAIL / runtime.relative_to("/"),
        ignore=shutil.ignore_patterns("__pycache__", "test", "tests"),
    )
    (JAIL / "work").mkdir()


class ChrootRunner:
    def run(self, code, stdin=""):
        folder = JAIL / "work" / uuid.uuid4().hex
        folder.mkdir(mode=0o755)
        candidate = folder / "candidate.py"
        candidate.write_text(code)
        candidate.chmod(0o444)
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_rule_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        lib.seccomp_load.argtypes = [ctypes.c_void_p]
        lib.seccomp_release.argtypes = [ctypes.c_void_p]

        def restrict():
            os.chroot(JAIL)
            os.chdir("/work")
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
            for limit, value in [
                (resource.RLIMIT_AS, 512 * 1048576),
                (resource.RLIMIT_CPU, 3),
                (resource.RLIMIT_NPROC, 1),
                (resource.RLIMIT_FSIZE, 65536),
                (resource.RLIMIT_NOFILE, 32),
                (resource.RLIMIT_CORE, 0),
            ]:
                resource.setrlimit(limit, (value, value))
            ctx = lib.seccomp_init(0x7FFF0000)
            if not ctx:
                os._exit(120)
            for name in [
                "socket",
                "socketpair",
                "clone",
                "clone3",
                "fork",
                "vfork",
                "ptrace",
                "process_vm_readv",
                "process_vm_writev",
                "kill",
                "tgkill",
                "tkill",
                "mount",
                "unshare",
                "setns",
            ]:
                number = lib.seccomp_syscall_resolve_name(name.encode())
                if (
                    number >= 0
                    and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0)
                    != 0
                ):
                    os._exit(121)
            if lib.seccomp_load(ctx) != 0:
                os._exit(122)
            lib.seccomp_release(ctx)

        start = time.monotonic()
        try:
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                try:
                    p = subprocess.run(
                        [
                            "/usr/bin/python3.12",
                            "-I",
                            "-S",
                            f"/work/{folder.name}/candidate.py",
                        ],
                        input=(
                            stdin if stdin.endswith("\n") else stdin + "\n"
                        ).encode(),
                        stdout=out,
                        stderr=err,
                        timeout=3,
                        env={},
                        preexec_fn=restrict,
                        start_new_session=True,
                    )
                    status = "ok" if p.returncode == 0 else "runtime_error"
                    rc = p.returncode
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    rc = None
                out.seek(0)
                err.seek(0)
                return Execution(
                    status,
                    out.read(65536).decode(errors="replace"),
                    err.read(65536).decode(errors="replace"),
                    time.monotonic() - start,
                    rc,
                )
        finally:
            shutil.rmtree(folder)


def verify(item):
    row, mode = item
    runner = ChrootRunner() if mode == "server" else Sandbox()
    return {
        "id": row["id"],
        "gold": grade_program(
            row["fixed_code"], row["tests"], row["problem_id"], runner
        ).to_dict(),
        "buggy": grade_program(
            row["buggy_code"], row["tests"], row["problem_id"], runner
        ).to_dict(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local", "server"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = json.loads(Path(a.input).read_text())
    if a.mode == "server":
        setup()
        smoke = ChrootRunner().run(
            'import os,socket\nprint(os.getuid())\nfor f in [lambda:open("/workspace/.env"),lambda:socket.socket(),lambda:os.fork()]:\n try:f();print("UNSAFE")\n except OSError:print("blocked")'
        )
        if (
            smoke.status != "ok"
            or smoke.stdout.strip() != "65534\nblocked\nblocked\nblocked"
        ):
            raise RuntimeError(smoke)
    reports = []
    for workers in [1, 4]:
        start = time.monotonic()
        if workers == 1:
            results = [verify((r, a.mode)) for r in rows]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(verify, [(r, a.mode) for r in rows]))
        duration = time.monotonic() - start
        report = {
            "mode": a.mode,
            "workers": workers,
            "pairs": len(rows),
            "seconds": duration,
            "pairs_per_minute": len(rows) * 60 / duration,
            "results": results,
        }
        reports.append(report)
        Path(a.output).write_text(json.dumps(reports, indent=2))
        print({k: v for k, v in report.items() if k != "results"}, flush=True)


if __name__ == "__main__":
    main()
