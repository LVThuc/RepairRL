"""Server-owned chroot runner using the preserved Python3.14 candidate runtime."""

import ctypes
import errno
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from agent.sandbox import Execution, SandboxUnavailable

JAIL = Path("/workspace/repairrl-grader-runtime")


def launch(candidate, timeout, memory, output):
    # Trusted helper runs in a separate process: no preexec_fn in HTTP threads.
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
    os.chroot(JAIL)
    os.chdir("/work")
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    for limit, value in [
        (resource.RLIMIT_AS, memory * 1048576),
        (resource.RLIMIT_CPU, math.ceil(timeout)),
        (resource.RLIMIT_NPROC, 1),
        (resource.RLIMIT_FSIZE, output),
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
            and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0) != 0
        ):
            os._exit(121)
    if lib.seccomp_load(ctx) != 0:
        os._exit(122)
    lib.seccomp_release(ctx)
    os.execve(
        "/usr/bin/python3.14",
        ["python3.14", "-I", "-S", candidate],
        {"LANG": "C.UTF-8", "OMP_NUM_THREADS": "1"},
    )


class ServerSandbox:
    def __init__(self, timeout=3.0, memory_mb=512, output_bytes=65536):
        self.timeout, self.memory_mb, self.output_bytes = (
            timeout,
            memory_mb,
            output_bytes,
        )
        if not (JAIL / "usr/bin/python3.14").exists():
            raise SandboxUnavailable("Preserved runtime missing")

    def run(self, code, stdin=""):
        start = time.monotonic()
        folder = JAIL / "work" / uuid.uuid4().hex
        folder.mkdir(mode=0o755)
        path = folder / "candidate.py"
        path.write_text(code, encoding="utf-8")
        path.chmod(0o444)
        try:
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                command = [
                    sys.executable,
                    "-m",
                    "agent.server_sandbox",
                    f"/work/{folder.name}/candidate.py",
                    str(self.timeout),
                    str(self.memory_mb),
                    str(self.output_bytes),
                ]
                try:
                    p = subprocess.run(
                        command,
                        input=(
                            stdin if stdin.endswith("\n") else stdin + "\n"
                        ).encode(),
                        stdout=out,
                        stderr=err,
                        timeout=self.timeout,
                        start_new_session=True,
                    )
                    rc = p.returncode
                    status = "ok" if rc == 0 else "runtime_error"
                except subprocess.TimeoutExpired:
                    rc = None
                    status = "timeout"
                out.seek(0, 2)
                outsize = out.tell()
                err.seek(0, 2)
                errsize = err.tell()
                if rc == -signal.SIGXFSZ or max(outsize, errsize) > self.output_bytes:
                    status = "output_limit"
                out.seek(0)
                err.seek(0)
                return Execution(
                    status,
                    out.read(self.output_bytes).decode(errors="replace"),
                    err.read(self.output_bytes).decode(errors="replace"),
                    time.monotonic() - start,
                    rc,
                )
        finally:
            shutil.rmtree(folder)

    def check(self):
        code = 'import os,socket,sys\nprint(os.getuid(),sys.version.split()[0])\nfor f in [lambda:open("/workspace/.env"),lambda:socket.socket(),lambda:os.fork()]:\n try:f();print("UNSAFE")\n except OSError:print("blocked")'
        result = self.run(code)
        if (
            result.status != "ok"
            or result.stdout.strip() != "65534 3.14.4\nblocked\nblocked\nblocked"
        ):
            raise SandboxUnavailable(str(result))
        return result


if __name__ == "__main__":
    launch(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
