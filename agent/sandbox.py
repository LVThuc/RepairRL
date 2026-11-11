"""Fail-closed Linux Bubblewrap executor. No host execution fallback."""
import dataclasses
import math
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile
import time


class SandboxUnavailable(RuntimeError):
    pass


@dataclasses.dataclass
class Execution:
    status: str
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    returncode: int | None = None


class Sandbox:
    def __init__(self, timeout=3.0, memory_mb=512, output_bytes=65536):
        if timeout <= 0 or memory_mb < 64 or output_bytes < 1:
            raise ValueError("Invalid sandbox limits")
        self.timeout, self.memory_mb, self.output_bytes = timeout, memory_mb, output_bytes
        self.bwrap = shutil.which("bwrap")
        if not self.bwrap or not Path("/usr/bin/python3").exists():
            raise SandboxUnavailable("Linux bwrap and /usr/bin/python3 are required")

    def run(self, code, stdin=""):
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="repairrl-") as tmp, tempfile.TemporaryFile() as status_file:
            os.chmod(tmp, 0o755)
            candidate = Path(tmp) / "candidate.py"
            candidate.write_text(code, encoding="utf-8")
            candidate.chmod(0o444)
            # Hard limits are set before candidate code starts. Forks/threads are disallowed.
            launch = (
                "import resource,os;"
                f"resource.setrlimit(resource.RLIMIT_AS,({self.memory_mb * 1048576},)*2);"
                f"resource.setrlimit(resource.RLIMIT_CPU,({math.ceil(self.timeout)},)*2);"
                "resource.setrlimit(resource.RLIMIT_NPROC,(1,1));"
                "resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576));"
                "resource.setrlimit(resource.RLIMIT_NOFILE,(32,32));"
                "resource.setrlimit(resource.RLIMIT_CORE,(0,0));"
                "os.execv('/usr/bin/python3',['python3','-I','-S','/work/candidate.py'])"
            )
            cmd = [self.bwrap, "--json-status-fd", str(status_file.fileno()), "--unshare-all", "--unshare-user", "--die-with-parent", "--new-session",
                   "--uid", "65534", "--gid", "65534", "--cap-drop", "ALL",
                   "--disable-userns", "--clearenv", "--setenv", "LANG", "C.UTF-8",
                   "--setenv", "OMP_NUM_THREADS", "1", "--ro-bind", "/usr", "/usr",
                   "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                   "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
                   "--size", "16777216", "--tmpfs", "/tmp",
                   "--ro-bind", tmp, "/work", "--chdir", "/tmp",
                   "/usr/bin/python3", "-I", "-S", "-c", launch]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, start_new_session=True, env={}, pass_fds=[status_file.fileno()])
            buffers = {"stdout": bytearray(), "stderr": bytearray()}
            payload = memoryview((stdin if stdin.endswith("\n") else stdin + "\n").encode())
            sent, reason = 0, None
            try:
                with selectors.DefaultSelector() as sel:
                    for stream, name in [(p.stdout, "stdout"), (p.stderr, "stderr")]:
                        os.set_blocking(stream.fileno(), False)
                        sel.register(stream, selectors.EVENT_READ, name)
                    os.set_blocking(p.stdin.fileno(), False)
                    sel.register(p.stdin, selectors.EVENT_WRITE, "stdin")
                    while sel.get_map():
                        remaining = self.timeout - (time.monotonic() - start)
                        if remaining <= 0:
                            reason = "timeout"
                            break
                        for key, _ in sel.select(min(remaining, 0.05)):
                            if key.data == "stdin":
                                try:
                                    sent += os.write(key.fd, payload[sent:sent + 8192])
                                except BrokenPipeError:
                                    sent = len(payload)
                                if sent >= len(payload):
                                    sel.unregister(key.fileobj)
                                    key.fileobj.close()
                            else:
                                chunk = os.read(key.fd, 8192)
                                if not chunk:
                                    sel.unregister(key.fileobj)
                                    continue
                                buffers[key.data].extend(chunk)
                                if len(buffers[key.data]) > self.output_bytes:
                                    reason = "output_limit"
                                    break
                        if reason:
                            break
                    if not reason:
                        try:
                            p.wait(timeout=max(0.001, self.timeout - (time.monotonic() - start)))
                        except subprocess.TimeoutExpired:
                            reason = "timeout"
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
                for stream in (p.stdin, p.stdout, p.stderr):
                    stream.close()
            out, err = [bytes(buffers[k][:self.output_bytes]).decode("utf-8", "replace")
                        for k in ("stdout", "stderr")]
            status_file.seek(0)
            try:
                control = [json.loads(line) for line in status_file.read(8192).splitlines()]
            except (ValueError, UnicodeError):
                control = []
            if not any("child-pid" in event for event in control):
                raise SandboxUnavailable(err.strip() or "Bubblewrap did not start an isolated child")
            status = reason or ("ok" if p.returncode == 0 else "runtime_error")
            return Execution(status, out, err, time.monotonic() - start, p.returncode)

    def check(self):
        r = self.run('import os; print("sandbox-ok", os.getuid())')
        if r.status != "ok" or r.stdout.strip() != "sandbox-ok 65534":
            raise SandboxUnavailable(f"Sandbox smoke failed: {r}")
        return r
