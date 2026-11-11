"""Server-local grader; trusted tests stay outside the candidate chroot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.request import Request, urlopen
from agent.server_sandbox import ServerSandbox as Sandbox
from evaluation.runner import Grade, grade_program
from repairrl.common import read_jsonl, digest


class GradingClient:
    def __init__(self, timeout=180):
        self.url = os.environ.get("REPAIRRL_GRADER_URL", "http://127.0.0.1:18591")
        self.timeout = timeout

    def health(self):
        with urlopen(self.url + "/health", timeout=15) as response:
            return json.load(response)

    def grade(self, raw, problem_id=None, task_id=None, benchmark="runbugrun"):
        payload = json.dumps(
            dict(
                raw=raw,
                problem_id=str(problem_id),
                task_id=str(task_id),
                benchmark=benchmark,
            )
        ).encode()
        request = Request(
            self.url + "/grade",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return Grade(
            **{k: result[k] for k in ("passed", "total", "category", "valid", "tests")}
        )

    def batch(self, candidates):
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(
                pool.map(
                    lambda pair: self.grade(pair[0], problem_id=pair[1]), candidates
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18592)
    parser.add_argument("--tests", default="data/prepared/scale-verified/tests.json")
    parser.add_argument("--humaneval", default="data/external/humanevalpack.jsonl")
    args = parser.parse_args()
    suites = json.loads(Path(args.tests).read_text())
    humaneval = {str(row["id"]): row for row in read_jsonl(args.humaneval)}
    box = Sandbox(timeout=3.0)
    box.check()
    semaphore = threading.BoundedSemaphore(4)
    health = {
        "ready": True,
        "tests_sha256": digest(args.tests),
        "humaneval_sha256": digest(args.humaneval),
        "sandbox": "chroot-seccomp",
        "python": "python3.14",
        "runtime_version": "3.14.4",
        "runtime_location": "remote-server",
        "timeout": 3.0,
    }

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404,
                health if self.path == "/health" else {},
            )

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if self.path != "/grade" or not 0 < length <= 300000:
                    self.reply(400, {"error": "Invalid request"})
                    return
                value = json.loads(self.rfile.read(length))
                with semaphore:
                    if value["benchmark"] == "runbugrun":
                        pid = str(value["problem_id"])
                        grade = grade_program(value["raw"], suites[pid], pid, box)
                    elif value["benchmark"] == "humanevalpack":
                        from evaluation.humanevalpack_adapter import grade_humaneval

                        grade = grade_humaneval(
                            value["raw"], humaneval[str(value["task_id"])], box
                        )
                    else:
                        raise ValueError("Unknown benchmark")
                self.reply(200, grade.to_dict())
            except Exception as error:
                self.reply(500, {"error": type(error).__name__})

        def log_message(self, *args):
            pass

    print(json.dumps(health), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
