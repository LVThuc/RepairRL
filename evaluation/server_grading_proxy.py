"""Keep the existing loopback grading URL while serving entirely on the server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from urllib.request import Request, urlopen


class Handler(BaseHTTPRequestHandler):
    def forward(self):
        if self.path not in ("/health", "/grade"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        if n > 300000:
            self.send_error(400)
            return
        body = self.rfile.read(n) if n else None
        req = Request(
            "http://127.0.0.1:18592" + self.path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=180) as response:
                data = response.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as error:
            data = json.dumps({"error": type(error).__name__}).encode()
            self.send_response(502)
            self.end_headers()
            self.wfile.write(data)

    do_GET = forward
    do_POST = forward

    def log_message(self, *args):
        pass


def main():
    while True:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 18591), Handler)
            break
        except OSError as error:
            if error.errno != 98:
                raise
            time.sleep(0.2)
    server.serve_forever()


if __name__ == "__main__":
    main()
