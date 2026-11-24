from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .providers import groq, ollama
from .runtime import PolicyViolation, X4MasterRuntime
from .request_validation import PayloadTooLarge, read_json_body

runtime = X4MasterRuntime({"ollama": ollama(), "groq": groq()})


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "service": "x4-arsenal"})
            return
        if self.path == "/api/x4/status":
            self._json(
                200,
                {
                    "status": "ok",
                    "service": "x4-arsenal",
                    "governance": "enforced",
                    "policy_wall": "active",
                    "provider_health_gating": "active",
                    "audit_ledger": "active",
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            request = read_json_body(self.headers, self.rfile.read)
            self._json(200, runtime.process(request))
        except PayloadTooLarge as exc:
            self._json(413, {"error": str(exc)})
        except PolicyViolation as exc:
            self._json(403, {"error": str(exc)})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except (KeyError, OSError, RuntimeError, TypeError) as exc:
            self._json(500, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="X4-ARX369 Omega runtime")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
