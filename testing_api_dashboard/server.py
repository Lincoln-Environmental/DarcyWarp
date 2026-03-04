from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MAX_OUTPUT_CHARS = 250_000
DEFAULT_TAIL_CHARS = 20_000
RECENT_JOBS_LIMIT = 15
MAX_REQUEST_BODY_BYTES = 1_000_000

ASSET_MAP: dict[str, tuple[Path, str]] = {
    "/assets/graphical-abstract.png": (
        REPO_ROOT / "paper" / "tables_figures" / "graphical_abstract.png",
        "image/png",
    ),
    "/assets/recharge-throughput.png": (
        REPO_ROOT / "paper" / "tables_figures" / "recharge_change" / "throughput_vs_workers.png",
        "image/png",
    ),
    "/assets/recharge-walltime.png": (
        REPO_ROOT / "paper" / "tables_figures" / "recharge_change" / "walltime_vs_workers.png",
        "image/png",
    ),
    "/assets/t-throughput.png": (
        REPO_ROOT / "paper" / "tables_figures" / "t_change" / "throughput_vs_workers.png",
        "image/png",
    ),
    "/assets/t-walltime.png": (
        REPO_ROOT / "paper" / "tables_figures" / "t_change" / "walltime_vs_workers.png",
        "image/png",
    ),
    "/assets/canterbury-side-by-side.png": (
        REPO_ROOT / "paper" / "tables_figures" / "canterbury_case_study" / "canterbury_side_by_side.png",
        "image/png",
    ),
    "/assets/canterbury-obs-sim.png": (
        REPO_ROOT
        / "DARCY_WARP_PACKAGE"
        / "canterbury_case_study"
        / "results"
        / "obs_vs_sim_scatter.png",
        "image/png",
    ),
    "/assets/canterbury-transmissivity.png": (
        REPO_ROOT / "paper" / "tables_figures" / "canterbury_case_study" / "transmissivity_field_stage2.png",
        "image/png",
    ),
    "/artifacts/canterbury-results-summary.json": (
        REPO_ROOT
        / "DARCY_WARP_PACKAGE"
        / "canterbury_case_study"
        / "results"
        / "results_summary.json",
        "application/json; charset=utf-8",
    ),
    "/artifacts/canterbury-results.pt": (
        REPO_ROOT / "DARCY_WARP_PACKAGE" / "canterbury_case_study" / "results" / "results.pt",
        "application/octet-stream",
    ),
    "/code/bench_and_plot.py": (
        REPO_ROOT / "DARCY_WARP_PACKAGE" / "bench_and_plot.py",
        "text/plain; charset=utf-8",
    ),
    "/code/benchmark_plots.py": (
        REPO_ROOT / "DARCY_WARP_PACKAGE" / "benchmark_plots.py",
        "text/plain; charset=utf-8",
    ),
    "/code/model_benchmarking_recharge_change.py": (
        REPO_ROOT / "DARCY_WARP_PACKAGE" / "model_benchmarking_recharge_change.py",
        "text/plain; charset=utf-8",
    ),
    "/code/model_benchmarking_T_change.py": (
        REPO_ROOT / "DARCY_WARP_PACKAGE" / "model_benchmarking_T_change.py",
        "text/plain; charset=utf-8",
    ),
    "/code/canterbury_case_study.py": (
        REPO_ROOT
        / "DARCY_WARP_PACKAGE"
        / "canterbury_case_study"
        / "Canterbury_case_study.py",
        "text/plain; charset=utf-8",
    ),
    "/paper/manuscript.tex": (
        REPO_ROOT / "paper" / "DarcyWarp.tex",
        "text/plain; charset=utf-8",
    ),
}


@dataclass(frozen=True)
class CommandSpec:
    key: str
    label: str
    description: str
    base_cmd: tuple[str, ...]


@dataclass
class Job:
    id: str
    command_key: str
    command: list[str]
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    return_code: int | None = None
    output: str = ""
    process: subprocess.Popen[str] | None = field(default=None, repr=False)


def _build_commands() -> dict[str, CommandSpec]:
    py = sys.executable
    return {
        "tests": CommandSpec(
            key="tests",
            label="Run Unit Tests",
            description="Runs unittest discovery under tests/.",
            base_cmd=(py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"),
        ),
        "bench_recharge_quick": CommandSpec(
            key="bench_recharge_quick",
            label="Recharge Benchmark (Quick)",
            description=(
                "Runs a small recharge-change benchmark "
                "(auto-selects backend: warp, then mf6, then fd, unless run flags are supplied)."
            ),
            base_cmd=(
                py,
                "-m",
                "DARCY_WARP_PACKAGE.model_benchmarking_recharge_change",
                "--nx",
                "1000",
                "--ny",
                "1000",
                "--dx",
                "100.0",
                "--n_cases",
                "48",
                "--workers",
                "2",
                "--seed",
                "42",
                "--device",
                "cuda:0",
            ),
        ),
        "bench_t_quick": CommandSpec(
            key="bench_t_quick",
            label="Transmissivity Benchmark (Quick)",
            description=(
                "Runs a small transmissivity-change benchmark "
                "(auto-selects backend: warp, then mf6, then fd, unless run flags are supplied)."
            ),
            base_cmd=(
                py,
                "-m",
                "DARCY_WARP_PACKAGE.model_benchmarking_T_change",
                "--nx",
                "120",
                "--ny",
                "120",
                "--dx",
                "100.0",
                "--n_cases",
                "4",
                "--workers",
                "2",
                "--seed",
                "42",
                "--device",
                "cuda:0",
            ),
        ),
        "canterbury_case_study": CommandSpec(
            key="canterbury_case_study",
            label="Canterbury Case Study",
            description="Runs the Canterbury case study calibration workflow.",
            base_cmd=(py, "-m", "DARCY_WARP_PACKAGE.canterbury_case_study.Canterbury_case_study"),
        ),
    }


COMMANDS = _build_commands()
RUN_FLAGS = {"--run_warp", "--run_mf6", "--run_fd"}
JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _default_benchmark_run_flags() -> tuple[str, ...]:
    if _module_available("warp"):
        return ("--run_warp",)
    if _module_available("flopy"):
        return ("--run_mf6",)
    return ("--run_fd",)


def _append_output(job: Job, chunk: str) -> None:
    job.output += chunk
    if len(job.output) > MAX_OUTPUT_CHARS:
        job.output = job.output[-MAX_OUTPUT_CHARS:]


def _serialize_job(job: Job, *, include_output: bool, tail_chars: int) -> dict[str, Any]:
    output = ""
    output_truncated = False
    if include_output:
        if tail_chars > 0 and len(job.output) > tail_chars:
            output = job.output[-tail_chars:]
            output_truncated = True
        else:
            output = job.output

    return {
        "id": job.id,
        "command_key": job.command_key,
        "command": job.command,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "return_code": job.return_code,
        "output": output,
        "output_size": len(job.output),
        "output_truncated": output_truncated,
    }


def _normalize_extra_args(command_key: str, extra_args: list[str]) -> list[str]:
    if not command_key.startswith("bench_"):
        return extra_args
    if any(arg in RUN_FLAGS for arg in extra_args):
        return extra_args
    return [*extra_args, *_default_benchmark_run_flags()]


def _watch_process(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        process = job.process if job else None
    if job is None or process is None:
        return

    return_code = -1
    try:
        if process.stdout is not None:
            for line in process.stdout:
                with JOBS_LOCK:
                    current = JOBS.get(job_id)
                    if current is None:
                        return
                    _append_output(current, line)
        return_code = process.wait()
    except Exception as exc:
        return_code = -1
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if current is not None:
                _append_output(current, f"\n[runner] Error while reading process output: {exc}\n")
    finally:
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if current is not None:
                current.return_code = return_code
                current.finished_at = time.time()
                if current.status == "cancelling":
                    current.status = "cancelled"
                elif return_code == 0:
                    current.status = "succeeded"
                else:
                    current.status = "failed"


def _start_job(command_key: str, extra_args: list[str]) -> Job:
    spec = COMMANDS[command_key]
    extra_args = _normalize_extra_args(command_key, extra_args)
    command = [*spec.base_cmd, *extra_args]
    command_display = " ".join(shlex.quote(part) for part in command)
    job = Job(
        id=uuid.uuid4().hex[:10],
        command_key=command_key,
        command=command,
        output=f"$ {command_display}\n",
    )
    with JOBS_LOCK:
        JOBS[job.id] = job

    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        with JOBS_LOCK:
            current = JOBS.get(job.id)
            if current is not None:
                current.status = "failed"
                current.return_code = -1
                current.finished_at = time.time()
                _append_output(current, f"[runner] Failed to start command: {exc}\n")
        return job

    with JOBS_LOCK:
        current = JOBS.get(job.id)
        if current is not None:
            current.process = process

    watcher = threading.Thread(target=_watch_process, args=(job.id,), daemon=True)
    watcher.start()
    return job


def _cancel_job(job_id: str) -> tuple[bool, str]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return False, "not_found"
        if job.status not in {"running", "cancelling"}:
            return False, "not_running"
        process = job.process
        job.status = "cancelling"

    if process is not None:
        try:
            process.terminate()
        except Exception as exc:
            with JOBS_LOCK:
                current = JOBS.get(job_id)
                if current is not None:
                    _append_output(current, f"[runner] Failed to terminate process: {exc}\n")
            return False, "terminate_failed"
    return True, "cancelling"


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DarcyWarpDashboard/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if path in {"/dashboard", "/dashboard/"}:
            self._serve_static("dashboard.html", "text/html; charset=utf-8")
            return
        asset = ASSET_MAP.get(path)
        if asset is not None:
            asset_path, content_type = asset
            self._serve_file(asset_path, content_type)
            return
        if path in {"/docs/dashboard", "/docs/dashboard/"}:
            self._serve_static("README.md", "text/markdown; charset=utf-8")
            return
        if path in {"/docs/repo", "/docs/repo/"}:
            self._serve_file(REPO_ROOT / "README.md", "text/markdown; charset=utf-8")
            return

        if path == "/api/commands":
            commands = [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "description": spec.description,
                    "base_command": " ".join(shlex.quote(part) for part in spec.base_cmd),
                }
                for spec in COMMANDS.values()
            ]
            self._send_json({"commands": commands})
            return

        if path == "/api/jobs":
            with JOBS_LOCK:
                jobs = sorted(JOBS.values(), key=lambda item: item.started_at, reverse=True)
                payload = [
                    _serialize_job(job, include_output=False, tail_chars=0)
                    for job in jobs[:RECENT_JOBS_LIMIT]
                ]
            self._send_json({"jobs": payload})
            return

        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip("/")
            tail_raw = parse_qs(parsed.query).get("tail", [str(DEFAULT_TAIL_CHARS)])[0]
            try:
                tail_chars = max(0, min(MAX_OUTPUT_CHARS, int(tail_raw)))
            except ValueError:
                self._send_json({"error": "tail must be an integer"}, status=HTTPStatus.BAD_REQUEST)
                return
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    self._send_json({"error": "job not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                payload = _serialize_job(job, include_output=True, tail_chars=tail_chars)
            self._send_json(payload)
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            body, body_error = self._read_json_body()
            if body_error is not None:
                status = (
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                    if body_error == "request body too large"
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json({"error": body_error}, status=status)
                return
            assert body is not None
            command_key = body.get("command_key")
            extra_args_raw = body.get("extra_args", "")
            if not isinstance(command_key, str) or command_key not in COMMANDS:
                self._send_json({"error": "unknown command_key"}, status=HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(extra_args_raw, str):
                self._send_json({"error": "extra_args must be a string"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                extra_args = shlex.split(extra_args_raw)
            except ValueError as exc:
                self._send_json(
                    {"error": f"could not parse extra_args: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            job = _start_job(command_key, extra_args)
            with JOBS_LOCK:
                payload = _serialize_job(job, include_output=True, tail_chars=DEFAULT_TAIL_CHARS)
            self._send_json(payload, status=HTTPStatus.ACCEPTED)
            return

        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
            ok, reason = _cancel_job(job_id)
            if not ok:
                status = HTTPStatus.NOT_FOUND if reason == "not_found" else HTTPStatus.CONFLICT
                self._send_json({"error": reason}, status=status)
                return
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    self._send_json({"error": "job not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                payload = _serialize_job(job, include_output=True, tail_chars=DEFAULT_TAIL_CHARS)
            self._send_json(payload)
            return

        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _read_json_body(self) -> tuple[dict[str, Any] | None, str | None]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}, None
        try:
            length = int(raw_length)
        except ValueError:
            return None, "invalid Content-Length header"
        if length <= 0:
            return {}, None
        if length > MAX_REQUEST_BODY_BYTES:
            return None, "request body too large"
        raw_body = self.rfile.read(length)
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except UnicodeDecodeError:
            return None, "request body must be valid UTF-8"
        except json.JSONDecodeError:
            return None, "request body must be valid JSON"
        if not isinstance(parsed, dict):
            return None, "request body must be a JSON object"
        return parsed, None

    def _serve_static(self, filename: str, content_type: str) -> None:
        path = HERE.joinpath(filename)
        if not path.is_file():
            self._send_json({"error": "file not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._serve_file(path, content_type)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_json({"error": "file not found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Local dashboard for running tests and benchmarks.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
