from __future__ import annotations

import json
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import typer
import yaml

from psf.config_loader import load_system_definition
from psf.engine import ExecutionEngine
from psf.exceptions import ConfigError, ExecutionError
from psf.logging import RunLogger
from psf import cli as psf_cli


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
RUNS_DIR = REPO_ROOT / "runs"
SYSTEMS_DIR = REPO_ROOT / "systems"


def serve(host: str = "127.0.0.1", port: int = 8321) -> None:
    server = ThreadingHTTPServer((host, port), UIRequestHandler)
    server.serve_forever()


class UIRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ASSETS_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_error(404, "Not Found")
            return
        self._handle_api_post(parsed)

    def _handle_api_get(self, parsed) -> None:
        if parsed.path == "/api/info":
            payload = {
                "root": str(REPO_ROOT),
                "runs_dir": str(RUNS_DIR),
                "systems_dir": str(SYSTEMS_DIR),
            }
            self._send_json(200, payload)
            return

        if parsed.path == "/api/runs":
            payload = {"runs": _scan_runs(RUNS_DIR)}
            self._send_json(200, payload)
            return

        if parsed.path == "/api/run":
            params = parse_qs(parsed.query)
            run_dir = _get_param(params, "path")
            if not run_dir:
                self._send_error(400, "Missing run path")
                return
            try:
                payload = _load_run_detail(_resolve_under_root(run_dir))
            except Exception as exc:
                self._send_error(400, str(exc))
                return
            self._send_json(200, payload)
            return

        if parsed.path == "/api/benchmarks":
            payload = {"benchmarks": _scan_benchmarks(RUNS_DIR / "benchmarks")}
            self._send_json(200, payload)
            return

        if parsed.path == "/api/benchmark":
            params = parse_qs(parsed.query)
            bench_dir = _get_param(params, "path")
            if not bench_dir:
                self._send_error(400, "Missing benchmark path")
                return
            try:
                payload = _load_benchmark_detail(_resolve_under_root(bench_dir))
            except Exception as exc:
                self._send_error(400, str(exc))
                return
            self._send_json(200, payload)
            return

        if parsed.path == "/api/systems":
            payload = {"systems": _list_systems(SYSTEMS_DIR)}
            self._send_json(200, payload)
            return

        if parsed.path == "/api/system":
            params = parse_qs(parsed.query)
            system_path = _get_param(params, "path")
            if not system_path:
                self._send_error(400, "Missing system path")
                return
            try:
                payload = _load_system(_resolve_under_root(system_path))
            except Exception as exc:
                self._send_error(400, str(exc))
                return
            self._send_json(200, payload)
            return

        self._send_error(404, "Unknown endpoint")

    def _handle_api_post(self, parsed) -> None:
        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_error(400, f"Invalid JSON body: {exc}")
            return

        if parsed.path == "/api/system/save":
            path = (body or {}).get("path")
            system = (body or {}).get("system")
            raw_yaml = (body or {}).get("yaml")
            if not path or (system is None and raw_yaml is None):
                self._send_error(400, "Missing system payload")
                return
            try:
                payload = _save_system(_resolve_under_root(path), system, raw_yaml)
            except Exception as exc:
                self._send_error(400, str(exc))
                return
            self._send_json(200, payload)
            return

        if parsed.path == "/api/validate":
            path = (body or {}).get("path")
            if not path:
                self._send_error(400, "Missing system path")
                return
            try:
                system_path = _resolve_under_root(path)
                system_def = load_system_definition(system_path)
                payload = {
                    "ok": True,
                    "system_id": system_def.id,
                    "label": system_def.label,
                    "version": system_def.version,
                }
            except ConfigError as exc:
                self._send_json(200, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, payload)
            return

        if parsed.path == "/api/run":
            path = (body or {}).get("path")
            if not path:
                self._send_error(400, "Missing system path")
                return
            payload = (body or {}).get("input") or {}
            run_dir = (body or {}).get("run_dir") or str(RUNS_DIR)
            dry_run = bool((body or {}).get("dry_run"))
            try:
                system_path = _resolve_under_root(path)
                system_def = load_system_definition(system_path)
                if dry_run:
                    self._send_json(200, {"ok": True, "status": "dry_run"})
                    return
                engine = ExecutionEngine(system_def)
                run_root = Path(run_dir).expanduser().resolve()
                run_logger = RunLogger(run_root, system_def)
                run_logger.snapshot_system()
                run_logger.snapshot_input(payload)
                result = engine.run(payload)
                for record in result.node_results:
                    run_logger.log_node(record)
                run_logger.finalize(result, config_path=system_path)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "status": result.status,
                        "error": result.error,
                        "run_dir": str(run_logger.run_dir.relative_to(REPO_ROOT)),
                    },
                )
            except (ConfigError, ExecutionError, ValueError) as exc:
                self._send_json(200, {"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/benchmark":
            systems = (body or {}).get("systems") or []
            inputs_dir = (body or {}).get("inputs_dir")
            iterations = int((body or {}).get("iterations") or 5)
            expect_field = (body or {}).get("expect_field") or "expected_legality"
            output_field = (body or {}).get("output_field") or "is_legal"
            run_dir = (body or {}).get("run_dir") or str(RUNS_DIR)
            if not systems:
                self._send_error(400, "Missing system list")
                return
            if not inputs_dir:
                self._send_error(400, "Missing inputs_dir")
                return
            try:
                resolved_systems = [str(_resolve_under_root(path)) for path in systems]
                resolved_inputs = str(_resolve_under_root(inputs_dir))
                psf_cli.benchmark(
                    systems=resolved_systems,
                    inputs_dir=resolved_inputs,
                    iterations=iterations,
                    expect_field=expect_field,
                    output_field=output_field,
                    run_dir=run_dir,
                )
            except typer.Exit as exc:
                code = exc.exit_code if exc.exit_code is not None else 1
                self._send_json(200, {"ok": False, "error": f"Benchmark exited with code {code}"})
                return
            except Exception as exc:
                self._send_json(200, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, {"ok": True})
            return

        self._send_error(404, "Unknown endpoint")

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})


def _get_param(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    if not values:
        return None
    return values[0]


def _resolve_under_root(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_relative_to(REPO_ROOT):
        raise ValueError("Path is outside repository root")
    return candidate


def _scan_runs(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for system_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "benchmarks"):
        for run_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            summary_path = run_dir / "run.json"
            if not summary_path.exists():
                continue
            summary = _read_json(summary_path)
            started = _parse_iso(summary.get("started_at"))
            finished = _parse_iso(summary.get("finished_at"))
            duration_ms = None
            if started and finished:
                duration_ms = int((finished - started).total_seconds() * 1000)
            records.append(
                {
                    "system_id": summary.get("system_id") or system_dir.name,
                    "system_label": summary.get("system_label"),
                    "run_id": summary.get("run_id") or run_dir.name,
                    "status": summary.get("status"),
                    "error": summary.get("error"),
                    "started_at": summary.get("started_at"),
                    "finished_at": summary.get("finished_at"),
                    "duration_ms": duration_ms,
                    "metrics": summary.get("metrics") or {},
                    "models_used": summary.get("models_used") or summary.get("metrics", {}).get("models_used"),
                    "path": str(run_dir.relative_to(REPO_ROOT)),
                }
            )
    return records


def _load_run_detail(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run.json"
    if not summary_path.exists():
        raise ValueError("run.json not found in run directory")
    payload: dict[str, Any] = {"path": str(run_dir.relative_to(REPO_ROOT))}
    payload["run"] = _read_json(summary_path)
    nodes_log = run_dir / "nodes.jsonl"
    if nodes_log.exists():
        payload["nodes"] = list(_read_jsonl(nodes_log))
    system_path = run_dir / "system.yaml"
    if system_path.exists():
        payload["system"] = _read_yaml(system_path)
    input_path = run_dir / "input.json"
    if input_path.exists():
        payload["input"] = _read_json(input_path)
    return payload


def _scan_benchmarks(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for bench_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        manifest_path = bench_dir / "manifest.json"
        summary_path = bench_dir / "summary.json"
        suite_summary_path = bench_dir / "suite_summary.json"
        rec: dict[str, Any] = {
            "benchmark_id": bench_dir.name,
            "dir": str(bench_dir.relative_to(REPO_ROOT)),
            "has_manifest": manifest_path.exists(),
            "has_summary": summary_path.exists(),
            "has_suite_summary": suite_summary_path.exists(),
            "has_positions": (bench_dir / "positions.jsonl").exists(),
            "has_results": (bench_dir / "results.jsonl").exists(),
            "has_log": (bench_dir / "benchmark.log").exists() or (bench_dir / "run.log").exists(),
        }
        if rec["has_summary"] and rec["has_manifest"]:
            rec["type"] = "chess_random_suite"
        elif rec["has_suite_summary"]:
            rec["type"] = "psf_benchmark_suite"
        else:
            rec["type"] = "unknown_or_incomplete"
        if rec["has_manifest"]:
            m = _read_json(manifest_path)
            rec["task"] = m.get("task")
            rec["started_at"] = m.get("started_at")
            rec["positions"] = m.get("generated_positions")
            rec["plies"] = m.get("plies")
            rec["seed"] = m.get("seed")
            rec["illegal_ratio"] = m.get("illegal_ratio")
            rec["num_generators"] = len(m.get("generator_systems") or [])
            rec["num_validators"] = len(m.get("validator_systems") or [])
        if rec["has_summary"]:
            s = _read_json(summary_path)
            rec["finished_at"] = s.get("finished_at")
            rec["generators"] = len((s.get("generators") or {}).keys())
            rec["validators"] = len((s.get("validators") or {}).keys())
            rec["pipeline_enabled"] = bool((s.get("pipeline") or {}).get("enabled"))
        if rec["has_suite_summary"]:
            ss = _read_json(suite_summary_path)
            rec["created_at"] = ss.get("created_at")
            rec["system_count"] = ss.get("system_count")
            rec["scenario_count"] = len(ss.get("scenarios") or [])
        records.append(rec)
    return records


def _load_benchmark_detail(bench_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(bench_dir.relative_to(REPO_ROOT))}
    manifest_path = bench_dir / "manifest.json"
    summary_path = bench_dir / "summary.json"
    suite_summary_path = bench_dir / "suite_summary.json"
    if manifest_path.exists():
        payload["manifest"] = _read_json(manifest_path)
    if summary_path.exists():
        payload["summary"] = _read_json(summary_path)
    if suite_summary_path.exists():
        payload["suite_summary"] = _read_json(suite_summary_path)
    systems: list[dict[str, Any]] = []
    for system_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        summary = system_dir / "benchmark_summary.json"
        run_dirs = [p for p in system_dir.iterdir() if p.is_dir()]
        systems.append(
            {
                "system_id": system_dir.name,
                "benchmark_summary": _read_json(summary) if summary.exists() else None,
                "runs": [str(p.relative_to(REPO_ROOT)) for p in sorted(run_dirs)],
            }
        )
    if systems:
        payload["systems"] = systems
    return payload


def _list_systems(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not root.exists():
        return results
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".yaml", ".yml", ".json"}:
            results.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "name": path.name,
                }
            )
    return results


def _load_system(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = {
        "path": str(path.relative_to(REPO_ROOT)),
        "raw": text,
    }
    if path.suffix == ".json":
        payload["system"] = json.loads(text)
    else:
        payload["system"] = yaml.safe_load(text)
    return payload


def _save_system(path: Path, system: dict[str, Any] | None, raw_yaml: str | None) -> dict[str, Any]:
    if raw_yaml is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw_yaml, encoding="utf-8")
        return {"ok": True, "path": str(path.relative_to(REPO_ROOT))}
    if system is None:
        raise ValueError("Missing system data")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(system, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(system, sort_keys=False), encoding="utf-8")
    return {"ok": True, "path": str(path.relative_to(REPO_ROOT))}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_iso(dt: str | None) -> datetime | None:
    if not dt:
        return None
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None
