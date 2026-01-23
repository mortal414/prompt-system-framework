from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
import sys

import chess

from psf.config_loader import load_system_definition
from psf.engine import ExecutionEngine
from psf.engine import ExecutionResult
from psf.exceptions import ExecutionError
from psf.logging import RunLogger
from psf.providers import get_provider
from psf.providers.openai_provider import OpenAIBatchProvider


@dataclass(frozen=True)
class Scenario:
    name: str
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class RunOutcome:
    run_index: int
    record: dict[str, Any]
    counters: dict[str, int]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    if "input" in data:
        payload = data.get("input")
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: 'input' must be a JSON object")
        return payload
    data.pop("expected", None)
    data.pop("meta", None)
    data.pop("id", None)
    return data


def _collect_scenarios(inputs_dir: Path) -> list[Scenario]:
    paths = sorted(p for p in inputs_dir.glob("*.json") if p.is_file() and p.name != "manifest.json")
    scenarios: list[Scenario] = []
    for path in paths:
        scenarios.append(Scenario(name=path.stem, path=path, payload=_read_json(path)))
    return scenarios


def _generate_benchmark_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"chess_benchmark_{timestamp}_{suffix}"


def _oracle_is_legal(*, fen: str, move: str, notation: str) -> tuple[bool, Optional[str]]:
    board = chess.Board(fen)
    try:
        if notation == "uci":
            parsed = board.parse_uci(move)
        else:
            parsed = board.parse_san(move)
        return board.is_legal(parsed), None
    except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError) as exc:
        return False, str(exc)


def _infer_move_output_field(*, system_def: Any, preferred_field: str) -> str:
    system_output = getattr(system_def, "system_output", None)
    fields = getattr(system_output, "fields", {}) if system_output is not None else {}
    if not isinstance(fields, dict):
        return preferred_field
    field_names = set(fields.keys())
    if preferred_field in field_names:
        return preferred_field
    for candidate in ("move", "move_san", "move_uci"):
        if candidate in field_names:
            return candidate
    move_like = sorted(name for name in field_names if name.startswith("move"))
    if len(move_like) == 1:
        return move_like[0]
    return preferred_field


def _sanitize_slug(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "scenario"


def _run_system_with_psf_logging(
    *,
    system_def: Any,
    system_path: Path,
    payload: dict[str, Any],
    runs_root: Path,
    run_id: str,
    provider_resolver: Callable[[str], Any] | None = None,
) -> tuple[str, Optional[dict[str, Any]], Optional[str], str]:
    try:
        local_engine = ExecutionEngine(system_def, provider_resolver=provider_resolver)
        result = local_engine.run(payload)
    except ExecutionError as exc:
        result = ExecutionResult(
            status="failure",
            final_output=None,
            context={"runtime": {}},
            node_results=[],
            metrics={},
            error=str(exc),
        )

    runs_root.mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(runs_root, system_def, run_id=run_id, use_system_subdir=False)
    run_logger.snapshot_system()
    run_logger.snapshot_input(payload)
    for record in result.node_results:
        run_logger.log_node(record)
    run_logger.finalize(result, config_path=system_path)
    if result.status != "success" or result.final_output is None:
        return "failure", None, result.error or "run failed without final output", str(run_logger.run_dir)
    return "success", result.final_output, None, str(run_logger.run_dir)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _setup_logging(*, level: str, log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("psf.chess.evaluate")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate move-generation systems (SAN or UCI) using python-chess as the legality oracle."
    )
    parser.add_argument("--system", required=True, type=Path, help="Move-generation system YAML/JSON")
    parser.add_argument("--inputs-dir", required=True, type=Path, help="Directory of scenario JSON files")
    parser.add_argument("--iterations", type=int, default=10, help="Runs per scenario")
    parser.add_argument(
        "--output-field",
        type=str,
        default="move_san",
        help="Final output field containing the proposed move",
    )
    parser.add_argument(
        "--validator-system",
        type=Path,
        default=None,
        help="Optional legality-check system; when set, runs generator→validator loop until validator accepts",
    )
    parser.add_argument(
        "--validator-output-field",
        type=str,
        default="is_legal",
        help="Validator system output field containing the legality boolean",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max generator attempts per iteration when using --validator-system",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/benchmarks"),
        help="Root directory to store benchmark artifacts",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N runs when --log-level is INFO/DEBUG (0 disables)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help='Optional log file path, or "auto" to write into the benchmark artifacts directory',
    )
    parser.add_argument(
        "--openai-batch",
        action="store_true",
        help="Use OpenAI Batch API for model calls (requires OpenAI provider).",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=None,
        help="Requests per OpenAI batch job (default: 2).",
    )
    parser.add_argument(
        "--openai-batch-max-wait",
        type=float,
        default=0.5,
        help="Max seconds to wait after the last queued request before flushing an underfilled batch.",
    )
    parser.add_argument(
        "--openai-batch-poll-interval",
        type=float,
        default=1.0,
        help="Poll interval (seconds) while waiting for a batch job to complete.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Worker threads when --openai-batch is enabled (0=auto).",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if args.openai_batch_size is not None and args.openai_batch_size < 2:
        raise SystemExit("--openai-batch-size must be >= 2 (omit to use default)")
    if args.openai_batch_max_wait <= 0:
        raise SystemExit("--openai-batch-max-wait must be > 0")
    if args.openai_batch_poll_interval <= 0:
        raise SystemExit("--openai-batch-poll-interval must be > 0")
    if args.max_workers < 0:
        raise SystemExit("--max-workers must be >= 0")

    system_path: Path = args.system.expanduser().resolve()
    inputs_dir: Path = args.inputs_dir.expanduser().resolve()
    run_root: Path = args.run_dir.expanduser().resolve()

    system_def = load_system_definition(system_path)
    move_output_field = _infer_move_output_field(system_def=system_def, preferred_field=args.output_field)
    system_default_notation = "san"
    if move_output_field == "move_uci" or move_output_field.endswith("_uci"):
        system_default_notation = "uci"
    elif move_output_field == "move_san" or move_output_field.endswith("_san"):
        system_default_notation = "san"
    else:
        haystack = f"{system_def.id} {system_path.name}".lower()
        if "uci" in haystack:
            system_default_notation = "uci"
        elif "san" in haystack:
            system_default_notation = "san"

    validator_system_def = None
    validator_path: Path | None = None
    if args.validator_system is not None:
        validator_path = args.validator_system.expanduser().resolve()
        validator_system_def = load_system_definition(validator_path)

    scenarios = _collect_scenarios(inputs_dir)
    if not scenarios:
        raise SystemExit(f"No scenarios found in: {inputs_dir}")

    benchmark_id = _generate_benchmark_id()
    benchmark_dir = run_root / benchmark_id / system_def.id
    _ensure_dir(benchmark_dir)

    generator_runs_root = benchmark_dir / "psf_runs" / system_def.id
    validator_runs_root = benchmark_dir / "psf_runs" / (validator_system_def.id if validator_system_def else "validator")

    log_file: Path | None = None
    if args.log_file:
        if args.log_file.strip().lower() == "auto":
            log_file = benchmark_dir / "run.log"
        else:
            log_file = Path(args.log_file).expanduser().resolve()
    logger = _setup_logging(level=args.log_level, log_file=log_file)

    openai_batch_provider: OpenAIBatchProvider | None = None
    openai_batch_size: int | None = None
    max_workers = 1
    if args.openai_batch:
        openai_batch_size = int(args.openai_batch_size or 2)
        entry_model_groups: set[tuple[str, str]] = set()
        for candidate_def in [system_def, validator_system_def]:
            if candidate_def is None:
                continue
            try:
                entry = candidate_def.get_node(candidate_def.entry_node_id)
            except Exception:
                continue
            provider_name = getattr(entry, "provider", None) or "openai"
            model_name = getattr(entry, "model", None)
            if model_name:
                entry_model_groups.add((provider_name, model_name))
        parallel_model_groups = max(1, len(entry_model_groups))
        concurrent_batches = 2
        total_tasks = len(scenarios) * args.iterations
        inferred_workers = max(openai_batch_size, 4, openai_batch_size * concurrent_batches * parallel_model_groups)
        inferred_workers = min(inferred_workers, total_tasks, 2048)
        max_workers = int(args.max_workers) if args.max_workers else int(inferred_workers)
        openai_batch_provider = OpenAIBatchProvider(
            batch_size=openai_batch_size,
            logger=logger,
            max_wait_seconds=float(args.openai_batch_max_wait),
            poll_interval_seconds=float(args.openai_batch_poll_interval),
        )

    def _provider_resolver(name: str) -> Any:
        if openai_batch_provider is not None and name == "openai":
            return openai_batch_provider
        return get_provider(name)

    results_path = benchmark_dir / "results.jsonl"
    started_at = datetime.now(timezone.utc)
    total_runs = 0
    oracle_legal = 0
    oracle_parseable = 0
    engine_failures = 0
    missing_output = 0
    validator_failures = 0
    validator_rejections = 0
    accepted_by_validator = 0

    logger.info("Benchmark id: %s", benchmark_id)
    logger.info("Generator system: %s (%s)", system_def.id, system_def.label or "unnamed")
    logger.info("Generator move output field: %s", move_output_field)
    logger.info("Inputs: %s | scenarios=%s | iterations=%s", inputs_dir, len(scenarios), args.iterations)
    if validator_system_def is not None:
        logger.info("Validator system: %s (%s)", validator_system_def.id, validator_system_def.label or "unnamed")
        logger.info("Max attempts per run: %s", args.max_attempts)
    if openai_batch_provider is not None:
        logger.info(
            "OpenAI batch enabled: batch_size=%s max_workers=%s max_wait=%.2fs poll_interval=%.2fs",
            openai_batch_size,
            max_workers,
            float(args.openai_batch_max_wait),
            float(args.openai_batch_poll_interval),
        )
    logger.info("Artifacts: %s", benchmark_dir)

    validator_enabled = validator_system_def is not None and validator_path is not None

    run_jobs: list[tuple[int, Scenario, int]] = []
    run_index = 0
    for scenario in scenarios:
        for iteration in range(1, args.iterations + 1):
            run_index += 1
            run_jobs.append((run_index, scenario, iteration))

    def _run_one(run_number: int, scenario: Scenario, iteration: int) -> RunOutcome:
        base_payload = copy.deepcopy(scenario.payload)
        fen = base_payload.get("position_fen")
        if not isinstance(fen, str) or not fen.strip():
            raise SystemExit(f"{scenario.path}: missing or invalid 'position_fen'")
        default_notation = system_default_notation
        notation = (
            base_payload.get("move_notation") if isinstance(base_payload.get("move_notation"), str) else default_notation
        )
        if notation not in {"san", "uci"}:
            raise SystemExit(f"{scenario.path}: unsupported move_notation '{notation}'")

        scenario_slug = _sanitize_slug(scenario.name)
        illegal_moves: list[str] = []
        if isinstance(base_payload.get("illegal_moves"), list):
            illegal_moves = [str(m) for m in base_payload["illegal_moves"]]

        attempt_records: list[dict[str, Any]] = []
        final_move: Optional[str] = None
        validator_accepted: Optional[bool] = None
        run_status = "success"
        error_detail: Optional[str] = None

        counters = {
            "total_runs": 1,
            "oracle_legal": 0,
            "oracle_parseable": 0,
            "generator_failures": 0,
            "missing_output": 0,
            "validator_failures": 0,
            "validator_rejections": 0,
            "validator_accepts": 0,
        }

        max_attempts = args.max_attempts if validator_enabled else 1
        for attempt in range(1, max_attempts + 1):
            payload = {
                "position_fen": fen,
                "move_notation": notation,
                "illegal_moves": illegal_moves,
                "attempt": attempt,
            }

            run_id = f"{benchmark_id}-{scenario_slug}-iter{iteration:03d}-attempt{attempt:02d}-gen"
            status, final_output, err, gen_run_dir = _run_system_with_psf_logging(
                system_def=system_def,
                system_path=system_path,
                payload=payload,
                runs_root=generator_runs_root,
                run_id=run_id,
                provider_resolver=_provider_resolver,
            )
            if status != "success" or final_output is None:
                counters["generator_failures"] += 1
                run_status = "generator_error"
                error_detail = err
                break

            if move_output_field not in final_output:
                counters["missing_output"] += 1
                run_status = "missing_output"
                error_detail = f"final output missing '{move_output_field}'"
                break

            move_value = final_output.get(move_output_field)
            if not isinstance(move_value, str) or not move_value.strip():
                run_status = "invalid_move"
                error_detail = f"output '{move_output_field}' must be a non-empty string"
                break

            final_move = move_value.strip()
            attempt_entry: dict[str, Any] = {
                "attempt": attempt,
                "move": final_move,
                "generator_run_dir": gen_run_dir,
            }

            if validator_enabled and validator_system_def is not None and validator_path is not None:
                validator_payload: dict[str, Any] = {
                    "position_fen": fen,
                    "candidate_move": final_move,
                    "move_notation": notation,
                }
                v_run_id = f"{benchmark_id}-{scenario_slug}-iter{iteration:03d}-attempt{attempt:02d}-val"
                v_status, v_output, v_err, val_run_dir = _run_system_with_psf_logging(
                    system_def=validator_system_def,
                    system_path=validator_path,
                    payload=validator_payload,
                    runs_root=validator_runs_root,
                    run_id=v_run_id,
                    provider_resolver=_provider_resolver,
                )
                if v_status != "success" or v_output is None:
                    counters["validator_failures"] += 1
                    run_status = "validator_error"
                    error_detail = v_err
                    attempt_entry["validator"] = {"status": "error", "error": v_err, "run_dir": val_run_dir}
                    attempt_records.append(attempt_entry)
                    break
                if args.validator_output_field not in v_output:
                    counters["validator_failures"] += 1
                    run_status = "validator_missing_output"
                    error_detail = f"validator output missing '{args.validator_output_field}'"
                    attempt_entry["validator"] = {"status": "missing_output", "run_dir": val_run_dir}
                    attempt_records.append(attempt_entry)
                    break
                v_value = v_output.get(args.validator_output_field)
                if not isinstance(v_value, bool):
                    counters["validator_failures"] += 1
                    run_status = "validator_invalid_output"
                    error_detail = f"validator output '{args.validator_output_field}' must be boolean"
                    attempt_entry["validator"] = {"status": "invalid_output", "run_dir": val_run_dir}
                    attempt_records.append(attempt_entry)
                    break

                validator_accepted = bool(v_value)
                attempt_entry["validator"] = {
                    "status": "success",
                    "is_legal": validator_accepted,
                    "run_dir": val_run_dir,
                }
                attempt_records.append(attempt_entry)

                if validator_accepted:
                    counters["validator_accepts"] += 1
                    break
                counters["validator_rejections"] += 1
                illegal_moves.append(final_move)
                continue

            attempt_records.append(attempt_entry)
            break

        if validator_enabled and run_status == "success" and validator_accepted is False:
            run_status = "validator_rejected"
            error_detail = f"validator rejected all {max_attempts} attempt(s)"

        oracle_is_legal = False
        oracle_error: Optional[str] = None
        if final_move is not None:
            oracle_is_legal, oracle_error = _oracle_is_legal(fen=fen, move=final_move, notation=notation)
            if oracle_error is None:
                counters["oracle_parseable"] += 1
            if oracle_is_legal:
                counters["oracle_legal"] += 1

        record = {
            "scenario": scenario.name,
            "path": str(scenario.path),
            "iteration": iteration,
            "status": run_status,
            "error": error_detail,
            "move_notation": notation,
            "final_move": final_move,
            "oracle_is_legal": oracle_is_legal,
            "oracle_error": oracle_error,
            "validator_accepted": validator_accepted,
            "attempts": attempt_records,
        }
        return RunOutcome(run_index=run_number, record=record, counters=counters)

    def _consume_outcome(outcome: RunOutcome, out_fh: Any) -> None:
        nonlocal total_runs, oracle_legal, oracle_parseable, engine_failures, missing_output, validator_failures, validator_rejections, accepted_by_validator
        out_fh.write(json.dumps(outcome.record, ensure_ascii=False) + "\n")
        total_runs += int(outcome.counters.get("total_runs", 0))
        oracle_legal += int(outcome.counters.get("oracle_legal", 0))
        oracle_parseable += int(outcome.counters.get("oracle_parseable", 0))
        engine_failures += int(outcome.counters.get("generator_failures", 0))
        missing_output += int(outcome.counters.get("missing_output", 0))
        validator_failures += int(outcome.counters.get("validator_failures", 0))
        validator_rejections += int(outcome.counters.get("validator_rejections", 0))
        accepted_by_validator += int(outcome.counters.get("validator_accepts", 0))

    with results_path.open("w", encoding="utf-8") as out_fh:
        if max_workers > 1:
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for run_number, scenario, iteration in run_jobs:
                    futures.append(executor.submit(_run_one, run_number, scenario, iteration))
                pending: dict[int, RunOutcome] = {}
                next_to_write = 1
                completed = 0
                for future in as_completed(futures):
                    outcome = future.result()
                    pending[outcome.run_index] = outcome
                    completed += 1
                    while next_to_write in pending:
                        _consume_outcome(pending.pop(next_to_write), out_fh)
                        if args.progress_every > 0 and total_runs % args.progress_every == 0:
                            logger.info("Progress: %s runs complete", total_runs)
                        next_to_write += 1
        else:
            for run_number, scenario, iteration in run_jobs:
                _consume_outcome(_run_one(run_number, scenario, iteration), out_fh)
                if args.progress_every > 0 and total_runs % args.progress_every == 0:
                    logger.info("Progress: %s runs complete", total_runs)

    finished_at = datetime.now(timezone.utc)
    summary = {
        "benchmark_id": benchmark_id,
        "task": "move_generation",
        "system_id": system_def.id,
        "system_label": system_def.label,
        "system_version": system_def.version,
        "system_path": str(system_path),
        "move_output_field": move_output_field,
        "inputs_dir": str(inputs_dir),
        "iterations_per_scenario": args.iterations,
        "scenario_count": len(scenarios),
        "total_runs": total_runs,
        "oracle_parseable": oracle_parseable,
        "oracle_legal": oracle_legal,
        "oracle_parse_rate": (oracle_parseable / total_runs) if total_runs else 0.0,
        "oracle_legal_rate": (oracle_legal / total_runs) if total_runs else 0.0,
        "generator_failures": engine_failures,
        "missing_output": missing_output,
        "validator_system_path": str(validator_path) if validator_path else None,
        "validator_system_id": getattr(validator_system_def, "id", None),
        "validator_failures": validator_failures,
        "validator_rejections": validator_rejections,
        "validator_accepts": accepted_by_validator,
        "openai_batch": {
            "enabled": bool(openai_batch_provider is not None),
            "batch_size": openai_batch_size,
            "max_wait_seconds": float(args.openai_batch_max_wait),
            "poll_interval_seconds": float(args.openai_batch_poll_interval),
            "max_workers": max_workers,
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "artifacts": {
            "results": str(results_path),
            "log": str(log_file) if log_file else None,
            "psf_runs": str(benchmark_dir / "psf_runs"),
        },
    }
    _write_json(benchmark_dir / "summary.json", summary)

    parse_pct = summary["oracle_parse_rate"] * 100
    legal_pct = summary["oracle_legal_rate"] * 100
    logger.info("Oracle parse rate: %.1f%% | Oracle legal rate: %.1f%%", parse_pct, legal_pct)
    print(f"Benchmark: {benchmark_id}")
    print(f"System: {system_def.id} ({system_def.label or 'unnamed'})")
    print(f"Scenarios: {len(scenarios)} | Iterations: {args.iterations} | Total runs: {total_runs}")
    print(f"Oracle parse rate: {parse_pct:.1f}% | Oracle legal rate: {legal_pct:.1f}%")
    if validator_system_def is not None:
        print(f"Validator: {validator_system_def.id} | accepts: {accepted_by_validator} | rejections: {validator_rejections} | failures: {validator_failures}")
    print(f"Artifacts: {benchmark_dir}")


if __name__ == "__main__":
    main()
