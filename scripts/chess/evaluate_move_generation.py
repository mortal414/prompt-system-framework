from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import sys

import chess

from psf.config_loader import load_system_definition
from psf.engine import ExecutionEngine
from psf.engine import ExecutionResult
from psf.exceptions import ExecutionError
from psf.logging import RunLogger


@dataclass(frozen=True)
class Scenario:
    name: str
    path: Path
    payload: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
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

def _sanitize_slug(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "scenario"


def _run_system_with_psf_logging(
    *,
    engine: ExecutionEngine,
    system_def: Any,
    system_path: Path,
    payload: dict[str, Any],
    runs_root: Path,
    run_id: str,
) -> tuple[str, Optional[dict[str, Any]], Optional[str], str]:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(runs_root, system_def, run_id=run_id, use_system_subdir=False)
    run_logger.snapshot_system()
    run_logger.snapshot_input(payload)
    try:
        result = engine.run(payload)
    except ExecutionError as exc:
        result = ExecutionResult(
            status="failure",
            final_output=None,
            context={"runtime": {}},
            node_results=[],
            metrics={},
            error=str(exc),
        )
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
        description="Evaluate SAN move-generation systems using python-chess as the legality oracle."
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
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")

    system_path: Path = args.system.expanduser().resolve()
    inputs_dir: Path = args.inputs_dir.expanduser().resolve()
    run_root: Path = args.run_dir.expanduser().resolve()

    system_def = load_system_definition(system_path)
    engine = ExecutionEngine(system_def)

    validator_engine: ExecutionEngine | None = None
    validator_system_def = None
    validator_path: Path | None = None
    if args.validator_system is not None:
        validator_path = args.validator_system.expanduser().resolve()
        validator_system_def = load_system_definition(validator_path)
        validator_engine = ExecutionEngine(validator_system_def)

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
    logger.info("Inputs: %s | scenarios=%s | iterations=%s", inputs_dir, len(scenarios), args.iterations)
    if validator_engine is not None and validator_system_def is not None:
        logger.info("Validator system: %s (%s)", validator_system_def.id, validator_system_def.label or "unnamed")
        logger.info("Max attempts per run: %s", args.max_attempts)
    logger.info("Artifacts: %s", benchmark_dir)

    with results_path.open("w", encoding="utf-8") as out_fh:
        for scenario in scenarios:
            base_payload = copy.deepcopy(scenario.payload)
            fen = base_payload.get("position_fen")
            if not isinstance(fen, str) or not fen.strip():
                raise SystemExit(f"{scenario.path}: missing or invalid 'position_fen'")
            notation = base_payload.get("move_notation") if isinstance(base_payload.get("move_notation"), str) else "san"
            if notation not in {"san", "uci"}:
                raise SystemExit(f"{scenario.path}: unsupported move_notation '{notation}'")

            scenario_slug = _sanitize_slug(scenario.name)
            logger.debug("Scenario %s (%s): notation=%s", scenario.name, scenario.path.name, notation)
            for iteration in range(1, args.iterations + 1):
                total_runs += 1
                illegal_moves: list[str] = []
                if isinstance(base_payload.get("illegal_moves"), list):
                    illegal_moves = [str(m) for m in base_payload["illegal_moves"]]

                attempt_records: list[dict[str, Any]] = []
                final_move: Optional[str] = None
                validator_accepted: Optional[bool] = None
                run_status = "success"
                error_detail: Optional[str] = None

                max_attempts = args.max_attempts if validator_engine is not None else 1
                for attempt in range(1, max_attempts + 1):
                    logger.debug(
                        "Run: scenario=%s iter=%s attempt=%s illegal_moves=%s",
                        scenario.name,
                        iteration,
                        attempt,
                        illegal_moves,
                    )
                    payload = {
                        "position_fen": fen,
                        "move_notation": notation,
                        "illegal_moves": illegal_moves,
                        "attempt": attempt,
                    }

                    run_id = f"{benchmark_id}-{scenario_slug}-iter{iteration:03d}-attempt{attempt:02d}-gen"
                    status, final_output, err, gen_run_dir = _run_system_with_psf_logging(
                        engine=engine,
                        system_def=system_def,
                        system_path=system_path,
                        payload=payload,
                        runs_root=generator_runs_root,
                        run_id=run_id,
                    )
                    if status != "success" or final_output is None:
                        engine_failures += 1
                        run_status = "generator_error"
                        error_detail = err
                        logger.error("Generator error (scenario=%s iter=%s): %s", scenario.name, iteration, err)
                        break

                    if args.output_field not in final_output:
                        missing_output += 1
                        run_status = "missing_output"
                        error_detail = f"final output missing '{args.output_field}'"
                        logger.error(
                            "Missing output field '%s' (scenario=%s iter=%s)",
                            args.output_field,
                            scenario.name,
                            iteration,
                        )
                        break

                    move_value = final_output.get(args.output_field)
                    if not isinstance(move_value, str) or not move_value.strip():
                        run_status = "invalid_move"
                        error_detail = f"output '{args.output_field}' must be a non-empty string"
                        logger.error(
                            "Invalid move output (scenario=%s iter=%s): %r",
                            scenario.name,
                            iteration,
                            move_value,
                        )
                        break

                    final_move = move_value.strip()
                    attempt_entry: dict[str, Any] = {
                        "attempt": attempt,
                        "move": final_move,
                        "generator_run_dir": gen_run_dir,
                    }
                    logger.debug("Generated move: %s", final_move)

                    if validator_engine is not None:
                        validator_payload: dict[str, Any] = {
                            "position_fen": fen,
                            "candidate_move": final_move,
                        }
                        v_run_id = f"{benchmark_id}-{scenario_slug}-iter{iteration:03d}-attempt{attempt:02d}-val"
                        v_status, v_output, v_err, val_run_dir = _run_system_with_psf_logging(
                            engine=validator_engine,
                            system_def=validator_system_def,
                            system_path=validator_path,
                            payload=validator_payload,
                            runs_root=validator_runs_root,
                            run_id=v_run_id,
                        )
                        if v_status != "success" or v_output is None:
                            validator_failures += 1
                            run_status = "validator_error"
                            error_detail = v_err
                            attempt_entry["validator"] = {"status": "error", "error": v_err, "run_dir": val_run_dir}
                            attempt_records.append(attempt_entry)
                            logger.error(
                                "Validator error (scenario=%s iter=%s attempt=%s): %s",
                                scenario.name,
                                iteration,
                                attempt,
                                v_err,
                            )
                            break
                        if args.validator_output_field not in v_output:
                            validator_failures += 1
                            run_status = "validator_missing_output"
                            error_detail = f"validator output missing '{args.validator_output_field}'"
                            attempt_entry["validator"] = {"status": "missing_output", "run_dir": val_run_dir}
                            attempt_records.append(attempt_entry)
                            logger.error(
                                "Validator missing output '%s' (scenario=%s iter=%s attempt=%s)",
                                args.validator_output_field,
                                scenario.name,
                                iteration,
                                attempt,
                            )
                            break
                        v_value = v_output.get(args.validator_output_field)
                        if not isinstance(v_value, bool):
                            validator_failures += 1
                            run_status = "validator_invalid_output"
                            error_detail = f"validator output '{args.validator_output_field}' must be boolean"
                            attempt_entry["validator"] = {"status": "invalid_output", "run_dir": val_run_dir}
                            attempt_records.append(attempt_entry)
                            logger.error(
                                "Validator invalid output '%s' (scenario=%s iter=%s attempt=%s): %r",
                                args.validator_output_field,
                                scenario.name,
                                iteration,
                                attempt,
                                v_value,
                            )
                            break

                        validator_accepted = bool(v_value)
                        attempt_entry["validator"] = {
                            "status": "success",
                            "is_legal": validator_accepted,
                            "run_dir": val_run_dir,
                        }
                        attempt_records.append(attempt_entry)
                        logger.debug(
                            "Validator verdict: %s",
                            "ACCEPT" if validator_accepted else "REJECT",
                        )

                        if validator_accepted:
                            accepted_by_validator += 1
                            break
                        validator_rejections += 1
                        illegal_moves.append(final_move)
                        continue

                    attempt_records.append(attempt_entry)
                    break

                if validator_engine is not None and run_status == "success" and validator_accepted is False:
                    run_status = "validator_rejected"
                    error_detail = f"validator rejected all {max_attempts} attempt(s)"
                    logger.info(
                        "Validator rejected run (scenario=%s iter=%s): attempts=%s",
                        scenario.name,
                        iteration,
                        max_attempts,
                    )

                oracle_is_legal = False
                oracle_error: Optional[str] = None
                if final_move is not None:
                    oracle_is_legal, oracle_error = _oracle_is_legal(fen=fen, move=final_move, notation=notation)
                    if oracle_error is None:
                        oracle_parseable += 1
                    if oracle_is_legal:
                        oracle_legal += 1
                    logger.debug(
                        "Oracle: legal=%s parse_error=%s",
                        oracle_is_legal,
                        oracle_error,
                    )

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
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
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
    if validator_engine is not None and validator_system_def is not None:
        print(f"Validator: {validator_system_def.id} | accepts: {accepted_by_validator} | rejections: {validator_rejections} | failures: {validator_failures}")
    print(f"Artifacts: {benchmark_dir}")


if __name__ == "__main__":
    main()
