from __future__ import annotations

import argparse
import json
import logging
import random
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chess

from psf.chess_utils import GeneratedPosition, generate_illegal_san, play_random_plies
from psf.config_loader import load_system_definition
from psf.engine import ExecutionEngine, ExecutionResult
from psf.exceptions import ExecutionError
from psf.logging import RunLogger
from psf.models import SystemDef


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    status: str
    final_output: Optional[dict[str, Any]]
    error: Optional[str]
    metrics: dict[str, Any]


def _setup_logging(*, level: str, log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("psf.chess.benchmark")
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


def _generate_benchmark_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"chess_random_{timestamp}_{suffix}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _oracle_san_legality(*, fen: str, move_san: str) -> tuple[bool, Optional[str]]:
    board = chess.Board(fen)
    try:
        parsed = board.parse_san(move_san)
    except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError) as exc:
        return False, str(exc)
    return board.is_legal(parsed), None


def _execute_with_run_logger(
    *,
    engine: ExecutionEngine,
    system_def: SystemDef,
    system_path: Path,
    payload: dict[str, Any],
    runs_root: Path,
    run_id: str,
) -> RunArtifacts:
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
    return RunArtifacts(
        run_dir=run_logger.run_dir,
        status=result.status,
        final_output=result.final_output,
        error=result.error,
        metrics=result.metrics,
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"legal", "true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"illegal", "false", "f", "0", "no", "n"}:
            return False
    raise ValueError("expected a boolean-like label (legal/illegal, true/false, 1/0)")


def _load_system(path: Path) -> tuple[SystemDef, ExecutionEngine]:
    system_def = load_system_definition(path)
    return system_def, ExecutionEngine(system_def)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-position chess benchmark suite (SAN-first). "
        "Generates N random positions and runs each once per system, with full per-node run logs."
    )
    parser.add_argument("--iterations", type=int, default=100, help="Number of random positions (games) to generate")
    parser.add_argument("--plies", type=int, default=8, help="Random legal plies to play before sampling a position")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic RNG seed for position generation")
    parser.add_argument(
        "--illegal-ratio",
        type=float,
        default=0.5,
        help="Fraction of validator-baseline cases to label illegal (0-1)",
    )
    parser.add_argument(
        "--generator-system",
        type=Path,
        default=None,
        help="Move generator system YAML/JSON (SAN).",
    )
    parser.add_argument(
        "--generator-output-field",
        type=str,
        default="move_san",
        help="System output field containing the generated SAN move",
    )
    parser.add_argument(
        "--validator-system",
        type=Path,
        default=None,
        help="Legality-check system YAML/JSON (SAN).",
    )
    parser.add_argument(
        "--validator-output-field",
        type=str,
        default="is_legal",
        help="Validator system output field containing the legality boolean",
    )
    parser.add_argument(
        "--pipeline-max-attempts",
        type=int,
        default=3,
        help="Max generator attempts per position when using generator+validator pipeline",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/benchmarks"),
        help="Root directory to store benchmark artifacts",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="auto",
        help='Optional log file path, or "auto" to write into the benchmark directory',
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log progress every N positions (0 disables)",
    )
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.plies < 0:
        raise SystemExit("--plies must be >= 0")
    if args.pipeline_max_attempts < 1:
        raise SystemExit("--pipeline-max-attempts must be >= 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if not (0.0 <= args.illegal_ratio <= 1.0):
        raise SystemExit("--illegal-ratio must be between 0 and 1")
    if args.generator_system is None and args.validator_system is None:
        raise SystemExit("Provide at least one of --generator-system or --validator-system")

    run_root: Path = args.run_dir.expanduser().resolve()
    benchmark_id = _generate_benchmark_id()
    benchmark_dir = run_root / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=False)

    log_file: Path | None = None
    if args.log_file:
        if args.log_file.strip().lower() == "auto":
            log_file = benchmark_dir / "benchmark.log"
        else:
            log_file = Path(args.log_file).expanduser().resolve()
    logger = _setup_logging(level=args.log_level, log_file=log_file)

    generator_path = args.generator_system.expanduser().resolve() if args.generator_system else None
    validator_path = args.validator_system.expanduser().resolve() if args.validator_system else None

    generator_def: SystemDef | None = None
    generator_engine: ExecutionEngine | None = None
    if generator_path is not None:
        generator_def, generator_engine = _load_system(generator_path)

    validator_def: SystemDef | None = None
    validator_engine: ExecutionEngine | None = None
    if validator_path is not None:
        validator_def, validator_engine = _load_system(validator_path)

    manifest = {
        "benchmark_id": benchmark_id,
        "task": "chess_random_suite",
        "generated_positions": args.iterations,
        "plies": args.plies,
        "seed": args.seed,
        "illegal_ratio": args.illegal_ratio,
        "pipeline_max_attempts": args.pipeline_max_attempts,
        "generator_system": str(generator_path) if generator_path else None,
        "validator_system": str(validator_path) if validator_path else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "positions": "positions.jsonl",
            "results": "results.jsonl",
            "summary": "summary.json",
            "psf_runs": "psf_runs/",
            "log": str(log_file) if log_file else None,
        },
    }
    _write_json(benchmark_dir / "manifest.json", manifest)

    positions_path = benchmark_dir / "positions.jsonl"
    results_path = benchmark_dir / "results.jsonl"
    summary_path = benchmark_dir / "summary.json"

    logger.info("Benchmark: %s", benchmark_id)
    logger.info("Positions: %s | plies=%s | seed=%s", args.iterations, args.plies, args.seed)
    if generator_def is not None:
        logger.info("Generator: %s (%s)", generator_def.id, generator_def.label or "unnamed")
    if validator_def is not None:
        logger.info("Validator: %s (%s)", validator_def.id, validator_def.label or "unnamed")
    logger.info("Artifacts: %s", benchmark_dir)

    pos_rng = random.Random(args.seed)
    case_rng = random.Random(args.seed + 1)

    # Aggregates
    validator_baseline_total = 0
    validator_baseline_correct = 0
    validator_baseline_errors = 0
    validator_on_generator_total = 0
    validator_on_generator_correct = 0
    validator_on_generator_errors = 0
    generator_total = 0
    generator_errors = 0
    generator_missing_output = 0
    generator_invalid_move = 0
    generator_oracle_parseable = 0
    generator_oracle_legal = 0
    pipeline_total = 0
    pipeline_accepts = 0
    pipeline_rejects = 0
    pipeline_false_accepts = 0
    pipeline_oracle_legal_accepts = 0
    pipeline_errors = 0
    pipeline_total_attempts = 0

    token_totals: dict[str, int] = {}

    pipeline_enabled = (
        generator_def is not None
        and generator_engine is not None
        and generator_path is not None
        and validator_def is not None
        and validator_engine is not None
        and validator_path is not None
    )

    for index in range(1, args.iterations + 1):
        position: GeneratedPosition = play_random_plies(plies=args.plies, rng=pos_rng)
        _append_jsonl(
            positions_path,
            {
                "index": index,
                "position_fen": position.fen,
                "move_history_san": position.move_history_san,
                "move_history_uci": position.move_history_uci,
            },
        )

        record: dict[str, Any] = {
            "index": index,
            "position_fen": position.fen,
            "meta": {
                "move_history_san": position.move_history_san,
                "move_history_uci": position.move_history_uci,
            },
        }

        # --- Validator baseline on random labeled moves (optional) ---
        if validator_def is not None and validator_engine is not None and validator_path is not None:
            board = chess.Board(position.fen)
            want_illegal = case_rng.random() < args.illegal_ratio
            if want_illegal:
                candidate_move = generate_illegal_san(board, case_rng)
            else:
                legal_moves = list(board.legal_moves)
                chosen = case_rng.choice(legal_moves)
                candidate_move = board.san(chosen)
            expected_is_legal, expected_error = _oracle_san_legality(fen=position.fen, move_san=candidate_move)
            expected_legality = "legal" if expected_is_legal else "illegal"

            validator_baseline_total += 1
            payload = {
                "position_fen": position.fen,
                "candidate_move": candidate_move,
                "move_notation": "san",
                "expected_legality": expected_legality,
            }
            run_id = f"pos{index:04d}_validator_baseline"
            artifacts = _execute_with_run_logger(
                engine=validator_engine,
                system_def=validator_def,
                system_path=validator_path,
                payload=payload,
                runs_root=benchmark_dir / "psf_runs" / validator_def.id,
                run_id=run_id,
            )
            token_totals[validator_def.id] = token_totals.get(validator_def.id, 0) + int(
                (artifacts.metrics or {}).get("total_tokens") or 0
            )
            baseline_entry: dict[str, Any] = {
                "candidate_move": candidate_move,
                "expected_is_legal": expected_is_legal,
                "expected_error": expected_error,
                "run_dir": str(artifacts.run_dir),
                "status": artifacts.status,
                "error": artifacts.error,
            }
            predicted_is_legal: Optional[bool] = None
            if artifacts.status == "success" and artifacts.final_output is not None:
                try:
                    predicted_is_legal = _coerce_bool(artifacts.final_output.get(args.validator_output_field))
                except Exception as exc:
                    baseline_entry["parse_error"] = str(exc)
            if predicted_is_legal is None:
                validator_baseline_errors += 1
                baseline_entry["predicted_is_legal"] = None
                baseline_entry["correct"] = False
            else:
                baseline_entry["predicted_is_legal"] = predicted_is_legal
                correct = predicted_is_legal == expected_is_legal
                baseline_entry["correct"] = correct
                if correct:
                    validator_baseline_correct += 1
            record["validator_baseline"] = baseline_entry

        # --- Generator run (optional) ---
        generator_move: Optional[str] = None
        generator_oracle_is_legal: Optional[bool] = None
        generator_oracle_error: Optional[str] = None
        generator_run_dir: Optional[str] = None

        if generator_def is not None and generator_engine is not None and generator_path is not None:
            generator_total += 1
            payload = {
                "position_fen": position.fen,
                "move_notation": "san",
                "illegal_moves": [],
                "attempt": 1,
            }
            run_id = f"pos{index:04d}_generator"
            artifacts = _execute_with_run_logger(
                engine=generator_engine,
                system_def=generator_def,
                system_path=generator_path,
                payload=payload,
                runs_root=benchmark_dir / "psf_runs" / generator_def.id,
                run_id=run_id,
            )
            token_totals[generator_def.id] = token_totals.get(generator_def.id, 0) + int(
                (artifacts.metrics or {}).get("total_tokens") or 0
            )
            generator_run_dir = str(artifacts.run_dir)
            gen_entry: dict[str, Any] = {
                "run_dir": generator_run_dir,
                "status": artifacts.status,
                "error": artifacts.error,
            }
            if artifacts.status != "success" or artifacts.final_output is None:
                generator_errors += 1
                gen_entry["move_san"] = None
                record["generator"] = gen_entry
            else:
                if args.generator_output_field not in artifacts.final_output:
                    generator_missing_output += 1
                    gen_entry["move_san"] = None
                    gen_entry["missing_output_field"] = args.generator_output_field
                    record["generator"] = gen_entry
                else:
                    move_value = artifacts.final_output.get(args.generator_output_field)
                    if not isinstance(move_value, str) or not move_value.strip():
                        generator_invalid_move += 1
                        gen_entry["move_san"] = None
                        gen_entry["invalid_output"] = move_value
                        record["generator"] = gen_entry
                    else:
                        generator_move = move_value.strip()
                        generator_oracle_is_legal, generator_oracle_error = _oracle_san_legality(
                            fen=position.fen, move_san=generator_move
                        )
                        generator_oracle_is_legal = bool(generator_oracle_is_legal)
                        generator_oracle_error = generator_oracle_error
                        if generator_oracle_error is None:
                            generator_oracle_parseable += 1
                        if generator_oracle_is_legal:
                            generator_oracle_legal += 1
                        gen_entry["move_san"] = generator_move
                        gen_entry["oracle_is_legal"] = generator_oracle_is_legal
                        gen_entry["oracle_error"] = generator_oracle_error
                        record["generator"] = gen_entry

        # --- Validator on generator move + pipeline loop (optional) ---
        if pipeline_enabled:
            pipeline_total += 1
            if generator_move is None:
                pipeline_errors += 1
                pipeline_total_attempts += 1
                record["pipeline"] = {
                    "max_attempts": args.pipeline_max_attempts,
                    "status": "error",
                    "error": "generator did not produce a usable move on attempt 1",
                    "accepted": False,
                    "accepted_move_san": None,
                    "accepted_oracle_is_legal": None,
                    "accepted_oracle_error": None,
                    "attempts": [
                        {
                            "attempt": 1,
                            "generator_run_dir": generator_run_dir,
                            "move_san": None,
                        }
                    ],
                }
            else:
                # Validator on generator attempt 1.
                validator_on_generator_total += 1
                payload = {
                    "position_fen": position.fen,
                    "candidate_move": generator_move,
                    "move_notation": "san",
                }
                run_id = f"pos{index:04d}_validator_on_generator_attempt01"
                v_artifacts = _execute_with_run_logger(
                    engine=validator_engine,
                    system_def=validator_def,
                    system_path=validator_path,
                    payload=payload,
                    runs_root=benchmark_dir / "psf_runs" / validator_def.id,
                    run_id=run_id,
                )
                token_totals[validator_def.id] = token_totals.get(validator_def.id, 0) + int(
                    (v_artifacts.metrics or {}).get("total_tokens") or 0
                )
                v_entry: dict[str, Any] = {
                    "candidate_move": generator_move,
                    "oracle_is_legal": generator_oracle_is_legal,
                    "oracle_error": generator_oracle_error,
                    "run_dir": str(v_artifacts.run_dir),
                    "status": v_artifacts.status,
                    "error": v_artifacts.error,
                }
                predicted_is_legal: Optional[bool] = None
                if v_artifacts.status == "success" and v_artifacts.final_output is not None:
                    try:
                        predicted_is_legal = _coerce_bool(v_artifacts.final_output.get(args.validator_output_field))
                    except Exception as exc:
                        v_entry["parse_error"] = str(exc)
                if predicted_is_legal is None or generator_oracle_is_legal is None:
                    validator_on_generator_errors += 1
                    v_entry["predicted_is_legal"] = None
                    v_entry["correct"] = False
                else:
                    v_entry["predicted_is_legal"] = predicted_is_legal
                    correct = predicted_is_legal == generator_oracle_is_legal
                    v_entry["correct"] = correct
                    if correct:
                        validator_on_generator_correct += 1
                record["validator_on_generator"] = v_entry

                # Pipeline: retry generator until validator accepts (or attempts exhausted).
                pipeline_attempts: list[dict[str, Any]] = []
                illegal_moves: list[str] = []
                accepted_move: Optional[str] = None
                accepted_oracle_is_legal: Optional[bool] = None
                accepted_oracle_error: Optional[str] = None
                pipeline_error: Optional[str] = None

                # Attempt 1 reuses the generator+validator results above.
                pipeline_attempts.append(
                    {
                        "attempt": 1,
                        "generator_run_dir": generator_run_dir,
                        "validator_run_dir": str(v_artifacts.run_dir),
                        "move_san": generator_move,
                        "validator_is_legal": v_entry.get("predicted_is_legal"),
                        "oracle_is_legal": generator_oracle_is_legal,
                        "oracle_error": generator_oracle_error,
                    }
                )
                pipeline_total_attempts += 1

                first_verdict = v_entry.get("predicted_is_legal")
                if first_verdict is None:
                    pipeline_errors += 1
                    pipeline_error = "validator did not produce a usable verdict on attempt 1"
                elif first_verdict is True:
                    accepted_move = generator_move
                    accepted_oracle_is_legal = generator_oracle_is_legal
                    accepted_oracle_error = generator_oracle_error
                else:
                    illegal_moves.append(generator_move)
                    for attempt in range(2, args.pipeline_max_attempts + 1):
                        payload = {
                            "position_fen": position.fen,
                            "move_notation": "san",
                            "illegal_moves": illegal_moves,
                            "attempt": attempt,
                        }
                        run_id = f"pos{index:04d}_pipeline_generator_attempt{attempt:02d}"
                        gen_artifacts = _execute_with_run_logger(
                            engine=generator_engine,
                            system_def=generator_def,
                            system_path=generator_path,
                            payload=payload,
                            runs_root=benchmark_dir / "psf_runs" / generator_def.id,
                            run_id=run_id,
                        )
                        token_totals[generator_def.id] = token_totals.get(generator_def.id, 0) + int(
                            (gen_artifacts.metrics or {}).get("total_tokens") or 0
                        )
                        move_san: Optional[str] = None
                        oracle_is_legal: Optional[bool] = None
                        oracle_error: Optional[str] = None
                        gen_error: Optional[str] = None
                        if gen_artifacts.status != "success" or gen_artifacts.final_output is None:
                            gen_error = gen_artifacts.error or "generator failed"
                        else:
                            move_value = gen_artifacts.final_output.get(args.generator_output_field)
                            if not isinstance(move_value, str) or not move_value.strip():
                                gen_error = f"invalid generator output '{args.generator_output_field}'"
                            else:
                                move_san = move_value.strip()
                                oracle_is_legal, oracle_error = _oracle_san_legality(
                                    fen=position.fen, move_san=move_san
                                )

                        # Validator call for this attempt (only if we got a move).
                        v_run_dir: Optional[str] = None
                        v_status = "skipped"
                        v_error: Optional[str] = None
                        v_pred: Optional[bool] = None
                        if move_san is not None:
                            v_payload = {
                                "position_fen": position.fen,
                                "candidate_move": move_san,
                                "move_notation": "san",
                            }
                            v_run_id = f"pos{index:04d}_pipeline_validator_attempt{attempt:02d}"
                            v_art = _execute_with_run_logger(
                                engine=validator_engine,
                                system_def=validator_def,
                                system_path=validator_path,
                                payload=v_payload,
                                runs_root=benchmark_dir / "psf_runs" / validator_def.id,
                                run_id=v_run_id,
                            )
                            token_totals[validator_def.id] = token_totals.get(validator_def.id, 0) + int(
                                (v_art.metrics or {}).get("total_tokens") or 0
                            )
                            v_run_dir = str(v_art.run_dir)
                            v_status = v_art.status
                            v_error = v_art.error
                            if v_art.status == "success" and v_art.final_output is not None:
                                try:
                                    v_pred = _coerce_bool(v_art.final_output.get(args.validator_output_field))
                                except Exception as exc:
                                    v_error = str(exc)

                        pipeline_attempts.append(
                            {
                                "attempt": attempt,
                                "generator_run_dir": str(gen_artifacts.run_dir),
                                "generator_status": gen_artifacts.status,
                                "generator_error": gen_error,
                                "validator_run_dir": v_run_dir,
                                "validator_status": v_status,
                                "validator_error": v_error,
                                "move_san": move_san,
                                "validator_is_legal": v_pred,
                                "oracle_is_legal": oracle_is_legal,
                                "oracle_error": oracle_error,
                            }
                        )
                        pipeline_total_attempts += 1

                        if gen_error is not None:
                            pipeline_errors += 1
                            pipeline_error = gen_error
                            break
                        if v_pred is None:
                            pipeline_errors += 1
                            pipeline_error = v_error or "validator did not produce a usable verdict"
                            break
                        if v_pred is True:
                            accepted_move = move_san
                            accepted_oracle_is_legal = bool(oracle_is_legal)
                            accepted_oracle_error = oracle_error
                            break
                        if move_san is not None:
                            illegal_moves.append(move_san)

                status = "accepted" if accepted_move is not None else ("error" if pipeline_error else "rejected")
                pipeline_entry: dict[str, Any] = {
                    "max_attempts": args.pipeline_max_attempts,
                    "status": status,
                    "error": pipeline_error,
                    "accepted": accepted_move is not None,
                    "accepted_move_san": accepted_move,
                    "accepted_oracle_is_legal": accepted_oracle_is_legal,
                    "accepted_oracle_error": accepted_oracle_error,
                    "attempts": pipeline_attempts,
                }
                if status == "accepted":
                    pipeline_accepts += 1
                    if accepted_oracle_is_legal is True:
                        pipeline_oracle_legal_accepts += 1
                    elif accepted_oracle_is_legal is False:
                        pipeline_false_accepts += 1
                elif status == "rejected":
                    pipeline_rejects += 1
                record["pipeline"] = pipeline_entry

        _append_jsonl(results_path, record)
        if args.progress_every > 0 and index % args.progress_every == 0:
            logger.info("Progress: %s/%s positions", index, args.iterations)

    finished_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "benchmark_id": benchmark_id,
        "task": "chess_random_suite",
        "positions": args.iterations,
        "plies": args.plies,
        "seed": args.seed,
        "illegal_ratio": args.illegal_ratio,
        "generator": {
            "system_id": getattr(generator_def, "id", None),
            "system_label": getattr(generator_def, "label", None),
            "total": generator_total,
            "errors": generator_errors,
            "missing_output": generator_missing_output,
            "invalid_move": generator_invalid_move,
            "oracle_parseable": generator_oracle_parseable,
            "oracle_legal": generator_oracle_legal,
            "oracle_parse_rate": (generator_oracle_parseable / generator_total) if generator_total else 0.0,
            "oracle_legal_rate": (generator_oracle_legal / generator_total) if generator_total else 0.0,
        },
        "validator_baseline": {
            "system_id": getattr(validator_def, "id", None),
            "system_label": getattr(validator_def, "label", None),
            "total": validator_baseline_total,
            "correct": validator_baseline_correct,
            "errors": validator_baseline_errors,
            "accuracy": (validator_baseline_correct / validator_baseline_total) if validator_baseline_total else 0.0,
        },
        "validator_on_generator": {
            "total": validator_on_generator_total,
            "correct": validator_on_generator_correct,
            "errors": validator_on_generator_errors,
            "accuracy": (validator_on_generator_correct / validator_on_generator_total) if validator_on_generator_total else 0.0,
        },
        "pipeline": {
            "enabled": pipeline_enabled,
            "max_attempts": args.pipeline_max_attempts,
            "total": pipeline_total,
            "accepts": pipeline_accepts,
            "oracle_legal_accepts": pipeline_oracle_legal_accepts,
            "rejects": pipeline_rejects,
            "errors": pipeline_errors,
            "false_accepts": pipeline_false_accepts,
            "accept_rate": (pipeline_accepts / pipeline_total) if pipeline_total else 0.0,
            "oracle_legal_rate": (pipeline_oracle_legal_accepts / pipeline_total) if pipeline_total else 0.0,
            "avg_attempts": (pipeline_total_attempts / pipeline_total) if pipeline_total else 0.0,
        },
        "token_totals": token_totals,
        "artifacts": {
            "benchmark_dir": str(benchmark_dir),
            "positions": str(positions_path),
            "results": str(results_path),
            "summary": str(summary_path),
        },
        "finished_at": finished_at,
    }
    _write_json(summary_path, summary)

    logger.info("Done.")
    if generator_total:
        logger.info(
            "Generator oracle legal rate: %.1f%% (parse rate %.1f%%)",
            summary["generator"]["oracle_legal_rate"] * 100,
            summary["generator"]["oracle_parse_rate"] * 100,
        )
    if validator_baseline_total:
        logger.info("Validator baseline accuracy: %.1f%%", summary["validator_baseline"]["accuracy"] * 100)
    if validator_on_generator_total:
        logger.info("Validator-on-generator accuracy: %.1f%%", summary["validator_on_generator"]["accuracy"] * 100)
    if pipeline_total:
        logger.info(
            "Pipeline accept rate: %.1f%% | oracle legal rate: %.1f%% (avg attempts %.2f, false accepts=%s)",
            summary["pipeline"]["accept_rate"] * 100,
            summary["pipeline"]["oracle_legal_rate"] * 100,
            summary["pipeline"]["avg_attempts"],
            summary["pipeline"]["false_accepts"],
        )
    logger.info("Artifacts: %s", benchmark_dir)


if __name__ == "__main__":
    main()
