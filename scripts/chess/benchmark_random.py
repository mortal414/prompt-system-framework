from __future__ import annotations

import argparse
import json
import logging
import random
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import chess

from psf.chess_utils import GeneratedPosition, generate_illegal_san, generate_illegal_uci, play_random_plies
from psf.config_loader import load_system_definition
from psf.engine import ExecutionEngine, ExecutionResult
from psf.exceptions import ExecutionError
from psf.logging import RunLogger
from psf.models import SystemDef
from psf.providers import get_provider
from psf.providers.openai_provider import OpenAIBatchProvider


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    status: str
    final_output: Optional[dict[str, Any]]
    error: Optional[str]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class GeneratorSystem:
    system_path: Path
    system_def: SystemDef
    move_notation: str
    move_output_field: str


@dataclass(frozen=True)
class ValidatorSystem:
    system_path: Path
    system_def: SystemDef
    move_notation: str


@dataclass(frozen=True)
class PositionOutcome:
    index: int
    record: dict[str, Any]
    token_totals: dict[str, int]
    validator_baseline: dict[str, dict[str, int]]
    generator: dict[str, dict[str, int]]
    validator_on_generator: dict[str, dict[str, int]]
    pipeline: dict[str, int]


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


def _oracle_uci_legality(*, fen: str, move_uci: str) -> tuple[bool, Optional[str]]:
    board = chess.Board(fen)
    try:
        parsed = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        return False, str(exc)
    return board.is_legal(parsed), None


def _execute_with_run_logger(
    *,
    system_def: SystemDef,
    system_path: Path,
    payload: dict[str, Any],
    runs_root: Path,
    run_id: str,
    provider_resolver: Callable[[str], Any] | None = None,
) -> RunArtifacts:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(runs_root, system_def, run_id=run_id, use_system_subdir=False)
    run_logger.snapshot_system()
    run_logger.snapshot_input(payload)
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


def _load_system(path: Path) -> SystemDef:
    return load_system_definition(path)


def _infer_move_notation(*, system_def: SystemDef, system_path: Path, default: str) -> str:
    try:
        output_fields = set(system_def.system_output.fields.keys())
    except Exception:
        output_fields = set()
    if "move_uci" in output_fields and "move_san" not in output_fields:
        return "uci"
    if "move_san" in output_fields and "move_uci" not in output_fields:
        return "san"
    haystack = f"{system_def.id} {system_path.name}".lower()
    if "uci" in haystack:
        return "uci"
    if "san" in haystack:
        return "san"
    return default


def _infer_move_output_field(*, system_def: SystemDef, preferred_field: str) -> str:
    fields = set(system_def.system_output.fields.keys())
    if preferred_field in fields:
        return preferred_field
    for candidate in ("move", "move_san", "move_uci"):
        if candidate in fields:
            return candidate
    move_like = sorted(name for name in fields if name.startswith("move"))
    if len(move_like) == 1:
        return move_like[0]
    return preferred_field


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
        action="append",
        default=[],
        help="Move generator system YAML/JSON (repeatable).",
    )
    parser.add_argument(
        "--generator-output-field",
        type=str,
        default="move_san",
        help="System output field containing the generated move (SAN or UCI)",
    )
    parser.add_argument(
        "--validator-system",
        type=Path,
        action="append",
        default=[],
        help="Legality-check system YAML/JSON (repeatable).",
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
    parser.add_argument(
        "--openai-batch",
        action="store_true",
        help="Use OpenAI Batch API for model calls (requires OpenAI provider).",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=None,
        help="Requests per OpenAI batch job (default: max of system benchmark_batch_size).",
    )
    parser.add_argument(
        "--openai-batch-max-wait",
        type=float,
        default=0.5,
        help="Max seconds to wait to fill a batch before flushing.",
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
    if args.plies < 0:
        raise SystemExit("--plies must be >= 0")
    if args.pipeline_max_attempts < 1:
        raise SystemExit("--pipeline-max-attempts must be >= 1")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if not (0.0 <= args.illegal_ratio <= 1.0):
        raise SystemExit("--illegal-ratio must be between 0 and 1")
    if not args.generator_system and not args.validator_system:
        raise SystemExit("Provide at least one of --generator-system or --validator-system")
    if args.openai_batch_size is not None and args.openai_batch_size < 2:
        raise SystemExit("--openai-batch-size must be >= 2 (or omit for auto)")
    if args.openai_batch_max_wait <= 0:
        raise SystemExit("--openai-batch-max-wait must be > 0")
    if args.openai_batch_poll_interval <= 0:
        raise SystemExit("--openai-batch-poll-interval must be > 0")
    if args.max_workers < 0:
        raise SystemExit("--max-workers must be >= 0")

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

    generator_systems: list[GeneratorSystem] = []
    for path in args.generator_system:
        resolved = path.expanduser().resolve()
        system_def = _load_system(resolved)
        notation = _infer_move_notation(system_def=system_def, system_path=resolved, default="san")
        move_output_field = _infer_move_output_field(system_def=system_def, preferred_field=args.generator_output_field)
        generator_systems.append(
            GeneratorSystem(
                system_path=resolved,
                system_def=system_def,
                move_notation=notation,
                move_output_field=move_output_field,
            )
        )

    validator_systems: list[ValidatorSystem] = []
    for path in args.validator_system:
        resolved = path.expanduser().resolve()
        system_def = _load_system(resolved)
        notation = _infer_move_notation(system_def=system_def, system_path=resolved, default="san")
        validator_systems.append(
            ValidatorSystem(system_path=resolved, system_def=system_def, move_notation=notation)
        )

    openai_batch_provider: OpenAIBatchProvider | None = None
    openai_batch_size: int | None = None
    max_workers = 1
    if args.openai_batch:
        configured_sizes: list[int] = []
        for system in [*generator_systems, *validator_systems]:
            configured_sizes.append(int(getattr(system.system_def, "benchmark_batch_size", 1) or 1))
        inferred_size = max(configured_sizes) if configured_sizes else 2
        openai_batch_size = int(args.openai_batch_size or max(2, inferred_size))
        max_workers = int(args.max_workers) if args.max_workers else max(openai_batch_size, 4)
        openai_batch_provider = OpenAIBatchProvider(
            batch_size=openai_batch_size,
            max_wait_seconds=float(args.openai_batch_max_wait),
            poll_interval_seconds=float(args.openai_batch_poll_interval),
        )

    def _provider_resolver(name: str) -> Any:
        if openai_batch_provider is not None and name == "openai":
            return openai_batch_provider
        return get_provider(name)

    manifest = {
        "benchmark_id": benchmark_id,
        "task": "chess_random_suite",
        "generated_positions": args.iterations,
        "plies": args.plies,
        "seed": args.seed,
        "illegal_ratio": args.illegal_ratio,
        "pipeline_max_attempts": args.pipeline_max_attempts,
        "generator_systems": [
            {
                "path": str(s.system_path),
                "id": s.system_def.id,
                "label": s.system_def.label,
                "move_notation": s.move_notation,
                "move_output_field": s.move_output_field,
            }
            for s in generator_systems
        ],
        "validator_systems": [
            {
                "path": str(s.system_path),
                "id": s.system_def.id,
                "label": s.system_def.label,
                "move_notation": s.move_notation,
            }
            for s in validator_systems
        ],
        "openai_batch": {
            "enabled": bool(openai_batch_provider is not None),
            "batch_size": openai_batch_size,
            "max_wait_seconds": float(args.openai_batch_max_wait),
            "poll_interval_seconds": float(args.openai_batch_poll_interval),
            "max_workers": max_workers,
        },
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
    if generator_systems:
        logger.info(
            "Generators: %s",
            ", ".join(f"{s.system_def.id}({s.move_notation},{s.move_output_field})" for s in generator_systems),
        )
    if validator_systems:
        logger.info(
            "Validators: %s",
            ", ".join(f"{s.system_def.id}({s.move_notation})" for s in validator_systems),
        )
    if openai_batch_provider is not None:
        logger.info(
            "OpenAI batch enabled: batch_size=%s max_workers=%s max_wait=%.2fs poll_interval=%.2fs",
            openai_batch_size,
            max_workers,
            float(args.openai_batch_max_wait),
            float(args.openai_batch_poll_interval),
        )
    logger.info("Artifacts: %s", benchmark_dir)

    pos_rng = random.Random(args.seed)
    case_rng = random.Random(args.seed + 1)

    positions: list[GeneratedPosition] = []
    baseline_candidates: dict[int, tuple[str, str]] = {}
    for index in range(1, args.iterations + 1):
        position = play_random_plies(plies=args.plies, rng=pos_rng)
        positions.append(position)
        _append_jsonl(
            positions_path,
            {
                "index": index,
                "position_fen": position.fen,
                "move_history_san": position.move_history_san,
                "move_history_uci": position.move_history_uci,
            },
        )
        if validator_systems:
            board = chess.Board(position.fen)
            want_illegal = case_rng.random() < args.illegal_ratio
            if want_illegal:
                candidate_san = generate_illegal_san(board, case_rng)
                candidate_uci = generate_illegal_uci(board, case_rng)
            else:
                legal_moves = list(board.legal_moves)
                chosen = case_rng.choice(legal_moves)
                candidate_san = board.san(chosen)
                candidate_uci = chosen.uci()
            baseline_candidates[index] = (candidate_san, candidate_uci)

    # Aggregates
    validator_baseline: dict[str, dict[str, int]] = {
        s.system_def.id: {"total": 0, "correct": 0, "errors": 0} for s in validator_systems
    }
    validator_on_generator: dict[str, dict[str, int]] = {}
    generator: dict[str, dict[str, int]] = {
        s.system_def.id: {
            "total": 0,
            "errors": 0,
            "missing_output": 0,
            "invalid_move": 0,
            "oracle_parseable": 0,
            "oracle_legal": 0,
        }
        for s in generator_systems
    }
    pipeline_total = 0
    pipeline_accepts = 0
    pipeline_rejects = 0
    pipeline_false_accepts = 0
    pipeline_oracle_legal_accepts = 0
    pipeline_errors = 0
    pipeline_total_attempts = 0

    token_totals: dict[str, int] = {}

    pipeline_enabled = (
        len(generator_systems) == 1
        and len(validator_systems) == 1
        and generator_systems[0].move_notation == validator_systems[0].move_notation
    )
    if generator_systems and validator_systems and not pipeline_enabled:
        if len(generator_systems) != 1 or len(validator_systems) != 1:
            logger.info("Pipeline mode disabled (requires exactly 1 generator and 1 validator).")
        else:
            logger.info(
                "Pipeline mode disabled (generator notation=%s, validator notation=%s).",
                generator_systems[0].move_notation,
                validator_systems[0].move_notation,
            )

    def _oracle_legality(*, fen: str, move: str, notation: str) -> tuple[bool, Optional[str]]:
        if notation == "uci":
            return _oracle_uci_legality(fen=fen, move_uci=move)
        return _oracle_san_legality(fen=fen, move_san=move)

    def _merge_nested_counts(target: dict[str, dict[str, int]], inc: dict[str, dict[str, int]]) -> None:
        for system_id, stats in inc.items():
            bucket = target.setdefault(system_id, {})
            for key, value in stats.items():
                bucket[key] = bucket.get(key, 0) + int(value)

    def _merge_token_totals(target: dict[str, int], inc: dict[str, int]) -> None:
        for system_id, value in inc.items():
            target[system_id] = target.get(system_id, 0) + int(value)

    def _evaluate_position(index: int, position: GeneratedPosition) -> PositionOutcome:
        local_token_totals: dict[str, int] = {}
        local_validator_baseline: dict[str, dict[str, int]] = {}
        local_generator: dict[str, dict[str, int]] = {}
        local_validator_on_generator: dict[str, dict[str, int]] = {}
        local_pipeline: dict[str, int] = {
            "total": 0,
            "accepts": 0,
            "rejects": 0,
            "errors": 0,
            "false_accepts": 0,
            "oracle_legal_accepts": 0,
            "total_attempts": 0,
        }

        record: dict[str, Any] = {
            "index": index,
            "position_fen": position.fen,
            "meta": {
                "move_history_san": position.move_history_san,
                "move_history_uci": position.move_history_uci,
            },
        }

        # --- Validator baseline on random labeled moves (optional) ---
        if validator_systems:
            candidate_san, candidate_uci = baseline_candidates[index]
            record["validator_baseline"] = {}
            for validator in validator_systems:
                candidate_move = candidate_uci if validator.move_notation == "uci" else candidate_san
                expected_is_legal, expected_error = _oracle_legality(
                    fen=position.fen, move=candidate_move, notation=validator.move_notation
                )
                expected_legality = "legal" if expected_is_legal else "illegal"

                local_validator_baseline.setdefault(validator.system_def.id, {"total": 0, "correct": 0, "errors": 0})
                local_validator_baseline[validator.system_def.id]["total"] += 1
                payload = {
                    "position_fen": position.fen,
                    "candidate_move": candidate_move,
                    "move_notation": validator.move_notation,
                    "expected_legality": expected_legality,
                }
                run_id = f"pos{index:04d}_validator_baseline"
                artifacts = _execute_with_run_logger(
                    system_def=validator.system_def,
                    system_path=validator.system_path,
                    payload=payload,
                    runs_root=benchmark_dir / "psf_runs" / validator.system_def.id,
                    run_id=run_id,
                    provider_resolver=_provider_resolver,
                )
                local_token_totals[validator.system_def.id] = local_token_totals.get(validator.system_def.id, 0) + int(
                    (artifacts.metrics or {}).get("total_tokens") or 0
                )
                baseline_entry: dict[str, Any] = {
                    "candidate_move": candidate_move,
                    "move_notation": validator.move_notation,
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
                    local_validator_baseline[validator.system_def.id]["errors"] += 1
                    baseline_entry["predicted_is_legal"] = None
                    baseline_entry["correct"] = False
                else:
                    baseline_entry["predicted_is_legal"] = predicted_is_legal
                    correct = predicted_is_legal == expected_is_legal
                    baseline_entry["correct"] = correct
                    if correct:
                        local_validator_baseline[validator.system_def.id]["correct"] += 1
                record["validator_baseline"][validator.system_def.id] = baseline_entry

        # --- Generator run (optional) ---
        if generator_systems:
            record["generator"] = {}
        for generator_system in generator_systems:
            local_generator.setdefault(
                generator_system.system_def.id,
                {
                    "total": 0,
                    "errors": 0,
                    "missing_output": 0,
                    "invalid_move": 0,
                    "oracle_parseable": 0,
                    "oracle_legal": 0,
                },
            )
            local_generator[generator_system.system_def.id]["total"] += 1
            payload = {
                "position_fen": position.fen,
                "move_notation": generator_system.move_notation,
                "illegal_moves": [],
                "attempt": 1,
            }
            run_id = f"pos{index:04d}_generator"
            artifacts = _execute_with_run_logger(
                system_def=generator_system.system_def,
                system_path=generator_system.system_path,
                payload=payload,
                runs_root=benchmark_dir / "psf_runs" / generator_system.system_def.id,
                run_id=run_id,
                provider_resolver=_provider_resolver,
            )
            local_token_totals[generator_system.system_def.id] = local_token_totals.get(
                generator_system.system_def.id, 0
            ) + int((artifacts.metrics or {}).get("total_tokens") or 0)

            gen_entry: dict[str, Any] = {
                "run_dir": str(artifacts.run_dir),
                "status": artifacts.status,
                "error": artifacts.error,
                "move_notation": generator_system.move_notation,
                "move_output_field": generator_system.move_output_field,
            }
            if artifacts.status != "success" or artifacts.final_output is None:
                local_generator[generator_system.system_def.id]["errors"] += 1
                gen_entry["move"] = None
                record["generator"][generator_system.system_def.id] = gen_entry
                continue
            if generator_system.move_output_field not in artifacts.final_output:
                local_generator[generator_system.system_def.id]["missing_output"] += 1
                gen_entry["move"] = None
                gen_entry["missing_output_field"] = generator_system.move_output_field
                record["generator"][generator_system.system_def.id] = gen_entry
                continue
            move_value = artifacts.final_output.get(generator_system.move_output_field)
            if not isinstance(move_value, str) or not move_value.strip():
                local_generator[generator_system.system_def.id]["invalid_move"] += 1
                gen_entry["move"] = None
                gen_entry["invalid_output"] = move_value
                record["generator"][generator_system.system_def.id] = gen_entry
                continue
            move = move_value.strip()
            oracle_is_legal, oracle_error = _oracle_legality(
                fen=position.fen, move=move, notation=generator_system.move_notation
            )
            if oracle_error is None:
                local_generator[generator_system.system_def.id]["oracle_parseable"] += 1
            if oracle_is_legal:
                local_generator[generator_system.system_def.id]["oracle_legal"] += 1
            gen_entry["move"] = move
            if generator_system.move_notation == "uci":
                gen_entry["move_uci"] = move
            else:
                gen_entry["move_san"] = move
            gen_entry["oracle_is_legal"] = bool(oracle_is_legal)
            gen_entry["oracle_error"] = oracle_error
            record["generator"][generator_system.system_def.id] = gen_entry

        # --- Validator on generator move + pipeline loop (optional) ---
        if pipeline_enabled:
            pipeline_notation = generator_systems[0].move_notation
            pipeline_move_output_field = generator_systems[0].move_output_field
            generator_def = generator_systems[0].system_def
            generator_path = generator_systems[0].system_path
            validator_def = validator_systems[0].system_def
            validator_path = validator_systems[0].system_path

            local_validator_on_generator.setdefault(validator_def.id, {"total": 0, "correct": 0, "errors": 0})
            local_pipeline["total"] += 1

            generator_move: Optional[str] = None
            generator_oracle_is_legal: Optional[bool] = None
            generator_oracle_error: Optional[str] = None
            generator_run_dir: Optional[str] = None
            if record.get("generator"):
                only_gen_id = generator_def.id
                gen_entry = record["generator"].get(only_gen_id, {})
                generator_move = gen_entry.get("move")
                generator_oracle_is_legal = gen_entry.get("oracle_is_legal")
                generator_oracle_error = gen_entry.get("oracle_error")
                generator_run_dir = gen_entry.get("run_dir")

            if generator_move is None:
                local_pipeline["errors"] += 1
                local_pipeline["total_attempts"] += 1
                pipeline_entry: dict[str, Any] = {
                    "max_attempts": args.pipeline_max_attempts,
                    "status": "error",
                    "error": "generator did not produce a usable move on attempt 1",
                    "accepted": False,
                    "move_notation": pipeline_notation,
                    "accepted_move": None,
                    "accepted_oracle_is_legal": None,
                    "accepted_oracle_error": None,
                    "attempts": [
                        {
                            "attempt": 1,
                            "generator_run_dir": generator_run_dir,
                            "move": None,
                        }
                    ],
                }
                if pipeline_notation == "uci":
                    pipeline_entry["accepted_move_uci"] = None
                else:
                    pipeline_entry["accepted_move_san"] = None
                record["pipeline"] = pipeline_entry
            else:
                # Validator on generator attempt 1.
                local_validator_on_generator[validator_def.id]["total"] += 1
                payload = {
                    "position_fen": position.fen,
                    "candidate_move": generator_move,
                    "move_notation": pipeline_notation,
                }
                run_id = f"pos{index:04d}_validator_on_generator_attempt01"
                v_artifacts = _execute_with_run_logger(
                    system_def=validator_def,
                    system_path=validator_path,
                    payload=payload,
                    runs_root=benchmark_dir / "psf_runs" / validator_def.id,
                    run_id=run_id,
                    provider_resolver=_provider_resolver,
                )
                local_token_totals[validator_def.id] = local_token_totals.get(validator_def.id, 0) + int(
                    (v_artifacts.metrics or {}).get("total_tokens") or 0
                )
                v_entry: dict[str, Any] = {
                    "candidate_move": generator_move,
                    "move_notation": pipeline_notation,
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
                    local_validator_on_generator[validator_def.id]["errors"] += 1
                    v_entry["predicted_is_legal"] = None
                    v_entry["correct"] = False
                else:
                    v_entry["predicted_is_legal"] = predicted_is_legal
                    correct = predicted_is_legal == generator_oracle_is_legal
                    v_entry["correct"] = correct
                    if correct:
                        local_validator_on_generator[validator_def.id]["correct"] += 1
                record["validator_on_generator"] = v_entry

                # Pipeline: retry generator until validator accepts (or attempts exhausted).
                pipeline_attempts: list[dict[str, Any]] = []
                illegal_moves: list[str] = []
                accepted_move: Optional[str] = None
                accepted_oracle_is_legal: Optional[bool] = None
                accepted_oracle_error: Optional[str] = None
                pipeline_error: Optional[str] = None

                pipeline_attempts.append(
                    {
                        "attempt": 1,
                        "generator_run_dir": generator_run_dir,
                        "validator_run_dir": str(v_artifacts.run_dir),
                        "move": generator_move,
                        "move_notation": pipeline_notation,
                        "validator_is_legal": v_entry.get("predicted_is_legal"),
                        "oracle_is_legal": generator_oracle_is_legal,
                        "oracle_error": generator_oracle_error,
                    }
                )
                local_pipeline["total_attempts"] += 1

                first_verdict = v_entry.get("predicted_is_legal")
                if first_verdict is None:
                    local_pipeline["errors"] += 1
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
                            "move_notation": pipeline_notation,
                            "illegal_moves": illegal_moves,
                            "attempt": attempt,
                        }
                        run_id = f"pos{index:04d}_pipeline_generator_attempt{attempt:02d}"
                        gen_artifacts = _execute_with_run_logger(
                            system_def=generator_def,
                            system_path=generator_path,
                            payload=payload,
                            runs_root=benchmark_dir / "psf_runs" / generator_def.id,
                            run_id=run_id,
                            provider_resolver=_provider_resolver,
                        )
                        local_token_totals[generator_def.id] = local_token_totals.get(generator_def.id, 0) + int(
                            (gen_artifacts.metrics or {}).get("total_tokens") or 0
                        )
                        move: Optional[str] = None
                        oracle_is_legal: Optional[bool] = None
                        oracle_error: Optional[str] = None
                        gen_error: Optional[str] = None
                        if gen_artifacts.status != "success" or gen_artifacts.final_output is None:
                            gen_error = gen_artifacts.error or "generator failed"
                        else:
                            move_value = gen_artifacts.final_output.get(pipeline_move_output_field)
                            if not isinstance(move_value, str) or not move_value.strip():
                                gen_error = f"invalid generator output '{pipeline_move_output_field}'"
                            else:
                                move = move_value.strip()
                                oracle_is_legal, oracle_error = _oracle_legality(
                                    fen=position.fen, move=move, notation=pipeline_notation
                                )

                        # Validator call for this attempt (only if we got a move).
                        v_run_dir: Optional[str] = None
                        v_status = "skipped"
                        v_error: Optional[str] = None
                        v_pred: Optional[bool] = None
                        if move is not None:
                            v_payload = {
                                "position_fen": position.fen,
                                "candidate_move": move,
                                "move_notation": pipeline_notation,
                            }
                            v_run_id = f"pos{index:04d}_pipeline_validator_attempt{attempt:02d}"
                            v_art = _execute_with_run_logger(
                                system_def=validator_def,
                                system_path=validator_path,
                                payload=v_payload,
                                runs_root=benchmark_dir / "psf_runs" / validator_def.id,
                                run_id=v_run_id,
                                provider_resolver=_provider_resolver,
                            )
                            local_token_totals[validator_def.id] = local_token_totals.get(validator_def.id, 0) + int(
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

                        attempt_entry: dict[str, Any] = {
                            "attempt": attempt,
                            "generator_run_dir": str(gen_artifacts.run_dir),
                            "generator_status": gen_artifacts.status,
                            "generator_error": gen_error,
                            "validator_run_dir": v_run_dir,
                            "validator_status": v_status,
                            "validator_error": v_error,
                            "move": move,
                            "move_notation": pipeline_notation,
                            "validator_is_legal": v_pred,
                            "oracle_is_legal": oracle_is_legal,
                            "oracle_error": oracle_error,
                        }
                        if pipeline_notation == "uci":
                            attempt_entry["move_uci"] = move
                        else:
                            attempt_entry["move_san"] = move
                        pipeline_attempts.append(attempt_entry)
                        local_pipeline["total_attempts"] += 1

                        if gen_error is not None:
                            local_pipeline["errors"] += 1
                            pipeline_error = gen_error
                            break
                        if v_pred is None:
                            local_pipeline["errors"] += 1
                            pipeline_error = v_error or "validator did not produce a usable verdict"
                            break
                        if v_pred is True:
                            accepted_move = move
                            accepted_oracle_is_legal = bool(oracle_is_legal)
                            accepted_oracle_error = oracle_error
                            break
                        if move is not None:
                            illegal_moves.append(move)

                status = "accepted" if accepted_move is not None else ("error" if pipeline_error else "rejected")
                pipeline_entry = {
                    "max_attempts": args.pipeline_max_attempts,
                    "status": status,
                    "error": pipeline_error,
                    "accepted": accepted_move is not None,
                    "move_notation": pipeline_notation,
                    "accepted_move": accepted_move,
                    "accepted_oracle_is_legal": accepted_oracle_is_legal,
                    "accepted_oracle_error": accepted_oracle_error,
                    "attempts": pipeline_attempts,
                }
                if pipeline_notation == "uci":
                    pipeline_entry["accepted_move_uci"] = accepted_move
                else:
                    pipeline_entry["accepted_move_san"] = accepted_move
                if status == "accepted":
                    local_pipeline["accepts"] += 1
                    if accepted_oracle_is_legal is True:
                        local_pipeline["oracle_legal_accepts"] += 1
                    elif accepted_oracle_is_legal is False:
                        local_pipeline["false_accepts"] += 1
                elif status == "rejected":
                    local_pipeline["rejects"] += 1
                record["pipeline"] = pipeline_entry

        return PositionOutcome(
            index=index,
            record=record,
            token_totals=local_token_totals,
            validator_baseline=local_validator_baseline,
            generator=local_generator,
            validator_on_generator=local_validator_on_generator,
            pipeline=local_pipeline,
        )

    def _consume_outcome(outcome: PositionOutcome) -> None:
        _merge_token_totals(token_totals, outcome.token_totals)
        _merge_nested_counts(validator_baseline, outcome.validator_baseline)
        _merge_nested_counts(generator, outcome.generator)
        _merge_nested_counts(validator_on_generator, outcome.validator_on_generator)
        nonlocal pipeline_total, pipeline_accepts, pipeline_rejects, pipeline_false_accepts, pipeline_oracle_legal_accepts, pipeline_errors, pipeline_total_attempts
        pipeline_total += int(outcome.pipeline.get("total", 0))
        pipeline_accepts += int(outcome.pipeline.get("accepts", 0))
        pipeline_rejects += int(outcome.pipeline.get("rejects", 0))
        pipeline_false_accepts += int(outcome.pipeline.get("false_accepts", 0))
        pipeline_oracle_legal_accepts += int(outcome.pipeline.get("oracle_legal_accepts", 0))
        pipeline_errors += int(outcome.pipeline.get("errors", 0))
        pipeline_total_attempts += int(outcome.pipeline.get("total_attempts", 0))
        _append_jsonl(results_path, outcome.record)

    if max_workers > 1:
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, position in enumerate(positions, start=1):
                futures.append(executor.submit(_evaluate_position, idx, position))
            pending: dict[int, PositionOutcome] = {}
            next_to_write = 1
            completed = 0
            for future in as_completed(futures):
                outcome = future.result()
                pending[outcome.index] = outcome
                completed += 1
                while next_to_write in pending:
                    _consume_outcome(pending.pop(next_to_write))
                    next_to_write += 1
                if args.progress_every > 0 and completed % args.progress_every == 0:
                    logger.info("Progress: %s/%s positions", completed, args.iterations)
    else:
        for idx, position in enumerate(positions, start=1):
            _consume_outcome(_evaluate_position(idx, position))
            if args.progress_every > 0 and idx % args.progress_every == 0:
                logger.info("Progress: %s/%s positions", idx, args.iterations)

    finished_at = datetime.now(timezone.utc).isoformat()
    generator_summaries: dict[str, Any] = {}
    for system_id, stats in generator.items():
        total = stats["total"]
        generator_summaries[system_id] = {
            "system_id": system_id,
            "total": total,
            "errors": stats["errors"],
            "missing_output": stats["missing_output"],
            "invalid_move": stats["invalid_move"],
            "oracle_parseable": stats["oracle_parseable"],
            "oracle_legal": stats["oracle_legal"],
            "oracle_parse_rate": (stats["oracle_parseable"] / total) if total else 0.0,
            "oracle_legal_rate": (stats["oracle_legal"] / total) if total else 0.0,
        }

    validator_summaries: dict[str, Any] = {}
    for system_id, stats in validator_baseline.items():
        total = stats["total"]
        validator_summaries[system_id] = {
            "system_id": system_id,
            "total": total,
            "correct": stats["correct"],
            "errors": stats["errors"],
            "accuracy": (stats["correct"] / total) if total else 0.0,
        }

    summary = {
        "benchmark_id": benchmark_id,
        "task": "chess_random_suite",
        "positions": args.iterations,
        "plies": args.plies,
        "seed": args.seed,
        "illegal_ratio": args.illegal_ratio,
        "generators": generator_summaries,
        "validators": validator_summaries,
        "validator_on_generator": {
            "enabled": bool(validator_on_generator),
            "by_system": {
                system_id: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "errors": stats["errors"],
                    "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
                }
                for system_id, stats in validator_on_generator.items()
            },
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
    if len(generator_summaries) == 1:
        system_id, stats = next(iter(generator_summaries.items()))
        label = next((s.system_def.label for s in generator_systems if s.system_def.id == system_id), None)
        summary["generator"] = {
            "system_id": system_id,
            "system_label": label,
            "total": stats["total"],
            "errors": stats["errors"],
            "missing_output": stats["missing_output"],
            "invalid_move": stats["invalid_move"],
            "oracle_parseable": stats["oracle_parseable"],
            "oracle_legal": stats["oracle_legal"],
            "oracle_parse_rate": stats["oracle_parse_rate"],
            "oracle_legal_rate": stats["oracle_legal_rate"],
        }
    if len(validator_summaries) == 1:
        system_id, stats = next(iter(validator_summaries.items()))
        label = next((s.system_def.label for s in validator_systems if s.system_def.id == system_id), None)
        summary["validator_baseline"] = {
            "system_id": system_id,
            "system_label": label,
            "total": stats["total"],
            "correct": stats["correct"],
            "errors": stats["errors"],
            "accuracy": stats["accuracy"],
        }
    _write_json(summary_path, summary)

    logger.info("Done.")
    if summary["generators"]:
        logger.info(
            "Generators: %s",
            ", ".join(
                f"{sid} legal={stats['oracle_legal_rate']*100:.1f}% parse={stats['oracle_parse_rate']*100:.1f}%"
                for sid, stats in summary["generators"].items()
            ),
        )
    if validator_summaries:
        logger.info(
            "Validators: %s",
            ", ".join(
                f"{sid} acc={stats['accuracy']*100:.1f}%"
                for sid, stats in summary["validators"].items()
            ),
        )
    if validator_on_generator:
        for system_id, stats in summary["validator_on_generator"]["by_system"].items():
            logger.info("Validator-on-generator (%s) accuracy: %.1f%%", system_id, stats["accuracy"] * 100)
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
