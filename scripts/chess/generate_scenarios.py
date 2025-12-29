from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import chess

from psf.chess_utils import generate_illegal_san, play_random_plies


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _setup_logging(*, level: str, log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("psf.chess.generate")
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


def _generate_legal_check_san(
    *,
    out_dir: Path,
    count: int,
    plies: int,
    illegal_ratio: float,
    seed: int,
    overwrite: bool,
    progress_every: int,
    logger: logging.Logger,
) -> None:
    rng = random.Random(seed)

    if not overwrite:
        existing = [p for p in out_dir.glob("*.json") if p.is_file() and p.name != "manifest.json"]
        if existing:
            raise SystemExit(f"Output dir is not empty; pass --overwrite to replace: {out_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "task": "legal_check_san",
        "count": count,
        "plies": plies,
        "illegal_ratio": illegal_ratio,
        "seed": seed,
        "started_at": started_at,
    }
    _write_json(out_dir / "manifest.json", manifest)

    logger.info(
        "Generating %s scenarios (task=%s, plies=%s, seed=%s, illegal_ratio=%.2f) -> %s",
        count,
        "legal_check_san",
        plies,
        seed,
        illegal_ratio,
        out_dir,
    )

    for idx in range(1, count + 1):
        generated = play_random_plies(plies=plies, rng=rng)
        board = chess.Board(generated.fen)
        make_illegal = rng.random() < illegal_ratio
        if make_illegal:
            candidate_move = generate_illegal_san(board, rng)
            expected_legality = "illegal"
            candidate_move_uci = None
        else:
            move = rng.choice(list(board.legal_moves))
            candidate_move = board.san(move)
            expected_legality = "legal"
            candidate_move_uci = move.uci()

        payload: dict[str, Any] = {
            "position_fen": generated.fen,
            "board_context": generated.board_context,
            "candidate_move": candidate_move,
            "move_notation": "san",
            "expected_legality": expected_legality,
            "meta": {
                "plies": plies,
                "seed": seed,
                "move_history_san": generated.move_history_san,
                "move_history_uci": generated.move_history_uci,
                "candidate_move_uci": candidate_move_uci,
            },
        }
        out_path = out_dir / f"{idx:04d}_{expected_legality}.json"
        _write_json(out_path, payload)
        logger.debug("Wrote %s (move=%s, expected=%s)", out_path.name, candidate_move, expected_legality)
        if progress_every > 0 and idx % progress_every == 0:
            logger.info("Progress: %s/%s", idx, count)

    logger.info("Done. Wrote %s scenarios to %s", count, out_dir)


def _generate_move_generation_san(
    *,
    out_dir: Path,
    count: int,
    plies: int,
    seed: int,
    overwrite: bool,
    progress_every: int,
    logger: logging.Logger,
) -> None:
    rng = random.Random(seed)

    if not overwrite:
        existing = [p for p in out_dir.glob("*.json") if p.is_file() and p.name != "manifest.json"]
        if existing:
            raise SystemExit(f"Output dir is not empty; pass --overwrite to replace: {out_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "task": "move_generation_san",
        "count": count,
        "plies": plies,
        "seed": seed,
        "started_at": started_at,
    }
    _write_json(out_dir / "manifest.json", manifest)

    logger.info(
        "Generating %s scenarios (task=%s, plies=%s, seed=%s) -> %s",
        count,
        "move_generation_san",
        plies,
        seed,
        out_dir,
    )

    for idx in range(1, count + 1):
        generated = play_random_plies(plies=plies, rng=rng)
        payload: dict[str, Any] = {
            "position_fen": generated.fen,
            "board_context": generated.board_context,
            "move_notation": "san",
            "illegal_moves": [],
            "meta": {
                "plies": plies,
                "seed": seed,
                "move_history_san": generated.move_history_san,
                "move_history_uci": generated.move_history_uci,
            },
        }
        out_path = out_dir / f"{idx:04d}.json"
        _write_json(out_path, payload)
        logger.debug("Wrote %s", out_path.name)
        if progress_every > 0 and idx % progress_every == 0:
            logger.info("Progress: %s/%s", idx, count)

    logger.info("Done. Wrote %s scenarios to %s", count, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chess benchmark scenarios (SAN-first).")
    parser.add_argument(
        "--task",
        required=True,
        choices=("legal_check_san", "move_generation_san"),
        help="Which scenario task to generate",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for JSON scenarios")
    parser.add_argument("--count", type=int, default=100, help="Number of scenarios to generate")
    parser.add_argument("--plies", type=int, default=8, help="Number of random plies to play before sampling")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (deterministic output)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite scenarios in an existing out-dir")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N scenarios (0 disables)",
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
        help='Optional log file path, or "auto" to write into --out-dir',
    )
    parser.add_argument(
        "--illegal-ratio",
        type=float,
        default=0.5,
        help="Fraction of illegal scenarios for legal_check_san (0-1)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.plies < 0:
        raise SystemExit("--plies must be >= 0")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if not (0.0 <= args.illegal_ratio <= 1.0):
        raise SystemExit("--illegal-ratio must be between 0 and 1")

    log_file: Path | None = None
    if args.log_file:
        if args.log_file.strip().lower() == "auto":
            log_file = out_dir / "generate.log"
        else:
            log_file = Path(args.log_file).expanduser().resolve()
    logger = _setup_logging(level=args.log_level, log_file=log_file)

    if args.task == "legal_check_san":
        _generate_legal_check_san(
            out_dir=out_dir,
            count=args.count,
            plies=args.plies,
            illegal_ratio=args.illegal_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            logger=logger,
        )
    else:
        _generate_move_generation_san(
            out_dir=out_dir,
            count=args.count,
            plies=args.plies,
            seed=args.seed,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            logger=logger,
        )


if __name__ == "__main__":
    main()
