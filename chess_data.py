from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

import chess


@dataclass(frozen=True)
class GeneratedPosition:
    fen: str
    board_context: str
    move_history_san: list[str]
    move_history_uci: list[str]


def render_board_context(board: chess.Board) -> str:
    fen = board.fen()
    parts = fen.split()
    board_fen = parts[0] if len(parts) > 0 else ""
    turn = parts[1] if len(parts) > 1 else ""
    castling = parts[2] if len(parts) > 2 else ""
    ep = parts[3] if len(parts) > 3 else ""
    halfmove = parts[4] if len(parts) > 4 else ""
    fullmove = parts[5] if len(parts) > 5 else ""
    return "\n".join(
        [
            "CHESS_STATE_V1",
            f"FEN: {fen}",
            f"TURN: {'white' if turn == 'w' else 'black'}",
            f"CASTLING: {castling or '-'}",
            f"EP: {ep or '-'}",
            f"HALFMOVE: {halfmove or '0'}",
            f"FULLMOVE: {fullmove or '1'}",
            "BOARD:",
            str(board),
            "",
            f"BOARD_FEN: {board_fen}",
        ]
    )


def play_random_plies(*, plies: int, rng: random.Random, max_resets: int = 10_000) -> GeneratedPosition:
    for _ in range(max_resets):
        board = chess.Board()
        history_san: list[str] = []
        history_uci: list[str] = []
        for _ply in range(plies):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = rng.choice(legal_moves)
            history_san.append(board.san(move))
            history_uci.append(move.uci())
            board.push(move)
        if len(history_uci) != plies:
            continue
        if board.is_game_over(claim_draw=True):
            continue
        if not any(True for _ in board.legal_moves):
            continue
        fen = board.fen()
        return GeneratedPosition(
            fen=fen,
            board_context=render_board_context(board),
            move_history_san=history_san,
            move_history_uci=history_uci,
        )
    raise RuntimeError("Failed to generate a non-terminal position; try fewer plies or a different seed.")


def generate_illegal_san(board: chess.Board, rng: random.Random) -> str:
    squares = [chess.square_name(sq) for sq in chess.SQUARES]
    pieces = ["", "N", "B", "R", "Q", "K"]
    templates = ("castle_short", "castle_long", "pawn_push", "piece_move", "piece_capture")
    for _ in range(2_000):
        template = rng.choice(templates)
        if template == "castle_short":
            candidate = "O-O"
        elif template == "castle_long":
            candidate = "O-O-O"
        elif template == "pawn_push":
            candidate = rng.choice(squares)
        elif template == "piece_capture":
            candidate = f"{rng.choice(pieces[1:])}x{rng.choice(squares)}"
        else:
            candidate = f"{rng.choice(pieces[1:])}{rng.choice(squares)}"
        try:
            board.parse_san(candidate)
        except (chess.IllegalMoveError, chess.InvalidMoveError):
            return candidate
    return "Qz9"


def generate_illegal_uci(board: chess.Board, rng: random.Random) -> str:
    squares = [chess.square_name(sq) for sq in chess.SQUARES]
    promotion_pieces = ["", "q", "r", "b", "n"]
    for _ in range(5_000):
        from_sq = rng.choice(squares)
        to_sq = rng.choice(squares)
        if from_sq == to_sq:
            continue
        promotion = rng.choice(promotion_pieces)
        candidate = f"{from_sq}{to_sq}{promotion}"
        try:
            move = chess.Move.from_uci(candidate)
        except ValueError:
            continue
        if board.is_legal(move):
            continue
        return candidate
    return "a1a1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _setup_logging(*, level: str, log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger("psf.chess.data")
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


def generate_legal_check_scenarios(
    *,
    out_dir: Path,
    count: int,
    plies: int,
    illegal_ratio: float,
    seed: int,
    notation: str,
    overwrite: bool,
    progress_every: int,
    logger: logging.Logger,
) -> None:
    if notation not in {"san", "uci"}:
        raise ValueError("--notation must be 'san' or 'uci'")

    rng = random.Random(seed)

    if not overwrite:
        existing = [p for p in out_dir.glob("*.json") if p.is_file() and p.name != "manifest.json"]
        if existing:
            raise SystemExit(f"Output dir is not empty; pass --overwrite to replace: {out_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "scenario_format": "psf_benchmark_scenario_v1",
        "task": f"legal_check_{notation}",
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
        manifest["task"],
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
            if notation == "uci":
                candidate_move = generate_illegal_uci(board, rng)
            else:
                candidate_move = generate_illegal_san(board, rng)
            expected_legality = "illegal"
            candidate_move_uci: str | None = None
            candidate_move_san: str | None = None
        else:
            move = rng.choice(list(board.legal_moves))
            candidate_move_uci = move.uci()
            candidate_move_san = board.san(move)
            candidate_move = candidate_move_uci if notation == "uci" else candidate_move_san
            expected_legality = "legal"

        expected_is_legal = expected_legality == "legal"
        expected_answer = "true" if expected_is_legal else "false"
        payload: dict[str, Any] = {
            "input": {
                "position_fen": generated.fen,
                "board_context": generated.board_context,
                "candidate_move": candidate_move,
                "move_notation": notation,
            },
            "expected": expected_answer,
            "meta": {
                "plies": plies,
                "seed": seed,
                "expected_legality": expected_legality,
                "expected_is_legal": expected_is_legal,
                "move_history_san": generated.move_history_san,
                "move_history_uci": generated.move_history_uci,
                "candidate_move_san": candidate_move_san,
                "candidate_move_uci": candidate_move_uci,
            },
        }
        out_path = out_dir / f"{idx:04d}_{expected_legality}.json"
        _write_json(out_path, payload)
        logger.debug("Wrote %s (move=%s, expected=%s)", out_path.name, candidate_move, expected_legality)
        if progress_every > 0 and idx % progress_every == 0:
            logger.info("Progress: %s/%s", idx, count)

    logger.info("Done. Wrote %s scenarios to %s", count, out_dir)


def _to_notation(board: chess.Board, move: chess.Move, *, notation: str) -> str:
    if notation == "uci":
        return move.uci()
    return board.san(move)


def generate_move_generation_scenarios(
    *,
    out_dir: Path,
    count: int,
    plies: int,
    seed: int,
    notation: str,
    forbidden_moves_per_scenario: int,
    overwrite: bool,
    progress_every: int,
    logger: logging.Logger,
) -> None:
    if notation not in {"san", "uci"}:
        raise ValueError("--notation must be 'san' or 'uci'")

    rng = random.Random(seed)

    if not overwrite:
        existing = [p for p in out_dir.glob("*.json") if p.is_file() and p.name != "manifest.json"]
        if existing:
            raise SystemExit(f"Output dir is not empty; pass --overwrite to replace: {out_dir}")

    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "scenario_format": "psf_benchmark_scenario_v1",
        "task": f"move_generation_{notation}",
        "count": count,
        "plies": plies,
        "seed": seed,
        "forbidden_moves_per_scenario": forbidden_moves_per_scenario,
        "started_at": started_at,
    }
    _write_json(out_dir / "manifest.json", manifest)

    logger.info(
        "Generating %s scenarios (task=%s, plies=%s, seed=%s, forbidden=%s) -> %s",
        count,
        manifest["task"],
        plies,
        seed,
        forbidden_moves_per_scenario,
        out_dir,
    )

    for idx in range(1, count + 1):
        generated = play_random_plies(plies=plies, rng=rng)
        board = chess.Board(generated.fen)
        all_legal = [_to_notation(board, move, notation=notation) for move in board.legal_moves]
        all_legal = sorted(set(all_legal))

        illegal_moves: list[str] = []
        if forbidden_moves_per_scenario > 0:
            if forbidden_moves_per_scenario >= len(all_legal):
                raise SystemExit(
                    f"--forbidden-moves-per-scenario ({forbidden_moves_per_scenario}) must be < legal move count ({len(all_legal)})"
                )
            illegal_moves = rng.sample(all_legal, k=forbidden_moves_per_scenario)

        expected_legal_moves = [m for m in all_legal if m not in set(illegal_moves)]
        if not expected_legal_moves:
            raise SystemExit("Internal error: no remaining legal moves after forbidding moves")

        payload: dict[str, Any] = {
            "input": {
                "position_fen": generated.fen,
                "board_context": generated.board_context,
                "move_notation": notation,
                "illegal_moves": illegal_moves,
                "attempt": 1,
            },
            "expected": expected_legal_moves,
            "meta": {
                "plies": plies,
                "seed": seed,
                "forbidden_moves_per_scenario": forbidden_moves_per_scenario,
                "move_history_san": generated.move_history_san,
                "move_history_uci": generated.move_history_uci,
                "legal_move_count": len(all_legal),
            },
        }
        out_path = out_dir / f"{idx:04d}.json"
        _write_json(out_path, payload)
        logger.debug("Wrote %s (legal_moves=%s)", out_path.name, len(expected_legal_moves))
        if progress_every > 0 and idx % progress_every == 0:
            logger.info("Progress: %s/%s", idx, count)

    logger.info("Done. Wrote %s scenarios to %s", count, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chess dataset generator for PSF benchmarks (SAN + UCI).")
    parser.add_argument(
        "--task",
        required=True,
        choices=("legal_check_san", "legal_check_uci", "move_generation_san", "move_generation_uci"),
        help="Which dataset to generate",
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
        help="Fraction of illegal scenarios for legal_check_* tasks (0-1)",
    )
    parser.add_argument(
        "--forbidden-moves-per-scenario",
        type=int,
        default=0,
        help="How many legal moves to mark as illegal_moves (move_generation_* tasks only)",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.plies < 0:
        raise SystemExit("--plies must be >= 0")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")
    if args.forbidden_moves_per_scenario < 0:
        raise SystemExit("--forbidden-moves-per-scenario must be >= 0")
    if not (0.0 <= args.illegal_ratio <= 1.0):
        raise SystemExit("--illegal-ratio must be between 0 and 1")

    log_file: Path | None = None
    if args.log_file:
        if args.log_file.strip().lower() == "auto":
            log_file = out_dir / "generate.log"
        else:
            log_file = Path(args.log_file).expanduser().resolve()
    logger = _setup_logging(level=args.log_level, log_file=log_file)

    if args.task.startswith("legal_check_"):
        notation = args.task.split("_")[-1]
        generate_legal_check_scenarios(
            out_dir=out_dir,
            count=args.count,
            plies=args.plies,
            illegal_ratio=args.illegal_ratio,
            seed=args.seed,
            notation=notation,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            logger=logger,
        )
        return

    notation = args.task.split("_")[-1]
    generate_move_generation_scenarios(
        out_dir=out_dir,
        count=args.count,
        plies=args.plies,
        seed=args.seed,
        notation=notation,
        forbidden_moves_per_scenario=args.forbidden_moves_per_scenario,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        logger=logger,
    )


if __name__ == "__main__":
    main()
