from __future__ import annotations

from dataclasses import dataclass
import random

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

