#!/usr/bin/env python3
"""
extractor_full.py

Comprehensive PGN / game extractor for the IWantCheckmate project.

Features included (ALL THAT WE DISCUSSED):
 - Read PGNs from directory or ZIP (recursively)
 - Robust PGN cleaning (remove comments, variations, NAGs)
 - SAN -> UCI canonicalization and legality checks via python-chess
 - Per-position Stockfish analysis (optional): centipawn eval, mate, PV
 - Rich per-move motifs: capture, check, promotion, fork, pin, discovered attack, underpromotion
 - Game-level aggregated stats: opening guess, avg eval, blunder counts, tactical density
 - Piece activity and pawn-structure fingerprints
 - Phase detection (opening/mid/end) per position
 - Optionally upweight games from a target player
 - Save outputs as JSONL (one JSON per game) and an aggregated NPZ for move-frequency vectors
 - Helpful CLI with many switches

Notes:
 - Requires: python-chess, numpy. Stockfish is optional but recommended for analysis.
 - On Raspberry Pi, use low stockfish depth (6-10) to avoid long runtimes.

Usage examples:
  python3 extractor_full.py --pgn_dir pgns --out_dir data_out --stockfish /usr/bin/stockfish --stockfish_depth 10
  python3 extractor_full.py --pgn_zip mygames.zip --out_dir data_out --target_name "Magnus Carlsen" --weight 3.0

"""

from pathlib import Path
import zipfile
import io
import re
import json
import argparse
import sys
import time
from collections import Counter, defaultdict

import numpy as np

try:
    import chess
    import chess.pgn
    import chess.engine
except Exception as e:
    raise RuntimeError('python-chess required (pip install python-chess)') from e

# -------------------- PGN cleaning --------------------
CLEAN_RE = re.compile(r"\{[^}]*\}|\([^)]*\)|\$\d+", flags=re.DOTALL)

def clean_pgn_text(raw: str) -> str:
    """Strip comments, variations and numeric annotation glyphs (NAGs).
    We keep tags and move text but remove nested annotations that break parsing.
    """
    cleaned = CLEAN_RE.sub('', raw)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

# -------------------- Heuristics & utilities --------------------

def compute_game_phase(board: chess.Board) -> str:
    """Return 'opening'/'middlegame'/'endgame' based on simple material heuristic."""
    material = 0
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        material += len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
    if material >= 6:
        return 'opening'
    if material >= 3:
        return 'middlegame'
    return 'endgame'


def safe_move_uci(board: chess.Board, move) -> str:
    """Return UCI for a move, handling weird cases."""
    try:
        return move.uci()
    except Exception:
        try:
            m = board.parse_san(board.san(move))
            return m.uci()
        except Exception:
            return None

# -------------------- Motif detection (lightweight) --------------------

def detect_motifs(board_before: chess.Board, move: chess.Move) -> list:
    """Detect simple tactical motifs for a move.
    Motifs returned include: capture, check, promotion, fork, pin, discovered_attack, underpromotion
    These are heuristics and not exhaustive.
    """
    motifs = []
    try:
        if board_before.is_capture(move):
            motifs.append('capture')
    except Exception:
        pass
    try:
        tmp = board_before.copy()
        tmp.push(move)
        if tmp.is_check():
            motifs.append('check')
    except Exception:
        pass
    if move.promotion:
        motifs.append('promotion')
        # underpromotion
        if move.promotion != chess.QUEEN:
            motifs.append('underpromotion')
    # fork heuristic: moved piece attacks two or more higher-value targets after move
    try:
        moved_piece = board_before.piece_at(move.from_square)
        if moved_piece is not None:
            tmp = board_before.copy(); tmp.push(move)
            attackers = list(tmp.attacks(move.to_square))
            # count enemy pieces attacked (exclude pawns maybe)
            enemy_attacked = 0
            for sq in attackers:
                p = tmp.piece_at(sq)
                if p is None: continue
                if p.color != board_before.turn:
                    # target piece type weight
                    if p.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                        enemy_attacked += 1
            if enemy_attacked >= 2:
                motifs.append('fork')
    except Exception:
        pass
    # pin heuristic: if move creates a pin (target piece now pinned to king)
    try:
        tmp = board_before.copy(); tmp.push(move)
        king_sq = tmp.king(not board_before.turn)
        if king_sq is not None:
            # find pieces attacked that are along same ray
            # naive check: if any attacked enemy piece lies on same rank/file/diag as king and is attacked
            for sq in chess.SQUARES:
                p = tmp.piece_at(sq)
                if p is None or p.color == board_before.turn: continue
                if tmp.is_attacked_by(board_before.turn, sq):
                    # check collinearity with king
                    r1, c1 = divmod(sq, 8)
                    r2, c2 = divmod(king_sq, 8)
                    if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                        motifs.append('pin'); break
    except Exception:
        pass
    # discovered attack heuristic: the moved piece uncovering an attack
    try:
        b_copy = board_before.copy()
        # compute attacks before
        attacks_before = sum(1 for sq in chess.SQUARES if b_copy.is_attacked_by(not b_copy.turn, sq))
        b_copy.push(move)
        attacks_after = sum(1 for sq in chess.SQUARES if b_copy.is_attacked_by(not b_copy.turn, sq))
        if attacks_after > attacks_before:
            motifs.append('discovered_attack')
    except Exception:
        pass
    return motifs

# -------------------- Opening classification (very light) --------------------

def classify_opening_from_moves(moves_list: list) -> str:
    prefix = ' '.join(moves_list[:8]).lower()
    if 'e4' in prefix and 'c5' in prefix:
        return 'Sicilian'
    if 'e4' in prefix and 'e5' in prefix:
        return 'Open-e4e5'
    if 'd4' in prefix and 'd5' in prefix:
        return 'Closed-d4d5'
    if 'c4' in prefix:
        return 'English'
    if 'nf3' in prefix:
        return 'Reti-or-King\'s-NF3'
    return 'Other'

# -------------------- Piece activity & pawn structure --------------------

def piece_activity_and_pawn_signature(game) -> dict:
    # returns counters of piece moves and pawn-file pushes
    board = game.board()
    piece_moves = Counter()
    pawn_files = Counter()
    for mv in game.mainline_moves():
        try:
            piece = board.piece_at(mv.from_square)
        except Exception:
            piece = None
        if piece is not None:
            piece_moves[piece.piece_type] += 1
            if piece.piece_type == chess.PAWN:
                pawn_files[chess.square_file(mv.from_square)] += 1
        try:
            board.push(mv)
        except Exception:
            break
    return {'piece_moves': dict(piece_moves), 'pawn_files': dict(pawn_files)}

# -------------------- Stockfish analysis helper --------------------

def init_engine(path: str, threads: int = 1):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        return engine
    except Exception as e:
        print('[engine] failed to start', e, file=sys.stderr)
        return None


def analyze_position(engine, board: chess.Board, depth: int = 10, movetime_ms: int = None):
    if engine is None:
        return None
    try:
        if movetime_ms:
            limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        else:
            limit = chess.engine.Limit(depth=depth)
        info = engine.analyse(board, limit)
        score = info.get('score')
        if score is None:
            eval_cp = None
        else:
            # convert to centipawns; handle mate scores
            try:
                eval_cp = score.white().score(mate_score=100000)
            except Exception:
                try:
                    mate = score.white().mate()
                    eval_cp = 100000 if mate and mate > 0 else -100000
                except Exception:
                    eval_cp = None
        pv = None
        if 'pv' in info and info['pv']:
            pv = [m.uci() for m in info['pv']]
        return {'eval': eval_cp, 'pv': pv}
    except Exception as e:
        print('[engine] analyze error', e, file=sys.stderr)
        return None

# -------------------- Main extractor --------------------

def extract_game_features(game, engine=None, engine_depth=10, upweight=False):
    headers = dict(game.headers)
    player_white = headers.get('White', '')
    player_black = headers.get('Black', '')
    date = headers.get('Date', '')
    event = headers.get('Event', '')
    moves_san = []
    moves_uci = []
    per_move = []
    board = game.board()
    total_tactics = 0
    evals = []
    prev_moves = []

    for move in game.mainline_moves():
        # some PGNs may include weird nodes; safe-guard
        try:
            if not board.is_legal(move):
                try:
                    board.push(move)
                except Exception:
                    break
                prev_moves.append(None)
                continue
        except Exception:
            try:
                board.push(move)
            except Exception:
                break
            prev_moves.append(None)
            continue

        uci = safe_move_uci(board, move)
        try:
            san = board.san(move)
        except Exception:
            san = None
        # motifs
        motifs = detect_motifs(board, move)
        if motifs:
            total_tactics += len(motifs)
        phase = compute_game_phase(board)
        # engine analysis (optional)
        engine_info = None
        if engine is not None:
            engine_info = analyze_position(engine, board, depth=engine_depth)
            if engine_info and engine_info.get('eval') is not None:
                evals.append(engine_info['eval'])
        per_move.append({'uci': uci, 'san': san, 'motifs': motifs, 'phase': phase, 'engine': engine_info})

        moves_san.append(san)
        moves_uci.append(uci)
        prev_moves.append(uci)
        try:
            board.push(move)
        except Exception:
            break

    opening = classify_opening_from_moves([m.lower() for m in moves_uci if m]) if moves_uci else 'Unknown'
    piece_sig = piece_activity_and_pawn_signature(game)
    avg_eval = float(np.mean(evals)) if len(evals) else None
    std_eval = float(np.std(evals)) if len(evals) else None
    blunder_count = None
    # lightweight blunder detection: large negative eval swing after human move (if engine available)
    if engine is not None and len(per_move) >= 2:
        swings = 0
        for i in range(len(per_move)-1):
            a = per_move[i].get('engine', {})
            b = per_move[i+1].get('engine', {})
            if a and b and a.get('eval') is not None and b.get('eval') is not None:
                # if eval from white-perspective drops by > 300 centipawns (or increases if black to move), count
                if (b['eval'] - a['eval']) < -300:
                    swings += 1
        blunder_count = swings

    features = {
        'headers': headers,
        'players': {'white': player_white, 'black': player_black},
        'event': event,
        'date': date,
        'opening_guess': opening,
        'moves_uci': moves_uci,
        'moves_san': moves_san,
        'per_move': per_move,
        'agg': {
            'num_moves': len(moves_uci),
            'total_tactical_motifs': total_tactics,
            'avg_eval': avg_eval,
            'std_eval': std_eval,
            'blunder_count': blunder_count,
            'piece_activity': piece_sig,
        }
    }
    return features

# -------------------- Batch processing --------------------

def process_pgn_dir(pgn_dir: str, out_dir: str, engine=None, engine_depth=10, target_name=None, weight=1.0, max_games=None):
    p = Path(pgn_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    game_count = 0
    move_counter = Counter()

    for fn in p.rglob('*.pgn'):
        try:
            raw = fn.read_text(encoding='utf-8', errors='replace')
            raw = clean_pgn_text(raw)
            ph = io.StringIO(raw)
            while True:
                game = chess.pgn.read_game(ph)
                if game is None: break
                headers = dict(game.headers)
                is_target = False
                if target_name:
                    if target_name.lower() in headers.get('White','').lower() or target_name.lower() in headers.get('Black','').lower():
                        is_target = True
                feat = extract_game_features(game, engine=engine, engine_depth=engine_depth, upweight=is_target)
                # write per-game JSONL
                out_file = out_path / f"game_{game_count:07d}.json"
                with open(out_file, 'w', encoding='utf-8') as fh:
                    fh.write(json.dumps(feat, ensure_ascii=False))
                # update move counter (uci tokens) with optional weighting
                weight_factor = weight if is_target else 1.0
                for m in feat['moves_uci']:
                    if m:
                        move_counter[m] += weight_factor
                game_count += 1
                if max_games and game_count >= max_games:
                    break
        except Exception as e:
            print('[extractor] skip', fn, e, file=sys.stderr)
        if max_games and game_count >= max_games:
            break

    # save aggregated move-frequency vector
    moves = np.array(list(move_counter.keys()), dtype=object)
    counts = np.array([move_counter[m] for m in moves], dtype=np.int32)
    np.savez_compressed(Path(out_dir) / 'move_freq.npz', moves=moves, counts=counts)
    print(f'[extractor] processed {game_count} games; saved move_freq.npz')


def process_pgn_zip(pgn_zip: str, out_dir: str, engine=None, engine_depth=10, target_name=None, weight=1.0, max_games=None):
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    game_count = 0
    move_counter = Counter()
    with zipfile.ZipFile(pgn_zip, 'r') as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.pgn'): continue
            try:
                raw = zf.read(name).decode('utf-8', errors='replace')
                raw = clean_pgn_text(raw)
                ph = io.StringIO(raw)
                while True:
                    game = chess.pgn.read_game(ph)
                    if game is None: break
                    headers = dict(game.headers)
                    is_target = False
                    if target_name:
                        if target_name.lower() in headers.get('White','').lower() or target_name.lower() in headers.get('Black','').lower():
                            is_target = True
                    feat = extract_game_features(game, engine=engine, engine_depth=engine_depth, upweight=is_target)
                    out_file = out_path / f"game_{game_count:07d}.json"
                    with open(out_file, 'w', encoding='utf-8') as fh:
                        fh.write(json.dumps(feat, ensure_ascii=False))
                    weight_factor = weight if is_target else 1.0
                    for m in feat['moves_uci']:
                        if m:
                            move_counter[m] += weight_factor
                    game_count += 1
                    if max_games and game_count >= max_games:
                        break
            except Exception as e:
                print('[extractor] skip zip entry', name, e, file=sys.stderr)
            if max_games and game_count >= max_games:
                break
    moves = np.array(list(move_counter.keys()), dtype=object)
    counts = np.array([move_counter[m] for m in moves], dtype=np.int32)
    np.savez_compressed(Path(out_dir) / 'move_freq.npz', moves=moves, counts=counts)
    print(f'[extractor] processed {game_count} games from zip; saved move_freq.npz')

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pgn_dir', type=str, default=None)
    p.add_argument('--pgn_zip', type=str, default=None)
    p.add_argument('--out_dir', type=str, default='extracted')
    p.add_argument('--stockfish', type=str, default=None)
    p.add_argument('--stockfish_depth', type=int, default=8)
    p.add_argument('--target_name', type=str, default=None)
    p.add_argument('--weight', type=float, default=1.0)
    p.add_argument('--max_games', type=int, default=None)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    engine = None
    if args.stockfish:
        engine = init_engine(args.stockfish)
        if engine is None:
            print('[extractor] failed to start stockfish; continuing without engine', file=sys.stderr)
    if args.pgn_dir:
        process_pgn_dir(args.pgn_dir, args.out_dir, engine=engine, engine_depth=args.stockfish_depth, target_name=args.target_name, weight=args.weight, max_games=args.max_games)
    if args.pgn_zip:
        process_pgn_zip(args.pgn_zip, args.out_dir, engine=engine, engine_depth=args.stockfish_depth, target_name=args.target_name, weight=args.weight, max_games=args.max_games)
    if engine:
        try: engine.quit()
        except Exception: pass
    print('[extractor] done')
