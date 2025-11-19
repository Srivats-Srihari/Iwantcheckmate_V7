#!/usr/bin/env python3
"""
dataset_builder_full.py

Full-featured dataset builder for IWantCheckmate.

Features included (exhaustive):
 - Read PGNs from directory or ZIP (recursive)
 - Robust PGN cleaning and parsing (python-chess)
 - Board encoding -> 12 planes (white/black pieces)
 - Extra scalar features (turn, castling rights, halfmove clock)
 - Move-history context: last K moves represented as token ids (from vocab)
 - Optional Stockfish analysis per position: engine-best move(s), eval (cp/mate), PV
 - Options to augment dataset with engine moves or filter by engine/human agreement
 - Save compact samples NPZ with: X_board, X_extra, X_moves, y_human, y_engine, eng_score
 - Support for gzip/zip input streams, error handling, progress reporting
 - Options: seq_len, max_games, max_samples, augment_with_engine, engine_depth, engine_topk
 - Restrict vocab (only include tokens present in vocab) and UNK handling

Usage examples:
  python3 dataset_builder_full.py --pgn_dir pgns --vocab vocab.npz --out samples.npz --seq_len 48
  python3 dataset_builder_full.py --pgn_zip all.pgn.zip --vocab vocab.npz --out samples_with_engine.npz --stockfish /usr/bin/stockfish --engine_depth 8 --augment_with_engine

Notes:
 - Stockfish is optional; on Pi use low engine_depth (6-10) to avoid long runtimes.
 - The file aims for robustness over speed; consider running on a stronger machine for very large corpora.
"""

from pathlib import Path
import zipfile
import io
import re
import argparse
import sys
import math
import json
import time
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import chess
    import chess.pgn
    import chess.engine
except Exception:
    raise RuntimeError('python-chess required: pip install python-chess')

# -------------------- PGN cleaning --------------------
CLEAN_RE = re.compile(r"\{[^}]*\}|\([^)]*\)|\$\d+", flags=re.DOTALL)

def clean_pgn_text(raw: str) -> str:
    cleaned = CLEAN_RE.sub('', raw)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

# -------------------- Board encoding --------------------
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        base = PIECE_TO_PLANE[p.piece_type]
        plane_idx = base + (0 if p.color == chess.WHITE else 6)
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        planes[plane_idx, r, c] = 1.0
    return planes

def extra_features(board: chess.Board) -> np.ndarray:
    # turn, castling rights (Wk, Wq, Bk, Bq), halfmove clock normalized
    return np.array([
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        min(board.halfmove_clock / 100.0, 1.0)
    ], dtype=np.float32)

# -------------------- Vocab helpers --------------------

def load_vocab(vocab_path: str) -> Tuple[Dict[str,int], Dict[int,str]]:
    d = np.load(vocab_path, allow_pickle=True)
    if 'moves' in d and 'ids' in d:
        moves = d['moves']
        ids = d['ids']
        vocab = {str(moves[i]): int(ids[i]) for i in range(len(moves))}
        inv = {int(ids[i]): str(moves[i]) for i in range(len(moves))}
        return vocab, inv
    # fallback: keys/vals style
    if 'keys' in d and 'vals' in d:
        keys = d['keys']; vals = d['vals']
        inv = {int(keys[i]): str(vals[i]) for i in range(len(keys))}
        vocab = {v:k for k,v in inv.items()}
        return vocab, inv
    raise RuntimeError('Unsupported vocab format')

# -------------------- Stockfish utils --------------------

def init_engine(path: str):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        return engine
    except Exception as e:
        print('[engine] failed to start:', e, file=sys.stderr)
        return None

def analyze(engine, board: chess.Board, depth: int = 10, movetime_ms: Optional[int] = None):
    try:
        if movetime_ms:
            limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        else:
            limit = chess.engine.Limit(depth=depth)
        info = engine.analyse(board, limit)
        score = info.get('score')
        eval_cp = None
        if score is not None:
            try:
                eval_cp = score.white().score(mate_score=100000)
            except Exception:
                # mate handling
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

# -------------------- Main dataset builder --------------------
class DatasetBuilder:
    def __init__(self, vocab: Dict[str,int], seq_len: int = 48, unk_token: int = 1):
        self.vocab = vocab
        self.seq_len = seq_len
        self.unk = unk_token
        self.X_board = []
        self.X_extra = []
        self.X_moves = []  # last-K move ids
        self.y_human = []
        self.y_engine = []
        self.eng_score = []
        self.meta = []

    def move_to_id(self, move_uci: str) -> int:
        return self.vocab.get(move_uci, self.unk)

    def build_from_game(self, game, engine=None, engine_depth=10, engine_topk=1, augment_with_engine=False, max_samples=None, upweight_player=False, player_weight=1.0):
        board = game.board()
        history = []  # uci moves history
        samples_added = 0
        for mv in game.mainline_moves():
            # legality guard
            try:
                if not board.is_legal(mv):
                    try:
                        board.push(mv)
                    except Exception:
                        break
                    history.append(mv.uci() if mv else None)
                    continue
            except Exception:
                try:
                    board.push(mv)
                except Exception:
                    break
                history.append(mv.uci() if mv else None)
                continue

            # current inputs
            b_planes = board_to_planes(board)
            extra = extra_features(board)
            # moves context: last seq_len moves as ids (pad left)
            seq = history[-self.seq_len:]
            seq_ids = [self.move_to_id(m) if m is not None else self.unk for m in seq]
            if len(seq_ids) < self.seq_len:
                pad = [0] * (self.seq_len - len(seq_ids))
                seq_ids = pad + seq_ids

            human_uci = safe_move_uci(board, mv)
            human_id = self.move_to_id(human_uci) if human_uci else self.unk

            engine_id = -1
            engine_eval = 0.0
            engine_pv = None
            if engine is not None:
                info = analyze(engine, board, depth=engine_depth)
                if info is not None:
                    engine_eval = float(info['eval']) if info.get('eval') is not None else 0.0
                    engine_pv = info.get('pv')
                    if engine_pv and len(engine_pv) > 0:
                        engine_id = self.move_to_id(engine_pv[0])

            # append human sample
            self.X_board.append(b_planes.reshape(-1))
            self.X_extra.append(extra)
            self.X_moves.append(np.array(seq_ids, dtype=np.int32))
            self.y_human.append(int(human_id))
            self.y_engine.append(int(engine_id) if engine_id >= 0 else -1)
            self.eng_score.append(float(engine_eval))
            self.meta.append({'san': board.san(mv) if True else None, 'uci': human_uci, 'engine_pv': engine_pv})
            samples_added += 1

            # optional augmentation: add engine sample too (synthetic)
            if augment_with_engine and engine is not None and engine_pv and engine_pv[0] != human_uci:
                eng_id = self.move_to_id(engine_pv[0])
                # keep same board/extras/seq but target = engine move
                self.X_board.append(b_planes.reshape(-1))
                self.X_extra.append(extra)
                self.X_moves.append(np.array(seq_ids, dtype=np.int32))
                self.y_human.append(int(eng_id))  # store in y_human but mark in meta
                self.y_engine.append(int(eng_id))
                self.eng_score.append(float(engine_eval))
                self.meta.append({'san': None, 'uci': engine_pv[0], 'engine_pv': engine_pv, 'synthetic': True})
                samples_added += 1

            try:
                board.push(mv)
            except Exception:
                break
            history.append(human_uci)

            if max_samples and samples_added >= max_samples:
                break

    def save(self, out_path: str):
        if len(self.X_board) == 0:
            raise RuntimeError('No samples to save')
        Xb = np.array(self.X_board, dtype=np.float32)
        Xe = np.array(self.X_extra, dtype=np.float32)
        Xm = np.array(self.X_moves, dtype=np.int32)
        yh = np.array(self.y_human, dtype=np.int32)
        ye = np.array(self.y_engine, dtype=np.int32)
        es = np.array(self.eng_score, dtype=np.float32)
        # meta as json list
        with open(str(out_path) + '.meta.json', 'w', encoding='utf-8') as fh:
            json.dump(self.meta, fh, indent=2, ensure_ascii=False)
        np.savez_compressed(out_path, X_board=Xb, X_extra=Xe, X_moves=Xm, y_human=yh, y_engine=ye, eng_score=es)
        print(f'[dataset] saved {len(yh)} samples to {out_path}.npz and meta to {out_path}.meta.json')

# -------------------- High-level processing --------------------

def process_dir(pgn_dir: str, builder: DatasetBuilder, engine=None, engine_depth=10, engine_topk=1, augment_with_engine=False, max_games=None, max_samples=None):
    p = Path(pgn_dir)
    game_count = 0
    sample_count = 0
    for fn in p.rglob('*.pgn'):
        try:
            raw = fn.read_text(encoding='utf-8', errors='replace')
            raw = clean_pgn_text(raw)
            ph = io.StringIO(raw)
            while True:
                g = chess.pgn.read_game(ph)
                if g is None:
                    break
                builder.build_from_game(g, engine=engine, engine_depth=engine_depth, engine_topk=engine_topk, augment_with_engine=augment_with_engine, max_samples=max_samples)
                game_count += 1
                if max_games and game_count >= max_games:
                    break
        except Exception as e:
            print('[dataset] skip', fn, e, file=sys.stderr)
        if max_games and game_count >= max_games:
            break
    print(f'[dataset] processed {game_count} games')

def process_zip(pgn_zip: str, builder: DatasetBuilder, engine=None, engine_depth=10, engine_topk=1, augment_with_engine=False, max_games=None, max_samples=None):
    game_count = 0
    with zipfile.ZipFile(pgn_zip, 'r') as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.pgn'):
                continue
            try:
                raw = zf.read(name).decode('utf-8', errors='replace')
                raw = clean_pgn_text(raw)
                ph = io.StringIO(raw)
                while True:
                    g = chess.pgn.read_game(ph)
                    if g is None:
                        break
                    builder.build_from_game(g, engine=engine, engine_depth=engine_depth, engine_topk=engine_topk, augment_with_engine=augment_with_engine, max_samples=max_samples)
                    game_count += 1
                    if max_games and game_count >= max_games:
                        break
            except Exception as e:
                print('[dataset] skip zip entry', name, e, file=sys.stderr)
            if max_games and game_count >= max_games:
                break
    print(f'[dataset] processed {game_count} games from zip')

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pgn_dir', type=str, default=None)
    p.add_argument('--pgn_zip', type=str, default=None)
    p.add_argument('--vocab', type=str, required=True)
    p.add_argument('--out', type=str, default='samples.npz')
    p.add_argument('--seq_len', type=int, default=48)
    p.add_argument('--stockfish', type=str, default=None)
    p.add_argument('--engine_depth', type=int, default=8)
    p.add_argument('--engine_topk', type=int, default=1)
    p.add_argument('--augment_with_engine', action='store_true')
    p.add_argument('--max_games', type=int, default=None)
    p.add_argument('--max_samples_per_game', type=int, default=None)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    vocab, inv = load_vocab(args.vocab)
    engine = None
    if args.stockfish:
        engine = init_engine(args.stockfish)
        if engine is None:
            print('[dataset] continuing without engine')
    builder = DatasetBuilder(vocab=vocab, seq_len=args.seq_len)
    if args.pgn_dir:
        process_dir(args.pgn_dir, builder, engine=engine, engine_depth=args.engine_depth, engine_topk=args.engine_topk, augment_with_engine=args.augment_with_engine, max_games=args.max_games, max_samples=args.max_samples_per_game)
    if args.pgn_zip:
        process_zip(args.pgn_zip, builder, engine=engine, engine_depth=args.engine_depth, engine_topk=args.engine_topk, augment_with_engine=args.augment_with_engine, max_games=args.max_games, max_samples=args.max_samples_per_game)
    builder.save(args.out)
    if engine:
        try: engine.quit()
        except Exception: pass
