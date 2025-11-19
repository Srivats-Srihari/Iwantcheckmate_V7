#!/usr/bin/env python3
"""
vocab_builder_max.py

Maximally expanded vocabulary builder for IWantCheckmate project.
Features included:
 - Read PGNs from directory or ZIP
 - Clean PGN text (remove comments/variations/NAGs)
 - SAN -> UCI canonicalization and legality filtering via python-chess
 - Optional Stockfish per-position evaluation (centipawn/mate) tagging
 - Per-move metadata: frequency, phases (opening/middlegame/endgame), motifs (capture/check/promotion), avg eval, players seen
 - Weighting: ability to upweight moves from a target player (by name in PGN headers)
 - N-gram support: unigram (move), bigram (prev_move+move), trigram tokens
 - Option to restrict vocab to moves used by target player only
 - Options to set minimum frequency, max vocab size, and output files
 - Outputs: vocab.npz (moves, ids), vocab_meta.json (rich metadata), optionally vocab_ngrams.npz

Usage examples:
  python3 vocab_builder_max.py --pgn_dir pgns --out vocab.npz
  python3 vocab_builder_max.py --pgn_zip all.pgn.zip --stockfish /usr/bin/stockfish --stockfish_depth 10 --target_name "John Doe" --weight 5.0 --min_freq 2 --max_vocab 20000

Note: Stockfish is optional but recommended for richer metadata. On Pi, prefer low depth.
"""

from pathlib import Path
import zipfile
import io
import re
import json
import argparse
import sys
import math
from collections import Counter, defaultdict

import numpy as np

try:
    import chess
    import chess.pgn
    import chess.engine
except Exception as e:
    raise RuntimeError('python-chess required: pip install python-chess') from e

# ------------------ Helpers ------------------
CLEAN_RE = re.compile(r"\{[^}]*\}|\([^)]*\)|\$\d+", flags=re.DOTALL)
SAN_NO_ANNOTS_RE = re.compile(r"[?!]+$")

def clean_pgn_text(raw: str) -> str:
    """Strip comments, variations and NAGs that break parsing."""
    cleaned = CLEAN_RE.sub('', raw)
    # collapse multiple whitespace
    cleaned = re.sub(r"\s+", ' ', cleaned)
    return cleaned

# Game phase heuristic
def compute_game_phase(board: chess.Board) -> str:
    # material heuristic: count non-pawn material (rough)
    material = 0
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        material += len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
    # thresholds tuned empirically
    if material >= 6:
        return 'opening'
    if material >= 3:
        return 'middlegame'
    return 'endgame'

# Detect light tactical motifs
def detect_motifs(board_before: chess.Board, move: chess.Move) -> list:
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
    # simple fork-ish heuristic: after move, attacked piece types count increases
    return motifs

# N-gram helper
def ngram_token(prev_moves: list, move_uci: str, n: int) -> str:
    if n == 1:
        return move_uci
    if n == 2:
        prev = prev_moves[-1] if len(prev_moves) >= 1 else '<START>'
        return prev + ' ' + move_uci
    if n == 3:
        p1 = prev_moves[-2] if len(prev_moves) >= 2 else '<START>'
        p2 = prev_moves[-1] if len(prev_moves) >= 1 else '<START>'
        return p1 + ' ' + p2 + ' ' + move_uci
    return move_uci

# Safe SAN -> UCI canonicalization
def safe_move_uci(board: chess.Board, move) -> str:
    try:
        return move.uci()
    except Exception:
        # fallback: try to rebuild move from SAN
        try:
            san = board.san(move)
            m = board.parse_san(san)
            return m.uci()
        except Exception:
            return None

# ------------------ Main builder ------------------
class VocabBuilder:
    def __init__(self, stockfish_path=None, stockfish_depth=12, weight_target=1.0, target_name=None, ngram=(1,)):
        self.move_counter = Counter()
        self.meta = defaultdict(lambda: {'count':0, 'phase':Counter(), 'motifs':Counter(), 'players':Counter(), 'avg_eval_sum':0.0, 'eval_count':0, 'ngrams':Counter()})
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.target_name = target_name
        self.weight_target = float(weight_target) if weight_target is not None else 1.0
        self.ngram = tuple(sorted(set(ngram)))
        self.engine = None
        if stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            except Exception as e:
                print('[vocab] Warning: Failed to start Stockfish:', e, file=sys.stderr)
                self.engine = None

    def close(self):
        if self.engine:
            try: self.engine.quit()
            except Exception: pass

    def process_game_fh(self, fh, upweight_target=False):
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            headers = {k: v for k,v in game.headers.items()}
            player_white = headers.get('White','')
            player_black = headers.get('Black','')
            is_target_game = False
            if self.target_name:
                if self.target_name.lower() in player_white.lower() or self.target_name.lower() in player_black.lower():
                    is_target_game = True
            board = game.board()
            prev_moves = []
            # per-game engine analysis optional: we can compute per-position eval
            for node in game.mainline():
                move = node.move
                if move is None:
                    # skip weird nodes
                    continue
                # ensure legality
                try:
                    if not board.is_legal(move):
                        # try to push anyway to keep sync; skip counting
                        try:
                            board.push(move)
                        except Exception:
                            break
                        continue
                except Exception:
                    try:
                        board.push(move)
                    except Exception:
                        break
                    continue

                uci = safe_move_uci(board, move)
                if uci is None:
                    try:
                        board.push(move)
                    except Exception:
                        break
                    continue

                # compute weighting
                w = self.weight_target if is_target_game else 1.0

                # detect phase & motifs
                phase = compute_game_phase(board)
                motifs = detect_motifs(board, move)

                # optional stockfish eval
                eval_cp = None
                if self.engine is not None:
                    try:
                        info = self.engine.analyse(board, chess.engine.Limit(depth=self.stockfish_depth))
                        score = info.get('score')
                        if score is not None:
                            try:
                                eval_cp = score.white().score(mate_score=100000)
                            except Exception:
                                # mate scores -> encode as large cp
                                try:
                                    eval_cp = 100000 if score.white().mate() > 0 else -100000
                                except Exception:
                                    eval_cp = None
                    except Exception:
                        eval_cp = None

                # add basic unigrams and n-grams
                for n in self.ngram:
                    tok = ngram_token(prev_moves, uci, n)
                    self.move_counter[tok] += w
                    meta = self.meta[tok]
                    meta['count'] += w
                    meta['phase'][phase] += 1
                    for m in motifs:
                        meta['motifs'][m] += 1
                    if player_white: meta['players'][player_white] += 1
                    if player_black: meta['players'][player_black] += 1
                    if eval_cp is not None:
                        meta['avg_eval_sum'] += float(eval_cp)
                        meta['eval_count'] += 1
                    meta['ngrams'][len(prev_moves)] += 1

                # record previous move tokens (for bigrams/trigrams prior context)
                prev_moves.append(uci)
                try:
                    board.push(move)
                except Exception:
                    break

    def process_pgn_dir(self, pgn_dir, max_games=None):
        p = Path(pgn_dir)
        gcount = 0
        for fn in p.rglob('*.pgn'):
            try:
                raw = fn.read_text(encoding='utf-8', errors='replace')
                raw = clean_pgn_text(raw)
                fh = io.StringIO(raw)
                self.process_game_fh(fh)
                gcount += 1
                if max_games and gcount >= max_games:
                    break
            except Exception as e:
                print('[vocab] skip', fn, e, file=sys.stderr)

    def process_pgn_zip(self, pgn_zip, max_games=None):
        with zipfile.ZipFile(pgn_zip, 'r') as zf:
            gcount = 0
            for name in zf.namelist():
                if not name.lower().endswith('.pgn'):
                    continue
                try:
                    raw = zf.read(name).decode('utf-8', errors='replace')
                    raw = clean_pgn_text(raw)
                    fh = io.StringIO(raw)
                    self.process_game_fh(fh)
                    gcount += 1
                    if max_games and gcount >= max_games:
                        break
                except Exception as e:
                    print('[vocab] skip zip entry', name, e, file=sys.stderr)

    def finalize_and_save(self, out_vocab='vocab.npz', out_meta='vocab_meta.json', min_freq=1, max_vocab=None, restrict_to_target=False):
        # apply min_freq and optional restriction
        items = [(m,c) for m,c in self.move_counter.items() if c >= min_freq]
        # if restricting to target moves only, filter by player presence in meta
        if restrict_to_target and self.target_name:
            items = [ (m,c) for m,c in items if self.target_name in self.meta[m]['players'] ]

        # sort by count desc
        items.sort(key=lambda x: x[1], reverse=True)
        if max_vocab:
            items = items[:max_vocab]

        moves = [m for m,_ in items]
        ids = list(range(len(moves)))
        # save npz
        np.savez_compressed(out_vocab, moves=np.array(moves, dtype=object), ids=np.array(ids, dtype=np.int32))

        # prepare meta for JSON (convert Counters)
        out_meta_dict = {}
        for m,_ in items:
            mm = self.meta[m]
            entry = {
                'count': int(mm['count']),
                'phase': {k:v for k,v in mm['phase'].items()},
                'motifs': {k:v for k,v in mm['motifs'].items()},
                'players': {k:v for k,v in mm['players'].items()},
                'avg_eval': float(mm['avg_eval_sum']/mm['eval_count']) if mm['eval_count']>0 else None,
                'eval_count': int(mm['eval_count'])
            }
            out_meta_dict[m] = entry
        with open(out_meta, 'w', encoding='utf-8') as fh:
            json.dump(out_meta_dict, fh, indent=2)
        print(f'[vocab] saved {len(moves)} tokens to {out_vocab} and metadata to {out_meta}')

# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pgn_dir', type=str, default=None)
    p.add_argument('--pgn_zip', type=str, default=None)
    p.add_argument('--stockfish', type=str, default=None)
    p.add_argument('--stockfish_depth', type=int, default=8)
    p.add_argument('--target_name', type=str, default=None)
    p.add_argument('--weight', type=float, default=1.0, help='upweight factor for moves from target player')
    p.add_argument('--ngram', type=int, nargs='+', default=[1], help='which ngrams to include, e.g. --ngram 1 2')
    p.add_argument('--min_freq', type=int, default=1)
    p.add_argument('--max_vocab', type=int, default=None)
    p.add_argument('--restrict_to_target', action='store_true')
    p.add_argument('--out_vocab', type=str, default='vocab.npz')
    p.add_argument('--out_meta', type=str, default='vocab_meta.json')
    p.add_argument('--max_games', type=int, default=None)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    builder = VocabBuilder(stockfish_path=args.stockfish, stockfish_depth=args.stockfish_depth, weight_target=args.weight, target_name=args.target_name, ngram=tuple(args.ngram))
    if args.pgn_zip:
        builder.process_pgn_zip(args.pgn_zip, max_games=args.max_games)
    if args.pgn_dir:
        builder.process_pgn_dir(args.pgn_dir, max_games=args.max_games)
    builder.finalize_and_save(out_vocab=args.out_vocab, out_meta=args.out_meta, min_freq=args.min_freq, max_vocab=args.max_vocab, restrict_to_target=args.restrict_to_target)
    builder.close()
