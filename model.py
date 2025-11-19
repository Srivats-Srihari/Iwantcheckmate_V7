#!/usr/bin/env python3
"""
model.py

Full-featured PyTorch model for IWantCheckmate.

Capabilities included:
 - Board encoder (dense projection over 12x8x8 planes)
 - Move-history encoder (Embedding + GRU)
 - Style-vector conditioning (dense projection)
 - Fusion of board / move-history / style
 - Dual-headed outputs (human imitation head, optional engine-guidance head)
 - Sampling helpers (temperature, top-k, deterministic argmax)
 - Save / load utilities (checkpoint metadata stored)
 - CPU- and GPU-friendly device handling
 - Small utilities for converting model logits -> move choices given legal moves

Model is intentionally modular so you can swap in a ConvNet or Transformer later.

"""

from typing import Dict, List, Optional, Tuple
import math
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("iwant_model")

# -------------------- Hyper defaults --------------------
DEFAULT_SEQ_LEN = 48
DEFAULT_EMB = 128
DEFAULT_STYLE_DIM = 64
DEFAULT_HIDDEN = 512

# -------------------- Model --------------------
class IWantModel(nn.Module):
    """Modular imitation model.

    Inputs:
      - board_x: FloatTensor (B, board_dim) where board_dim == 12*8*8
      - moves_x: LongTensor (B, seq_len) token ids for recent moves
      - extra_x: FloatTensor (B, extra_dim) optional scalar features
      - style_x: FloatTensor (B, style_dim)

    Outputs:
      - logits_human: (B, vocab_size)
      - logits_engine: (B, vocab_size) (optional; may be same as human head if not used)
    """
    def __init__(self,
                 vocab_size: int,
                 board_dim: int = 12*8*8,
                 extra_dim: int = 6,
                 seq_len: int = DEFAULT_SEQ_LEN,
                 emb_dim: int = DEFAULT_EMB,
                 style_dim: int = DEFAULT_STYLE_DIM,
                 hidden: int = DEFAULT_HIDDEN,
                 n_move_tokens: int = 20000,
                 tie_heads: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.board_dim = int(board_dim)
        self.extra_dim = int(extra_dim)
        self.seq_len = int(seq_len)
        self.emb_dim = int(emb_dim)
        self.style_dim = int(style_dim)
        self.hidden = int(hidden)
        self.n_move_tokens = int(n_move_tokens)
        self.tie_heads = bool(tie_heads)
        self.device = device or torch.device('cpu')

        # Board encoder (simple MLP over flattened planes; easy on Pi)
        self.board_fc = nn.Sequential(
            nn.Linear(self.board_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
        )

        # extra features projection
        self.extra_fc = nn.Sequential(
            nn.Linear(self.extra_dim, max(8, emb_dim // 8)),
            nn.GELU(),
        )

        # Move history embedding + GRU
        self.move_embed = nn.Embedding(self.n_move_tokens, emb_dim // 2, padding_idx=0)
        self.move_gru = nn.GRU(input_size=emb_dim // 2, hidden_size=emb_dim // 2, num_layers=1, batch_first=True)

        # style projection
        self.style_proj = nn.Linear(self.style_dim, emb_dim)

        # fusion & MLP trunk
        fused_dim = emb_dim + (emb_dim // 2) + (emb_dim // 8) + emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )

        # heads
        self.human_head = nn.Linear(hidden // 2, self.vocab_size)
        if not self.tie_heads:
            self.engine_head = nn.Linear(hidden // 2, self.vocab_size)
        else:
            self.engine_head = None

        # small initializer
        self._init_weights()

        # move to device
        self.to(self.device)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, board_x: torch.FloatTensor, moves_x: torch.LongTensor, extra_x: torch.FloatTensor, style_x: torch.FloatTensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        board_x: (B, board_dim)
        moves_x: (B, seq_len)
        extra_x: (B, extra_dim)
        style_x: (B, style_dim)

        returns: (logits_human, logits_engine_or_None)
        """
        b = self.board_fc(board_x)
        e = self.extra_fc(extra_x)

        # moves
        mv_emb = self.move_embed(moves_x)  # (B, L, Dm)
        _, h = self.move_gru(mv_emb)       # h: (1, B, Dm)
        h = h.squeeze(0)                  # (B, Dm)

        s = self.style_proj(style_x)      # (B, emb_dim)

        comb = torch.cat([b, h, e, s], dim=1)
        trunk = self.trunk(comb)
        logits_h = self.human_head(trunk)
        if self.engine_head is not None:
            logits_e = self.engine_head(trunk)
        else:
            logits_e = logits_h
        return logits_h, logits_e

    # ------------------ helpers ------------------
    @torch.no_grad()
    def predict_logits_for_candidates(self, board_x: torch.FloatTensor, moves_x: torch.LongTensor, extra_x: torch.FloatTensor, style_x: torch.FloatTensor, candidate_token_ids: torch.LongTensor) -> torch.Tensor:
        """Compute logits for a list of candidate token ids.

        candidate_token_ids: LongTensor shape (N_candidates,) or (B, N_candidates)
        Returns: probs shape (B, N_candidates)
        """
        logits_h, _ = self.forward(board_x, moves_x, extra_x, style_x)
        # logits_h: (B, vocab_size)
        if candidate_token_ids.dim() == 1:
            idx = candidate_token_ids.unsqueeze(0).expand(logits_h.shape[0], -1)
        else:
            idx = candidate_token_ids
        gathered = logits_h.gather(1, idx.to(logits_h.device))
        probs = F.softmax(gathered, dim=1)
        return probs

    @torch.no_grad()
    def sample_move_from_candidates(self, board_x: torch.FloatTensor, moves_x: torch.LongTensor, extra_x: torch.FloatTensor, style_x: torch.FloatTensor, candidate_token_ids: torch.LongTensor, temperature: float = 1.0, top_k: int = 0) -> int:
        """Sample a single candidate index for batch-size 1. Returns the chosen candidate index (int).
        Use temperature and top_k to control exploration.
        """
        probs = self.predict_logits_for_candidates(board_x, moves_x, extra_x, style_x, candidate_token_ids).squeeze(0).cpu().numpy()
        if top_k > 0 and top_k < len(probs):
            top_idx = np.argpartition(-probs, top_k-1)[:top_k]
            top_probs = probs[top_idx]
            top_probs = np.maximum(top_probs, 1e-12)
            top_probs = top_probs / top_probs.sum()
            choice_idx = np.random.choice(top_idx, p=top_probs)
            return int(choice_idx)
        # apply temperature
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / max(1e-9, temperature))
            probs = probs / probs.sum()
        probs = np.nan_to_num(probs)
        if probs.sum() <= 0:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    @torch.no_grad()
    def best_move_from_candidates(self, board_x: torch.FloatTensor, moves_x: torch.LongTensor, extra_x: torch.FloatTensor, style_x: torch.FloatTensor, candidate_token_ids: torch.LongTensor) -> int:
        logits_h, _ = self.forward(board_x, moves_x, extra_x, style_x)
        if candidate_token_ids.dim() == 1:
            idx = candidate_token_ids.unsqueeze(0).expand(logits_h.shape[0], -1)
        else:
            idx = candidate_token_ids
        gathered = logits_h.gather(1, idx.to(logits_h.device))
        best = torch.argmax(gathered, dim=1).cpu().numpy()
        return int(best[0])

    # ------------------ save / load ------------------
    def save(self, path: str):
        ckpt = {
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'board_dim': self.board_dim,
            'extra_dim': self.extra_dim,
            'seq_len': self.seq_len,
            'emb_dim': self.emb_dim,
            'style_dim': self.style_dim,
            'hidden': self.hidden,
            'n_move_tokens': self.n_move_tokens,
            'tie_heads': self.tie_heads
        }
        torch.save(ckpt, path)
        LOG.info('Saved model checkpoint to %s', path)

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None) -> 'IWantModel':
        device = device or torch.device('cpu')
        ckpt = torch.load(path, map_location=device)
        m = IWantModel(vocab_size=int(ckpt['vocab_size']), board_dim=int(ckpt['board_dim']), extra_dim=int(ckpt['extra_dim']), seq_len=int(ckpt.get('seq_len', DEFAULT_SEQ_LEN)), emb_dim=int(ckpt.get('emb_dim', DEFAULT_EMB)), style_dim=int(ckpt.get('style_dim', DEFAULT_STYLE_DIM)), hidden=int(ckpt.get('hidden', DEFAULT_HIDDEN)), n_move_tokens=int(ckpt.get('n_move_tokens', 20000)), tie_heads=bool(ckpt.get('tie_heads', True)), device=device)
        m.load_state_dict(ckpt['state_dict'])
        m.to(device)
        m.eval()
        LOG.info('Loaded model from %s', path)
        return m


# -------------------- Misc helpers --------------------

def prepare_tensors_for_single_position(board_planes: np.ndarray, moves_seq_ids: List[int], extra_feats: np.ndarray, style_vec: np.ndarray, seq_len: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """Utility to convert numpy inputs to batched torch tensors (batch size 1) suitable for model methods."""
    board_flat = board_planes.reshape(-1)
    board_t = torch.from_numpy(board_flat[None, :].astype(np.float32))
    # moves_seq_ids: already padded list length seq_len
    mv = np.array(moves_seq_ids, dtype=np.int64)[None, :]
    mv_t = torch.from_numpy(mv)
    extra_t = torch.from_numpy(extra_feats[None, :].astype(np.float32))
    style_t = torch.from_numpy(style_vec[None, :].astype(np.float32))
    return board_t, mv_t, extra_t, style_t

# -------------------- End of file --------------------
