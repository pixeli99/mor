"""DecayTrajAttn ported verbatim from fprm-official (models/loop_attn.py).

The loop axis here is the recursion axis of the recursive Llama variant:
every recursion's block output is written to a running exponentially-weighted
average, and the read N/D replaces the carry-last hidden state that feeds the
next recursion (and the coda/head after the last one).
"""
from typing import List, Tuple

import torch
from torch import nn


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float = 1e-6) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class DecayTrajAttn(nn.Module):
    """O(1) trajectory readout over one loop axis, replacing the carry-last residual.

    Every iterate written to the axis enters a running exponentially-weighted
    average; the read N/D is what the next iteration (and the final head) sees:

        weight(source at age a) ∝ exp(w_h · RMSNorm(source) / temp) · beta_h^a

        N ← beta·N + e^{s}·o ;  D ← beta·D + e^{s} ;  read = N/D.

    beta = sigmoid(logit) is learnable per head; heads>1 reads the trajectory at
    multiple timescales (per-head decay, RetNet-style, but on the loop axis).
    w is zero-init, so at step 0 the read is a pure decay-weighted average;
    content=False freezes w at 0 (the pure learned-beta EMA ablation). Values
    are the raw states, so the read stays in their convex hull — bounded at any
    eval depth, no growing-history OOD blowup (accumulators run in fp32).
    """

    def __init__(self, hidden_size: int, heads: int = 1, beta_init: float = 0.1,
                 temp: float = 1.0, content: bool = True, beta_spread: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        assert hidden_size % heads == 0
        self.heads = heads
        self.hdim = hidden_size // heads
        self.temp = temp
        self.eps = eps
        self.w = nn.Parameter(torch.zeros(heads, hidden_size), requires_grad=content)
        # beta_spread (default): heads>1 get a fast..slow ladder, which is what makes
        # them multi-timescale. Set it False to give every head the same beta_init, so
        # "more heads" can be measured apart from "different initial timescales".
        beta = (torch.linspace(0.2, 0.9, heads) if (heads > 1 and beta_spread)
                else torch.full((heads,), float(beta_init)))
        self.beta_logit = nn.Parameter(torch.logit(beta))
        # Diagnostics, filled only when .collect is on. Kept as GPU tensors; the
        # logger syncs them.
        self.collect = False
        self.stats = {}

    def _weight(self, s: torch.Tensor) -> torch.Tensor:
        # exp(content logit), fp32 [B, T, H, 1]; clamp keeps e^s finite in fp32.
        logit = torch.einsum("hd,btd->bth", self.w.to(s.dtype), rms_norm(s, self.eps)).float() / self.temp
        if self.collect:
            with torch.no_grad():   # how selective the content query is across tokens
                self.stats["logit_std"] = logit.std()
        return logit.clamp(-8.0, 8.0).exp().unsqueeze(-1)

    def init_state(self, anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = anchor.shape
        e = self._weight(anchor)
        return e * anchor.view(B, T, self.heads, self.hdim).float(), e

    def forward(self, writes: List[torch.Tensor], state):
        N, D = state
        beta = self.beta_logit.sigmoid().float().view(1, 1, self.heads, 1)
        N, D = beta * N, beta * D
        for o in writes:
            B, T, _ = o.shape
            e = self._weight(o)
            N = N + e * o.view(B, T, self.heads, self.hdim).float()
            D = D + e
        read = (N / D).flatten(2).to(writes[-1].dtype)
        if self.collect:
            with torch.no_grad():
                last = writes[-1]
                # share of the read that comes from the newest write. 1.0 => the read IS
                # carry-last and the attention is a no-op; lower => history is being used.
                self.stats["newest_frac"] = (e / D).mean()
                # how far the read moves the state away from plain carry-last, relative
                # to the state's own norm. ~0 => no-op regardless of what beta says.
                self.stats["read_delta"] = (
                    (read - last).float().norm(dim=-1) / (last.float().norm(dim=-1) + 1e-6)
                ).mean()
        return read, (N, D)
