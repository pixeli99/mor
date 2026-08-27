"""NVFP4 fake-quant of the recursion-carried hidden state (fp4 project graft).

At every recursion boundary the carried hidden_states is re-encoded as packed
NVFP4: per-16-channel blocks, E2M1 code (15-value ladder) x UE4M3 scale
(126-value table), 4.5 bit/element. Bit-exact LUT simulation via midpoint
searchsorted (round-half-up), STE backward, all in fp32 then cast back.

Scale rules (fp4 canary Gate-3 finding: never per-step re-fit blindly — both
are measured here because looped residuals grow across recursions):
  amax  — stateless re-fit at each boundary (standard fake-quant baseline)
  carry — persistent index with hysteresis across boundaries within one
          forward (up if r>1, down if r<0.35 and half the block underflows);
          reset() at the first recursion entry re-anchors from amax.

No parameters, no registered buffers: checkpoints are unaffected; must be
re-installed at load (pretrain + eval both do, same as residual_scale).
"""
import torch
from torch import nn

_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ue4m3_values():
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 0:
                v = (m / 8.0) * 2.0 ** -6          # subnormal
            else:
                v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
            if v > 0:
                vals.append(v)
    return sorted(set(vals))                        # 126 values, [2^-9, 448]


class NVFP4State(nn.Module):
    def __init__(self, scale_rule="amax"):
        super().__init__()
        assert scale_rule in ("amax", "carry")
        self.scale_rule = scale_rule
        self._tab = {}      # device -> dict of plain tensors (not buffers)
        self._s_idx = None  # carry state, lives only within one forward pass

    def _tables(self, device):
        if device not in self._tab:
            e = torch.tensor(_E2M1, device=device)
            u = torch.tensor(_ue4m3_values(), device=device)
            self._tab[device] = dict(
                e2m1=e, e2m1_mid=(e[1:] + e[:-1]) / 2,
                ue4m3=u, ue4m3_mid=(u[1:] + u[:-1]) / 2)
        return self._tab[device]

    def reset(self):
        self._s_idx = None

    def forward(self, h):
        t = self._tables(h.device)
        shape, dtype = h.shape, h.dtype
        u = h.float().view(*shape[:-1], shape[-1] // 16, 16)
        amax = u.abs().amax(-1)
        s_amax_idx = torch.searchsorted(t["ue4m3_mid"], (amax / 6).clamp(min=2.0 ** -9, max=448.0).contiguous())
        if self.scale_rule == "amax" or self._s_idx is None:
            s_idx = s_amax_idx
        else:
            s_prev = t["ue4m3"][self._s_idx]
            r = amax / (6 * s_prev)
            q_tmp = torch.searchsorted(t["e2m1_mid"], (u / s_prev.unsqueeze(-1)).contiguous())
            z = (q_tmp == 7).float().mean(-1)
            delta = (r > 1.0).long() - ((r < 0.35) & (z > 0.5) & (r <= 1.0)).long()
            s_idx = (self._s_idx + delta).clamp(0, 125)
        if self.scale_rule == "carry":
            self._s_idx = s_idx.detach()
        s = t["ue4m3"][s_idx].unsqueeze(-1)
        q = torch.searchsorted(t["e2m1_mid"], (u / s).contiguous())
        hard = (t["e2m1"][q] * s).view(shape)
        soft = ((u / s).clamp(-6.0, 6.0) * s).view(shape)
        return (soft + (hard - soft).detach()).to(dtype)
