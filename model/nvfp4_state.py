"""NVFP4 fake-quant of the recursion-carried hidden state (fp4 project graft).

At every recursion boundary the carried hidden_states is re-encoded as packed
NVFP4: per-16-channel blocks, E2M1 code (15-value ladder) x UE4M3 scale
(126-value table), 4.5 bit/element. Bit-exact LUT simulation via midpoint
searchsorted (round-half-up), STE backward, all in fp32 then cast back.

Scale rules (fp4 canary Gate-3 finding: never per-step re-fit blindly — all
are measured here because looped residuals grow across recursions):
  amax  — stateless re-fit at each boundary (standard fake-quant baseline)
  carry — persistent index with hysteresis across boundaries within one
          forward (up if r>1, down if r<0.35 and half the block underflows);
          reset() at the first recursion entry re-anchors from amax.
  fixed — frozen per-(boundary k, channel-block) UE4M3 index, independent of
          the token and of the input. This is the LM generalisation of the
          canary "fixed" rule (one constant scale per block): here the constant
          is indexed by boundary k too, because the carried norm grows across
          recursions. Calibration: run N sequences un-quantized in calibrate
          mode, record the per-token block amax at every boundary, take a
          per-(k, block) percentile over tokens (default p99.9; "max" and p99
          kept in the same file), fold amax/6 into the UE4M3 table, freeze.
          Stored in a small .pt (see `save_fixed_scale` / `load_fixed_scale`);
          at eval/train the same install path loads it. Tokens whose block
          amax exceeds 6*s are clipped (E2M1 saturates at +-6).

  entry_frozen — per-token block scale fitted ONCE from amax at the rollout
          entry and held for every later boundary of the same forward ("fast
          code, slow scale" in its LM form). Entry = the first boundary (k=0,
          default `entry_at: boundary0`) or the state entering the first looped
          block (`entry_at: loop_input`, fed through reset(h)). Later boundaries
          whose amax outgrew the frozen scale are clipped at +-6*s: on the
          present rec3 (block amax x2 per recursion) this is expected to hurt,
          which is the point of measuring it.

Optional second-level scale (`tensor_scale: true`, any rule but fixed): the
NVFP4 per-tensor FP32 scale s_t = amax_tensor / (6*448), re-fit at every
boundary from the whole carried tensor (frozen with the block scale under
entry_frozen). Block scales become UE4M3(amax_block / (6*s_t)) <= 448 by
construction, so the 448 saturation seen at boundary 1 (block amax up to 5600
> 6*448 = 2688) disappears; decode is E2M1 * UE4M3 * s_t.

Both are off by default; the amax / carry / fixed paths are bit-identical to
the pre-variant implementation when they are off (tests/test_nvfp4_state.py).

No parameters, no registered buffers: checkpoints are unaffected; must be
re-installed at load (pretrain + eval both do, same as residual_scale).
"""
import math

import torch
from torch import nn

_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

BLOCK = 16
FIXED_PERCENTILES = ("max", 99.9, 99.0)   # all folded at calibration; yaml picks one


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


def percentile_key(p):
    """'max' | 99.9 | '99.9' | 'p99.9' -> canonical key used in the .pt tables."""
    if isinstance(p, str):
        s = p.strip().lower()
        if s == "max":
            return "max"
        s = s[1:] if s.startswith("p") else s
        p = float(s)
    p = float(p)
    return "max" if p >= 100.0 else f"p{p:g}"


class NVFP4State(nn.Module):
    def __init__(self, scale_rule="amax", fixed_idx=None, calibrate=False,
                 tensor_scale=False, entry_at="boundary0"):
        super().__init__()
        assert scale_rule in ("amax", "carry", "fixed", "entry_frozen")
        assert entry_at in ("boundary0", "loop_input"), entry_at
        assert not (tensor_scale and scale_rule == "fixed"), "fixed tables were calibrated without a tensor scale"
        self.scale_rule = scale_rule
        self.tensor_scale = bool(tensor_scale)   # NVFP4 per-tensor FP32 second-level scale
        self.entry_at = entry_at                 # entry_frozen only
        self._tab = {}      # device -> dict of plain tensors (not buffers)
        self._s_idx = None  # carry / entry_frozen state, lives only within one forward pass
        self._s_t = None    # entry_frozen + tensor_scale: frozen per-tensor scale (0-d fp32)
        self._k = 0         # boundary counter within one forward (0-based)
        # fixed: long tensor [K, nb] of UE4M3 indices, K = number of boundaries
        self._fixed_idx = None if fixed_idx is None else fixed_idx.long().cpu()
        # calibrate: pass-through, record per-token block amax / token norm per boundary
        self.calibrate = bool(calibrate)
        self.calib_amax = {}    # k -> list of [N_tok, nb] fp32 cpu
        self.calib_norm = {}    # k -> list of [N_tok] fp32 cpu
        if scale_rule == "fixed" and not self.calibrate:
            assert self._fixed_idx is not None, "scale=fixed needs fixed_idx (load_fixed_scale) or calibrate=True"

    def _tables(self, device):
        if device not in self._tab:
            e = torch.tensor(_E2M1, device=device)
            u = torch.tensor(_ue4m3_values(), device=device)
            self._tab[device] = dict(
                e2m1=e, e2m1_mid=(e[1:] + e[:-1]) / 2,
                ue4m3=u, ue4m3_mid=(u[1:] + u[:-1]) / 2)
        return self._tab[device]

    def reset(self, h=None):
        """Called once per forward at the first recursion entry. `h` (the state
        entering the first looped block) is only used by entry_frozen with
        entry_at=loop_input; every other rule ignores it."""
        self._s_idx = None
        self._s_t = None
        self._k = 0
        if self.scale_rule == "entry_frozen" and self.entry_at == "loop_input" and h is not None and not self.calibrate:
            t = self._tables(h.device)
            u = h.detach().float().view(*h.shape[:-1], h.shape[-1] // BLOCK, BLOCK)
            amax = u.abs().amax(-1)
            s_t = self._tensor_scale(amax) if self.tensor_scale else None
            self._s_idx, self._s_t = self._amax_scale_idx(t, amax, s_t), s_t

    # ------------------------------------------------------------------ scale fits
    @staticmethod
    def _tensor_scale(amax):
        """NVFP4 second-level per-tensor FP32 scale: block amax / (6*s_t) <= 448
        by construction. Detached: the scale is a constant for the STE, exactly
        like the UE4M3 index."""
        return (amax.detach().max() / (6.0 * 448.0)).clamp(min=2.0 ** -20)

    @staticmethod
    def _amax_scale_idx(t, amax, s_t=None):
        """UE4M3 index of the per-block scale re-fit from amax; s_t=None is the
        original (no tensor scale) expression, kept verbatim for bit-exactness."""
        x = amax / 6 if s_t is None else amax / (6.0 * s_t)
        return torch.searchsorted(t["ue4m3_mid"], x.clamp(min=2.0 ** -9, max=448.0).contiguous())

    # ------------------------------------------------------------------ calibrate
    def _record(self, h, amax):
        k = self._k
        self.calib_amax.setdefault(k, []).append(amax.detach().reshape(-1, amax.shape[-1]).float().cpu())
        self.calib_norm.setdefault(k, []).append(h.detach().float().norm(dim=-1).reshape(-1).cpu())

    def calibration_tables(self, percentiles=FIXED_PERCENTILES):
        """Fold the recorded amax into frozen UE4M3 indices, one table per
        percentile, plus per-boundary audit statistics. Percentile = nearest
        rank from above (ceil), so p covers at least p% of calibration tokens."""
        assert self.calib_amax, "nothing recorded: run forward passes in calibrate mode first"
        t = self._tables(torch.device("cpu"))
        ks = sorted(self.calib_amax)
        assert ks == list(range(len(ks))), f"boundary counter gaps: {ks}"
        out = {"percentiles": {}, "amax_q": {}, "stats": [], "clip_frac": {}}
        idx_tabs = {percentile_key(p): [] for p in percentiles}
        q_tabs = {percentile_key(p): [] for p in percentiles}
        clip = {percentile_key(p): [] for p in percentiles}
        for k in ks:
            a = torch.cat(self.calib_amax[k], 0)          # [N, nb]
            n = torch.cat(self.calib_norm[k], 0)          # [N]
            a_sorted, _ = a.sort(dim=0)
            N = a.shape[0]
            flat = a.reshape(-1)
            fs, _ = flat.sort()
            def _q(sorted_1d, p):
                m = sorted_1d.numel()
                i = m - 1 if p >= 100.0 else min(m - 1, max(0, math.ceil(p / 100.0 * m) - 1))
                return sorted_1d[i].item()
            out["stats"].append(dict(
                k=k, n_tokens=N, n_blocks=a.shape[1],
                amax_median=_q(fs, 50.0), amax_p99=_q(fs, 99.0), amax_p999=_q(fs, 99.9),
                amax_max=_q(fs, 100.0), amax_mean=flat.mean().item(),
                norm_mean=n.mean().item(), norm_median=n.median().item(), norm_max=n.max().item(),
            ))
            for p in percentiles:
                key = percentile_key(p)
                if key == "max":
                    aq = a_sorted[-1]
                else:
                    i = min(N - 1, max(0, math.ceil(float(p) / 100.0 * N) - 1))
                    aq = a_sorted[i]
                s_idx = torch.searchsorted(t["ue4m3_mid"], (aq / 6).clamp(min=2.0 ** -9, max=448.0).contiguous())
                s = t["ue4m3"][s_idx]
                idx_tabs[key].append(s_idx)
                q_tabs[key].append(aq)
                clip[key].append((a > 6 * s).float().mean().item())
        for key in idx_tabs:
            out["percentiles"][key] = torch.stack(idx_tabs[key], 0)   # [K, nb] long
            out["amax_q"][key] = torch.stack(q_tabs[key], 0)          # [K, nb] float
            out["clip_frac"][key] = clip[key]                          # per k
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, h):
        t = self._tables(h.device)
        shape, dtype = h.shape, h.dtype
        u = h.float().view(*shape[:-1], shape[-1] // BLOCK, BLOCK)
        amax = u.abs().amax(-1)
        if self.calibrate:
            self._record(h, amax)
            self._k += 1
            return h
        s_t = None
        if self.tensor_scale:
            frozen = self.scale_rule == "entry_frozen" and self._s_t is not None
            s_t = self._s_t if frozen else self._tensor_scale(amax)
        s_amax_idx = self._amax_scale_idx(t, amax, s_t)
        if self.scale_rule == "fixed":
            k = self._k
            assert k < self._fixed_idx.shape[0], f"boundary {k} beyond fixed table K={self._fixed_idx.shape[0]} (reset() missing?)"
            assert self._fixed_idx.shape[1] == amax.shape[-1], "fixed table block count != hidden/16"
            s_idx = self._fixed_idx[k].to(h.device)          # [nb], broadcasts over tokens
        elif self.scale_rule == "entry_frozen":
            if self._s_idx is None:                           # rollout entry (boundary0) unless reset(h) already fixed it
                self._s_idx, self._s_t = s_amax_idx.detach(), s_t
            assert self._s_idx.shape == amax.shape, f"entry_frozen scale shape {tuple(self._s_idx.shape)} != {tuple(amax.shape)} (reset() missing?)"
            s_idx, s_t = self._s_idx, self._s_t
        elif self.scale_rule == "amax" or self._s_idx is None:
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
        self._k += 1
        s = t["ue4m3"][s_idx].unsqueeze(-1)
        if s_t is not None:
            s = s * s_t                                       # decode = E2M1 * UE4M3 * per-tensor FP32
        q = torch.searchsorted(t["e2m1_mid"], (u / s).contiguous())
        hard = (t["e2m1"][q] * s).view(shape)
        soft = ((u / s).clamp(-6.0, 6.0) * s).view(shape)
        return (soft + (hard - soft).detach()).to(dtype)


# ---------------------------------------------------------------------- fixed-scale file
def save_fixed_scale(path, tables, meta):
    """tables: output of NVFP4State.calibration_tables; meta: dict (exp, ends, calib source...)."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = dict(block=BLOCK, **meta,
                   percentiles=tables["percentiles"], amax_q=tables["amax_q"],
                   clip_frac=tables["clip_frac"], stats=tables["stats"])
    torch.save(payload, path)
    return payload


def load_fixed_scale(path, percentile=99.9, expected_boundaries=None):
    """Return the frozen [K, nb] UE4M3 index table for `percentile` from a
    calibration file. Fails loudly (no silent un-quantized eval)."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"nvfp4 fixed scale file not found: {path} (run calibrate_nvfp4_fixed.py first)")
    d = torch.load(path, map_location="cpu")
    key = percentile_key(percentile)
    if key not in d["percentiles"]:
        raise KeyError(f"percentile {key} not in {path}; available: {sorted(d['percentiles'])}")
    idx = d["percentiles"][key].long()
    if expected_boundaries is not None and idx.shape[0] != expected_boundaries:
        raise ValueError(f"fixed table has K={idx.shape[0]} boundaries, model has {expected_boundaries}")
    return idx, d
