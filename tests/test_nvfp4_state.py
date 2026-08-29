"""CPU unit tests for model/nvfp4_state.py (no GPU, ~seconds).

  python tests/test_nvfp4_state.py        # or: pytest tests/test_nvfp4_state.py

1. default-off bit-exactness: amax / carry / fixed outputs AND STE gradients are
   torch.equal to the pre-variant implementation (commit d75e45f), across three
   boundaries with reset() in between and with the new reset(h) signature.
2. tensor_scale: block amax 5600 (> 6*448 = 2688) saturates the UE4M3 ladder
   without it (max element decoded as 2688) and decodes within E2M1+UE4M3
   rounding with it; the block scale never pins at 448.
3. entry_frozen: the per-token block scale (and s_t) is fitted at boundary 0 and
   reused at boundaries 1, 2 even though their amax grew x2 per boundary, so the
   later boundaries are clipped at +-6*s0; boundary 0 equals the amax rule
   bit-for-bit. entry_at=loop_input fits from the tensor passed to reset(h).
4. tiny recursive LlamaForCausalLM on CPU: install_nvfp4_state with each variant,
   forward + backward run, ends/start are unchanged, and the disabled model is
   untouched.
"""
import os
import subprocess
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.nvfp4_state import NVFP4State, BLOCK  # noqa: E402

REF_COMMIT = "d75e45f"


def _reference_module():
    """The quantizer as it was before tensor_scale / entry_frozen existed."""
    src = subprocess.check_output(["git", "-C", ROOT, "show", f"{REF_COMMIT}:model/nvfp4_state.py"], text=True)
    mod = types.ModuleType("nvfp4_state_ref")
    exec(compile(src, "nvfp4_state_ref.py", "exec"), mod.__dict__)
    return mod


def _run_boundaries(q, hs, reset_arg=None, grad=True):
    """One forward's worth of boundaries: reset once, quantize each h in turn."""
    q.reset(reset_arg) if reset_arg is not None else q.reset()
    outs = []
    for h in hs:
        x = h.clone().requires_grad_(grad)
        y = q(x)
        if grad:
            y.backward(torch.arange(y.numel(), dtype=torch.float32).view_as(y).sin())
            outs.append((y.detach(), x.grad.clone()))
        else:
            outs.append((y.detach(), None))
    return outs


def _growing_hidden(seed=0, B=2, T=5, D=64, base=30.0, growth=2.0, K=3, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    hs = []
    for k in range(K):
        h = torch.randn(B, T, D, generator=g) * base * growth ** k
        h[0, 0, :BLOCK] *= 40          # one massive-activation block per boundary
        hs.append(h.to(dtype))
    return hs


def test_default_off_bit_exact():
    ref = _reference_module()
    hs = _growing_hidden()
    for rule in ("amax", "carry"):
        for dtype in (torch.bfloat16, torch.float32):
            a = [h.to(dtype) for h in hs]
            new = _run_boundaries(NVFP4State(rule), a, reset_arg=a[0])   # new reset(h) signature
            old = _run_boundaries(ref.NVFP4State(rule), a)
            for k, ((yn, gn), (yo, go)) in enumerate(zip(new, old)):
                assert torch.equal(yn, yo), f"{rule} {dtype} boundary {k}: output differs"
                assert torch.equal(gn, go), f"{rule} {dtype} boundary {k}: STE grad differs"
    # fixed: same frozen table -> identical
    nb = hs[0].shape[-1] // BLOCK
    tab = torch.randint(60, 120, (3, nb))
    new = _run_boundaries(NVFP4State("fixed", fixed_idx=tab), hs, reset_arg=hs[0])
    old = _run_boundaries(ref.NVFP4State("fixed", fixed_idx=tab), hs)
    for k, ((yn, gn), (yo, go)) in enumerate(zip(new, old)):
        assert torch.equal(yn, yo) and torch.equal(gn, go), f"fixed boundary {k} differs"
    # calibrate: pass-through and identical records
    qn, qo = NVFP4State("fixed", calibrate=True), ref.NVFP4State("fixed", calibrate=True)
    qn.reset(hs[0]); qo.reset()
    for h in hs:
        assert torch.equal(qn(h), h) and torch.equal(qo(h), h)
    for k in qo.calib_amax:
        assert all(torch.equal(x, y) for x, y in zip(qn.calib_amax[k], qo.calib_amax[k]))
    print("ok  default-off amax/carry/fixed/calibrate bit-exact vs", REF_COMMIT)


def test_tensor_scale_removes_448_saturation():
    h = torch.randn(1, 4, 64) * 20
    h[0, 1, :BLOCK] = torch.linspace(-5600, 5600, BLOCK)      # block amax 5600 > 2688
    plain, ts = NVFP4State("amax"), NVFP4State("amax", tensor_scale=True)
    plain.reset(h); ts.reset(h)
    yp, yt = plain(h), ts(h)
    m = h.abs().max()
    err_p = (yp[0, 1, :BLOCK].abs().max() - m).abs() / m
    err_t = (yt[0, 1, :BLOCK].abs().max() - m).abs() / m
    assert yp[0, 1, :BLOCK].abs().max() == 6 * 448, f"plain path should pin at 6*448=2688, got {yp[0,1,:BLOCK].abs().max()}"
    assert err_p > 0.4, f"plain path should be saturated: rel err {err_p:.3f}"
    assert err_t < 0.1, f"tensor_scale should decode the 5600 block: rel err {err_t:.3f}"
    # block scale under the tensor scale never pins at the UE4M3 top (idx 125 = 448)
    t = ts._tables(h.device)
    u = h.float().view(1, 4, 64 // BLOCK, BLOCK)
    amax = u.abs().amax(-1)
    s_t = ts._tensor_scale(amax)
    idx = ts._amax_scale_idx(t, amax, s_t)
    assert (amax / (6 * s_t)).max() <= 448.0 and idx.max() <= 125
    assert idx[0, 1, 0] < 125 or torch.isclose(t["ue4m3"][idx[0, 1, 0]], torch.tensor(448.0)) and (amax[0, 1, 0] / (6 * s_t)) <= 448
    # non-saturating input: the two ladders differ, but reconstruction error must be comparable
    h2 = torch.randn(2, 8, 128) * 30
    plain.reset(h2); ts.reset(h2)
    e_plain = (plain(h2) - h2).float().pow(2).mean()
    e_ts = (ts(h2) - h2).float().pow(2).mean()
    assert e_ts < 1.5 * e_plain and e_plain < 1.5 * e_ts, f"tensor_scale changed non-saturating error: {e_plain:.3f} vs {e_ts:.3f}"
    # STE: gradient is the clipped-identity of the (larger) effective range
    x = h.clone().requires_grad_(True)
    ts.reset(x); ts(x).sum().backward()
    assert x.grad[0, 1, :BLOCK].abs().sum() > 0
    print(f"ok  tensor_scale: rel err of the 5600 element {err_p:.3f} -> {err_t:.4f}, s_t={s_t.item():.4f}")


def test_entry_frozen():
    hs = _growing_hidden(seed=1, K=3)
    for ts in (False, True):
        q = NVFP4State("entry_frozen", tensor_scale=ts)
        a = NVFP4State("amax", tensor_scale=ts)
        q.reset(hs[0]); a.reset(hs[0])
        y0, y0a = q(hs[0]), a(hs[0])
        assert torch.equal(y0, y0a), "entry_frozen boundary 0 must equal the amax rule"
        s0, st0 = q._s_idx.clone(), (None if q._s_t is None else q._s_t.clone())
        t = q._tables(hs[0].device)
        for k in (1, 2):
            yk = q(hs[k])
            assert torch.equal(q._s_idx, s0), f"boundary {k}: block scale was re-fitted"
            if ts:
                assert torch.equal(q._s_t, st0), f"boundary {k}: tensor scale was re-fitted"
            s = t["ue4m3"][s0].unsqueeze(-1) * (st0 if ts else 1.0)
            cap = (6 * s).expand(*s.shape[:-1], BLOCK).reshape(hs[k].shape)
            assert (yk.float().abs() <= cap.float() * (1 + 2 ** -7)).all(), f"boundary {k}: decoded beyond +-6*s0"
            # the grown boundary really is clipped (rec3: amax x2 per recursion)
            clip_frac = (hs[k].float().abs() > cap.float()).float().mean().item()
            assert clip_frac > 0.05, f"boundary {k}: expected clipping, got {clip_frac:.3f}"
        # amax rule with the same shape must differ at k>0
        a.reset(hs[0]); a(hs[0])
        assert not torch.equal(a(hs[1]), q._tables and _run_boundaries(NVFP4State("entry_frozen", tensor_scale=ts), hs, reset_arg=hs[0], grad=False)[1][0])
        # boundary counter and reset semantics
        assert q._k == 3
        q.reset(hs[0]); assert q._s_idx is None and q._s_t is None and q._k == 0
        # STE backward works across all three boundaries
        _run_boundaries(NVFP4State("entry_frozen", tensor_scale=ts), hs, reset_arg=hs[0])
    # entry_at=loop_input: the scale comes from reset(h), not from boundary 0
    q = NVFP4State("entry_frozen", entry_at="loop_input", tensor_scale=True)
    h_in = hs[0] * 0.5
    q.reset(h_in)
    assert q._s_idx is not None and q._s_idx.shape == hs[0].shape[:-1] + (hs[0].shape[-1] // BLOCK,)
    t = q._tables(h_in.device)
    u = h_in.float().view(*h_in.shape[:-1], -1, BLOCK); am = u.abs().amax(-1)
    st = q._tensor_scale(am)
    assert torch.equal(q._s_idx, q._amax_scale_idx(t, am, st)) and torch.equal(q._s_t, st)
    y = q(hs[0]); assert torch.equal(q._s_idx, q._amax_scale_idx(t, am, st)), "loop_input entry scale re-fitted at boundary 0"
    q2 = NVFP4State("entry_frozen", entry_at="loop_input")
    q2.reset()                                # no h -> falls back to boundary 0
    q2(hs[0]); assert q2._s_idx is not None
    print("ok  entry_frozen (+tensor_scale, loop_input): scale frozen from the entry, later boundaries clipped")


def test_tiny_model_install():
    from omegaconf import OmegaConf
    from transformers import LlamaConfig
    from model.recursive_model.modeling_llama import LlamaForCausalLM
    torch.manual_seed(0)
    cfgm = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32, tie_word_embeddings=True)
    ids = torch.randint(0, 97, (2, 12))
    base = LlamaForCausalLM._from_config(cfgm, attn_implementation="eager", torch_dtype=torch.float32)
    ref_logits = base(input_ids=ids, labels=ids, use_cache=False).logits.detach()
    for ns in ({"enable": True, "scale": "amax"},
               {"enable": True, "scale": "amax", "tensor_scale": True},
               {"enable": True, "scale": "entry_frozen"},
               {"enable": True, "scale": "entry_frozen", "tensor_scale": True},
               {"enable": True, "scale": "entry_frozen", "tensor_scale": True, "entry_at": "loop_input"}):
        m = LlamaForCausalLM._from_config(cfgm, attn_implementation="eager", torch_dtype=torch.float32)
        m.load_state_dict(base.state_dict())
        cfg = OmegaConf.create({"recursive": {"enable": True, "num_recursion": 3, "sharing": "middle_cycle"}, "nvfp4_state": ns})
        m.install_nvfp4_state(cfg)
        assert m.model.nvfp4_start == 1 and sorted(m.model.nvfp4_ends) == [1, 2, 3]
        q = m.model.nvfp4_state
        assert q.scale_rule == ns["scale"] and q.tensor_scale == bool(ns.get("tensor_scale")) and q.entry_at == ns.get("entry_at", "boundary0")
        out = m(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        assert torch.isfinite(out.loss) and q._k == 3
        assert not torch.equal(out.logits.detach(), ref_logits), f"{ns}: quantizer had no effect"
        if ns["scale"] == "entry_frozen":
            assert q._s_idx is not None
    # untouched model: nvfp4_state None, identical logits
    assert base.model.nvfp4_state is None
    assert torch.equal(base(input_ids=ids, labels=ids, use_cache=False).logits.detach(), ref_logits)
    print("ok  tiny recursive model: install/forward/backward for all variants, disabled model untouched")


if __name__ == "__main__":
    test_default_off_bit_exact()
    test_tensor_scale_removes_448_saturation()
    test_entry_frozen()
    test_tiny_model_install()
    print("ALL PASSED")
