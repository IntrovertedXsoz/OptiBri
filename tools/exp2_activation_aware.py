"""Exp 2: activation-aware compressibility of real GLM-5.2 experts.

Uses real MoE inputs X (from run_forward.py) and measures each method by the relative error of the
expert's *output* on the tokens actually routed to it:  ||y_hat - y|| / ||y||.
Usage: python exp2_activation_aware.py FWD_DIR LAYER
"""
import os, sys, numpy as np, torch
import glm_ref as G
torch.set_num_threads(os.cpu_count())
D, L = sys.argv[1], int(sys.argv[2])
X = torch.from_numpy(np.load(f"{D}/x{L}.npy").astype(np.float32))
idx = torch.from_numpy(np.load(f"{D}/idx{L}.npy").astype(np.int64))
N = X.shape[0]

# --- expert usage ---
cnt = torch.bincount(idx.flatten(), minlength=256).float()
srt = cnt.sort(descending=True).values; share = srt.cumsum(0) / srt.sum()
print(f"L{L} usage: top-16 experts take {share[15]:.2%} of routes, top-64 {share[63]:.2%}, "
      f"unused {(cnt == 0).sum().item()}, max/min-nonzero {srt[0]:.0f}/{srt[srt > 0][-1]:.0f}")

# --- shared input subspace ---
C = X.T @ X / N
ev, U = torch.linalg.eigh(C.double()); ev, U = ev.flip(0).float(), U.flip(1).float()
ce = ev.cumsum(0) / ev.sum()
print("input energy dims for 90/99/99.9%:", [int((ce < f).sum()) + 1 for f in (0.9, 0.99, 0.999)])

def out(g, u, d, x):
    return (torch.nn.functional.silu(x @ g.T) * (x @ u.T)) @ d.T

def rel(a, b):
    return ((a - b).norm() / b.norm()).item()

def rtn(w, bits, gs=64):
    """Round-to-nearest, asymmetric, per-group along the input dim."""
    o, i = w.shape; wg = w.reshape(o, i // gs, gs)
    lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    s = (hi - lo).clamp_min(1e-12) / (2 ** bits - 1)
    return (((wg - lo) / s).round().clamp(0, 2 ** bits - 1) * s + lo).reshape(o, i)

ks = (512, 1024, 1536, 2048, 3072)
experts = cnt.argsort(descending=True)[:6].tolist() + cnt.argsort(descending=True)[100:106].tolist()
res = {}
for e, (g, u, d) in G.ExpertStream(L, experts):
    tok = (idx == e).any(-1).nonzero().flatten()
    xe = X[tok]; y = out(g, u, d, xe)
    r = {}
    for k in ks:                       # gate/up see the input projected on the shared top-k subspace
        P = U[:, :k]; xp = (xe @ P) @ P.T
        r[f"sub{k}"] = rel(out(g, u, d, xp), y)
    for b in (2, 3, 4):
        r[f"rtn{b}"] = rel(out(rtn(g, b), rtn(u, b), rtn(d, b), xe), y)
    for rk in (256, 512, 1024):        # plain truncated SVD per matrix
        def lr(w):
            Us, S, Vh = torch.linalg.svd(w, full_matrices=False); return (Us[:, :rk] * S[:rk]) @ Vh[:rk]
        r[f"svd{rk}"] = rel(out(lr(g), lr(u), lr(d), xe), y)
    res[e] = r
    print(f"expert {e:3d} ({len(tok):4d} tok): " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
keys = next(iter(res.values())).keys()
print("MEAN " + " ".join(f"{k}={np.mean([r[k] for r in res.values()]):.3f}" for k in keys))
