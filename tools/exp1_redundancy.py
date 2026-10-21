"""Exp 1: are GLM-5.2 experts redundant? (shared base, low rank, shared subspace)"""
import sys, numpy as np, hf_tensors as H
L = int(sys.argv[1]) if len(sys.argv) > 1 else 40
E = 32
for p in ("gate", "up", "down"):
    W = np.stack([H.fetch(f"model.layers.{L}.mlp.experts.{i}.{p}_proj.weight") for i in range(E)])  # E,out,in
    tot = (W**2).sum()
    mean = W.mean(0)
    base_res = ((W - mean)**2).sum() / tot
    # pairwise cosine between flattened experts
    F = W.reshape(E, -1); F = F / np.linalg.norm(F, axis=1, keepdims=True); C = F @ F.T
    off = C[~np.eye(E, dtype=bool)]
    # per-expert singular spectrum: energy captured by top r
    s = np.linalg.svd(W[0], compute_uv=False); e = np.cumsum(s**2) / (s**2).sum()
    ranks = {f: int(np.searchsorted(e, f)) + 1 for f in (0.5, 0.9, 0.99)}
    # shared subspace along the 6144 (hidden) side
    Wh = W if p != "down" else W.transpose(0, 2, 1)   # E, 2048, 6144
    M = np.ascontiguousarray(Wh).reshape(-1, Wh.shape[2]); G = M.T @ M   # 6144x6144 gram, BLAS
    ev = np.linalg.eigvalsh(G)[::-1]; ce = np.cumsum(ev) / ev.sum()
    hk = {f: int(np.searchsorted(ce, f)) + 1 for f in (0.5, 0.9, 0.99)}
    print(f"L{L} {p}: shape {W.shape[1:]} | residual after shared mean {base_res:.4f} | "
          f"cos mean {off.mean():.4f} max {np.abs(off).max():.4f} | per-expert rank for 50/90/99% energy {ranks} "
          f"| shared hidden-subspace dims for 50/90/99% {hk}", flush=True)
