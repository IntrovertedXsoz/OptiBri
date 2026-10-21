"""Exp 3: does the shared expert-input subspace generalize to unseen text? And do outputs share one too?

Fit the layer's input subspace (top-k eigenvectors of X^T X) on FIT sequences, evaluate expert output error on
held-out sequences: (a) same domain (Wikipedia), (b) a domain absent from the fit (Python code).
Output side: fit the top-k eigenvectors of the layer's routed-expert outputs Y and project each expert's down_proj
output onto them.
Usage: python exp3_subspace_heldout.py FWD_DIR LAYER
"""
import os, sys, numpy as np, torch
import glm_ref as G
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
D, L = sys.argv[1], int(sys.argv[2]); T = 1024
X = torch.from_numpy(np.load(f"{D}/x{L}.npy").astype(np.float32))
idx = torch.from_numpy(np.load(f"{D}/idx{L}.npy").astype(np.int64))
seq = torch.arange(X.shape[0]) // T                     # seqs 0-5 Wikipedia, 6-7 Python code
FIT = seq < 3; HELD_WIKI = (seq >= 3) & (seq < 6); HELD_CODE = seq >= 6

def basis(Z):
    ev, U = torch.linalg.eigh((Z.T @ Z).double()); return U.flip(1).float()

Uin = basis(X[FIT])
def silu_mul(g, u, x): return torch.nn.functional.silu(x @ g.T) * (x @ u.T)
def rel(a, b): return ((a - b).norm() / b.norm()).item()

cnt = torch.bincount(idx.flatten(), minlength=256)
experts = cnt.argsort(descending=True)[:4].tolist() + cnt.argsort(descending=True)[120:124].tolist()
ks = (1024, 1536, 2048, 3072)
acc = {}; Yfit = []
cache = {}
for e, (g, u, d) in G.ExpertStream(L, experts):
    cache[e] = (g, u, d)
    for name, mask in (("wiki", HELD_WIKI), ("code", HELD_CODE), ("fit", FIT)):
        tok = ((idx == e).any(-1) & mask).nonzero().flatten()
        if len(tok) < 8: continue
        xe = X[tok]; y = silu_mul(g, u, xe) @ d.T
        for k in ks:
            P = Uin[:, :k]
            acc.setdefault(f"in{k}_{name}", []).append(rel(silu_mul(g, u, (xe @ P) @ P.T) @ d.T, y))
        if name == "fit": Yfit.append(y)
# output side: basis from FIT outputs of these experts (proxy for the layer's output space)
Uout = basis(torch.cat(Yfit))
for e, (g, u, d) in cache.items():
    for name, mask in (("wiki", HELD_WIKI), ("code", HELD_CODE)):
        tok = ((idx == e).any(-1) & mask).nonzero().flatten()
        if len(tok) < 8: continue
        y = silu_mul(g, u, X[tok]) @ d.T
        for k in ks:
            P = Uout[:, :k]
            acc.setdefault(f"out{k}_{name}", []).append(rel((y @ P) @ P.T, y))
for key in sorted(acc, key=lambda s: (s.split("_")[1], s)):
    print(f"L{L} {key:14s} mean {np.mean(acc[key]):.3f}  max {np.max(acc[key]):.3f}  (n={len(acc[key])})")
