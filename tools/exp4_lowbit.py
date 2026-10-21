"""Exp 4: how good can ~2-bit experts get? Weight-only quantizers (no activation fitting, so no overfitting).

Methods (bits/weight include all scale/codebook-index overhead; layer-shared codebooks are kept in RAM, not counted):
  rtn{b}g{gs}  asymmetric round-to-nearest per group (fp16 scale + fp16 min per group)
  hrtn{b}      random-sign block-Hadamard rotation of the input dim, then per-row symmetric RTN (fp16 row scale)
  hvq{d}x{K}   rotation + per-row RMS normalisation + d-dim vector quantization with one K-entry codebook shared by
               the whole layer (fit on 2 experts, applied to all others): log2(K)/d bits/weight
Metric: relative error of the expert's output on the real tokens routed to it.
Usage: python exp4_lowbit.py FWD_DIR LAYER
"""
import os, sys, math, numpy as np, torch
import glm_ref as G
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
torch.manual_seed(0)
D, L = sys.argv[1], int(sys.argv[2])
X = torch.from_numpy(np.load(f"{D}/x{L}.npy").astype(np.float32))
idx = torch.from_numpy(np.load(f"{D}/idx{L}.npy").astype(np.int64))

from quant import fwht, rotate, unrotate, rtn, hrtn, assign, kmeans, vq_prep, vq_apply, SIGNS

def out(g, u, d, x): return (torch.nn.functional.silu(x @ g.T) * (x @ u.T)) @ d.T
def rel(a, b): return ((a - b).norm() / b.norm()).item()

cnt = torch.bincount(idx.flatten(), minlength=256)
order = cnt.argsort(descending=True).tolist()
fit_experts, test_experts = order[200:202], order[:4] + order[100:104]
VQ = [(2, 16), (4, 256), (4, 1024), (8, 65536)]
books = {}
for e, w3 in G.ExpertStream(L, fit_experts):
    for d, K in VQ:
        if d == 8 and K == 65536: K = 4096   # CPU budget: 8-dim with 4096 entries = 1.5 bpw (lower bound proxy)
        for j, w in enumerate(w3):
            books.setdefault((d, K, j), []).append(vq_prep(w, d)[0])
for key in list(books):
    v = torch.cat(books[key]); v = v[torch.randperm(len(v))[:400_000]]
    books[key] = kmeans(v, key[1])
methods = {"rtn2g64 (2.50)": lambda w, j: rtn(w, 2, 64), "rtn3g128 (3.25)": lambda w, j: rtn(w, 3, 128),
           "rtn4g128 (4.25)": lambda w, j: rtn(w, 4, 128), "hrtn2 (2.00)": lambda w, j: hrtn(w, 2),
           "hrtn3 (3.00)": lambda w, j: hrtn(w, 3)}
for (d, K, j) in books:
    if j == 0:
        methods[f"hvq{d}x{K} ({math.log2(K)/d:.2f})"] = (lambda d, K: lambda w, jj: vq_apply(w, d, books[(d, K, jj)]))(d, K)
acc = {}
for e, w3 in G.ExpertStream(L, test_experts):
    tok = (idx == e).any(-1).nonzero().flatten(); xe = X[tok]; y = out(*w3, xe)
    line = []
    for name, f in methods.items():
        r = rel(out(*[f(w, j) for j, w in enumerate(w3)], xe), y); acc.setdefault(name, []).append(r); line.append(f"{name.split()[0]}={r:.3f}")
    print(f"expert {e:3d}: " + " ".join(line), flush=True)
for name, v in acc.items():
    print(f"L{L} {name:18s} mean {np.mean(v):.3f} max {np.max(v):.3f}")
