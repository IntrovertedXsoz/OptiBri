"""Exp 5: routing locality. For windows of k consecutive tokens, how many distinct experts does a layer need?
If k tokens need far fewer than 8k experts, verifying k drafted tokens in one pass amortizes expert reads.
Also: how many of a token's experts were already used by the previous W tokens (RAM-cache hit rate proxy)."""
import sys, numpy as np
D = sys.argv[1]; T = 1024
for L in (3, 4, 10):
    idx = np.load(f"{D}/idx{L}.npy").astype(np.int64).reshape(8, T, 8)
    line = [f"L{L}"]
    for k in (1, 2, 4, 8, 16):
        u = [len(np.unique(idx[s, t:t + k])) for s in range(8) for t in range(0, T - k, k)]
        line.append(f"k={k}: {np.mean(u):.1f} experts ({np.mean(u) / (8 * k):.0%} of 8k)")
    for W in (1, 4, 16, 64):
        hit = [np.isin(idx[s, t], idx[s, t - W:t]).mean() for s in range(8) for t in range(W, T)]
        line.append(f"hit(prev {W})={np.mean(hit):.0%}")
    print(" | ".join(line))
