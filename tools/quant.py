"""Quantizers shared by experiments: block-Hadamard rotation, MSE-optimal uniform RTN, layer-shared VQ."""
import math, torch

def fwht(x):                                    # orthonormal Walsh-Hadamard on the last dim (power of 2)
    n = x.shape[-1]; h = 1; y = x.clone()
    while h < n:
        y = y.view(*x.shape[:-1], n // (2 * h), 2, h)
        y = torch.stack([y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]], -2); h *= 2
    return y.reshape(x.shape) / math.sqrt(n)

SIGNS = {}
def rotate(w):                                  # W -> W R^T with R = blockdiag(H * diag(signs)), blocks of 2048
    i = w.shape[1]
    if i not in SIGNS: SIGNS[i] = (torch.randint(0, 2, (i,)) * 2 - 1).float()
    return fwht((w * SIGNS[i]).view(w.shape[0], i // 2048, 2048)).view_as(w)
def unrotate(w):
    i = w.shape[1]; return fwht(w.view(w.shape[0], i // 2048, 2048)).view_as(w) * SIGNS[i]

def rtn(w, b, gs):
    o, i = w.shape; wg = w.reshape(o, i // gs, gs)
    lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True); s = (hi - lo).clamp_min(1e-12) / (2 ** b - 1)
    return (((wg - lo) / s).round().clamp(0, 2 ** b - 1) * s + lo).reshape(o, i)

STEP = {2: 0.9957, 3: 0.5860, 4: 0.3352}         # MSE-optimal uniform step for a unit Gaussian (Max 1960)
def hrtn(w, b):
    r = rotate(w); q = 2 ** (b - 1)              # 2^b levels at (k + 0.5) * step, k = -q..q-1
    s = r.pow(2).mean(1, keepdim=True).sqrt() * STEP[b]
    return unrotate(((r / s - 0.5).round().clamp(-q, q - 1) + 0.5) * s)

def assign(v, c, chunk=1 << 15):
    return torch.cat([torch.cdist(v[i:i + chunk], c).argmin(1) for i in range(0, len(v), chunk)])

def kmeans(v, K, it=12):
    c = v[torch.randperm(len(v))[:K]].clone()
    for _ in range(it):
        a = assign(v, c); n = torch.bincount(a, minlength=K)
        new = torch.zeros_like(c).index_add_(0, a, v) / n.clamp_min(1)[:, None].float()
        empty = n == 0; new[empty] = v[torch.randint(0, len(v), (int(empty.sum()),))]
        c = new
    return c

def vq_prep(w, d):
    r = rotate(w); s = r.pow(2).mean(1, keepdim=True).sqrt(); return (r / s).reshape(-1, d), s
def vq_apply(w, d, c):
    v, s = vq_prep(w, d); return unrotate(c[assign(v, c)].reshape(w.shape) * s)

