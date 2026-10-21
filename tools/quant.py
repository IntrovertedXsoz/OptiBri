"""Quantizers shared by experiments: block-Hadamard rotation, MSE-optimal uniform RTN, layer-shared VQ."""
import math, torch

def _had(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h
_HK = {}
def fwht(x):
    """Orthonormal Walsh-Hadamard on the last dim (power of 2), via H_n = H_a (x) H_b: two small BLAS matmuls."""
    n = x.shape[-1]
    if n not in _HK:
        a = 2 ** (int(math.log2(n)) // 2); _HK[n] = (_had(a), _had(n // a))
    ha, hb = (h.to(x.dtype) for h in _HK[n]); a, b = ha.shape[0], hb.shape[0]
    y = x.reshape(-1, a, b)
    y = torch.matmul(torch.matmul(ha, y), hb)       # H_a Y H_b (both symmetric); bf16 input -> AMX matmul
    return y.reshape(x.shape) * (1.0 / math.sqrt(n))

SIGNS = {}
def rotate(w):                                  # W -> W R^T with R = blockdiag(H * diag(signs)), blocks of 2048
    i = w.shape[1]
    if i not in SIGNS: SIGNS[i] = (torch.randint(0, 2, (i,)) * 2 - 1).float()
    return fwht((w * SIGNS[i].to(w.dtype)).view(w.shape[0], i // 2048, 2048)).view_as(w)
def unrotate(w):
    i = w.shape[1]; return fwht(w.view(w.shape[0], i // 2048, 2048)).view_as(w) * SIGNS[i]

def rtn(w, b, gs):
    o, i = w.shape; wg = w.reshape(o, i // gs, gs)
    lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True); s = (hi - lo).clamp_min(1e-12) / (2 ** b - 1)
    return (((wg - lo) / s).round().clamp(0, 2 ** b - 1) * s + lo).reshape(o, i)

STEP = {2: 0.9957, 3: 0.5860, 4: 0.3352}         # MSE-optimal uniform step for a unit Gaussian (Max 1960)
def hrtn_rot(w, b):
    """Q(rotate(w)) kept in the rotated basis: use as  rotate(x) @ hrtn_rot(w, b).T  (no unrotation needed)."""
    r = rotate(w).float(); q = 2 ** (b - 1)
    s = torch.linalg.vector_norm(r, dim=1, keepdim=True).mul_(STEP[b] / math.sqrt(r.shape[1]))
    return r.div_(s).floor_().clamp_(-q, q - 1).add_(0.5).mul_(s)   # floor(x) == round(x - 0.5) off ties

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

