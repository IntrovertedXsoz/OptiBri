"""End-to-end quality of compressed GLM-5.2 variants vs the original, in one streamed pass.

Every variant keeps its own hidden-state stream and its own routing (quantization can change which experts
are picked, a real effect). Each weight is downloaded once per layer and shared by all streams.
Only the routed experts are compressed here; everything else stays bf16, to isolate the expert budget.
At the end: KL(ref || variant), top-1 agreement and perplexity on the real next tokens.
Resumable. Usage: python run_e2e.py OUT
"""
import os, sys, json, time, numpy as np, torch
import glm_ref as G
from quant import hrtn_rot, rotate            # rotation + MSE-optimal uniform quantizer (tested in exp4)
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
# variants return weights in the rotated basis; activations are rotated instead of unrotating weights
q = lambda b: (lambda w: hrtn_rot(w, b).to(w.dtype))
VARIANTS = {"ref": None, "hrtn3": q(3), "hrtn2": q(2), "top4": None}
TOPK = {"top4": 4}   # bf16 experts, but only the 4 best-scored of the 8 routed experts (weights renormalized)
calib = np.load(os.path.join(os.environ["GLM_CACHE"], "calib_8x1024.npy"))
ids = np.stack([calib[3][:512], calib[7][:512]])      # one Wikipedia, one Python-code sequence
S, T = ids.shape; cs = G.rope_cs(T)

# per-layer memo for non-expert weights so each is downloaded once for all streams
_W = G.W; memo = {}
def W(name):
    if ".experts." in name: return _W(name)
    if name not in memo: memo[name] = _W(name)
    return memo[name]
G.W = W

state = os.path.join(OUT, "state.pt")
if os.path.exists(state):
    st = torch.load(state); h = st["h"]; start = st["layer"] + 1
else:
    h0 = G.embed(ids.flatten().tolist()).view(S, T, -1); h = {k: h0.clone() for k in VARIANTS}; start = 0

def moe_multi(L, xs):
    out, routes = {}, {}
    for k, x in xs.items():
        idx, wts = G.route(L, x)
        if k in TOPK:                                    # route() returns experts sorted by (biased) score
            n = TOPK[k]; s = wts.sum(-1, keepdim=True); idx, wts = idx[:, :n], wts[:, :n]
            wts = wts / wts.sum(-1, keepdim=True) * s
        routes[k] = (idx, wts); out[k] = G.mlp(f"model.layers.{L}.mlp.shared_experts.", x)
    used = sorted(set().union(*[set(r[0].flatten().tolist()) for r in routes.values()]))
    for e, w3 in G.ExpertStream(L, used):
        qcache = {}
        for k, x in xs.items():
            idx, wts = routes[k]; tok, slot = (idx == e).nonzero(as_tuple=True)
            if len(tok) == 0: continue
            if k not in qcache: qcache[k] = w3 if VARIANTS[k] is None else [VARIANTS[k](w) for w in w3]
            g, u, d = qcache[k]; xe = x[tok]; rot = VARIANTS[k] is not None
            if rot: xe = rotate(xe)
            a = torch.nn.functional.silu(G.mm(xe, g)) * G.mm(xe, u)
            if rot: a = rotate(a)
            out[k].index_add_(0, tok, G.mm(a, d) * wts[tok, slot, None])
    agree = {k: (routes[k][0].sort(-1).values == routes["ref"][0].sort(-1).values).float().mean().item()
             for k in xs if k not in TOPK}
    return out, agree

NL = G.CFG["num_hidden_layers"]
for L in range(start, NL):
    t0 = time.time(); memo.clear(); p = f"model.layers.{L}."; agree = {}
    for k in VARIANTS:
        hk = h[k]
        hk = hk + torch.stack([G.attention(L, G.rms(hk[s], W(p + "input_layernorm.weight")), cs) for s in range(S)])
        h[k] = hk
    xs = {k: G.rms(h[k], W(p + "post_attention_layernorm.weight")).view(S * T, -1) for k in VARIANTS}
    if G.CFG["mlp_layer_types"][L] == "dense":
        ys = {k: G.mlp(p + "mlp.", x) for k, x in xs.items()}
    else:
        ys, agree = moe_multi(L, xs)
    for k in VARIANTS: h[k] = h[k] + ys[k].view(S, T, -1)
    torch.save({"h": h, "layer": L}, state + ".tmp"); os.replace(state + ".tmp", state)
    dev = {k: ((h[k] - h["ref"]).norm() / h["ref"].norm()).item() for k in VARIANTS if k != "ref"}
    print(f"layer {L} {time.time()-t0:.0f}s hidden-dev " + " ".join(f"{k}={v:.3f}" for k, v in dev.items())
          + " route-agree " + " ".join(f"{k}={v:.3f}" for k, v in agree.items() if k != "ref"), flush=True)

memo.clear(); res = {}
logp = {k: torch.log_softmax(G.head(h[k].view(S * T, -1)), -1) for k in VARIANTS}
tgt = torch.from_numpy(ids[:, 1:].reshape(-1)); keep = torch.ones(S, T, dtype=torch.bool); keep[:, -1] = False
keep = keep.flatten()
for k in VARIANTS:
    lp = logp[k][keep]; lr = logp["ref"][keep]
    nll = -lp.gather(1, tgt[:, None]).mean().item()
    kl = (lr.exp() * (lr - lp)).sum(-1)
    res[k] = {"ppl": float(np.exp(nll)), "kl_mean": kl.mean().item(), "kl_p99": kl.quantile(0.99).item(),
              "top1_agree": (lp.argmax(-1) == lr.argmax(-1)).float().mean().item()}
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
