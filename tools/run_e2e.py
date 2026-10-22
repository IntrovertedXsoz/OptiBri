"""End-to-end quality of compressed GLM-5.2 variants vs the original, in one streamed pass.

Every variant keeps its own hidden-state stream and its own routing (quantization can change which experts
are picked, a real effect). Each weight is downloaded once per layer and shared by all streams.
Only the routed experts are compressed here; everything else stays bf16, to isolate the expert budget.
At the end: KL(ref || variant), top-1 agreement and perplexity on the real next tokens.
A per-layer "local damage" log (local_err.json) records, on the ref stream's own inputs, the relative error each
3-bit-quantized expert adds, so the worst layers/experts can be found directly.
Resumable. Usage: E2E_SPEC=v1|slices python run_e2e.py OUT
"""
import os, sys, json, time, numpy as np, torch
import glm_ref as G
from quant import hrtn_rot, rotate            # rotation + MSE-optimal uniform quantizer (tested in exp4)
torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
# variants return weights in the rotated basis; activations are rotated instead of unrotating weights
# Each variant: bits = expert quantization (None = bf16), layers = where it applies (None = all), topk = routed experts kept.
SPECS = {
    "v1": {"ref": {}, "hrtn3": {"bits": 3}, "hrtn2": {"bits": 2}, "top4": {"topk": 4}},
    "slices": {"ref": {}, **{f"q3_L{a:02d}_{b:02d}": {"bits": 3, "layers": range(a, b + 1)}
                             for a, b in ((3, 20), (21, 40), (41, 60), (61, 77))}},
}
VARIANTS = SPECS[os.environ.get("E2E_SPEC", "v1")]
LOCAL_BITS = 3                       # bits used for the local-damage log
def qw(w3, b): return [hrtn_rot(w, b).to(w.dtype) for w in w3]
def active(k, L): s = VARIANTS[k]; return s.get("bits") and (s.get("layers") is None or L in s["layers"])
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
        n = VARIANTS[k].get("topk")
        if n:                                            # route() returns experts sorted by (biased) score
            s = wts.sum(-1, keepdim=True); idx, wts = idx[:, :n], wts[:, :n]
            wts = wts / wts.sum(-1, keepdim=True) * s
        routes[k] = (idx, wts); out[k] = G.mlp(f"model.layers.{L}.mlp.shared_experts.", x)
    used = sorted(set().union(*[set(r[0].flatten().tolist()) for r in routes.values()]))
    loc = {"num": torch.zeros(256), "den": torch.zeros(256)}
    dref = torch.zeros_like(out["ref"]); yref = torch.zeros_like(out["ref"])
    def run(w, x, rot):
        if rot: x = rotate(x)
        a = torch.nn.functional.silu(G.mm(x, w[0])) * G.mm(x, w[1])
        if rot: a = rotate(a)
        return G.mm(a, w[2])
    for e, w3 in G.ExpertStream(L, used):
        qc = {}                                          # quantized weights, shared by every variant with the same bits
        def get(b):
            if b not in qc: qc[b] = qw(w3, b)
            return qc[b]
        for k, x in xs.items():
            idx, wts = routes[k]; tok, slot = (idx == e).nonzero(as_tuple=True)
            if len(tok) == 0: continue
            b = VARIANTS[k].get("bits") if active(k, L) else None
            y = run(get(b), x[tok], True) if b else run(w3, x[tok], False)
            out[k].index_add_(0, tok, y * wts[tok, slot, None])
            if k == "ref":                               # local damage of a LOCAL_BITS expert on the ref inputs
                yq = run(get(LOCAL_BITS), x[tok], True)
                loc["num"][e] = (yq - y).pow(2).sum(); loc["den"][e] = y.pow(2).sum()
                dref.index_add_(0, tok, (yq - y) * wts[tok, slot, None]); yref.index_add_(0, tok, y * wts[tok, slot, None])
    local = {"layer_rel": (dref.norm() / yref.norm()).item(),
             "expert_rel": (loc["num"] / loc["den"].clamp_min(1e-30)).sqrt().tolist(),
             "expert_share": (loc["num"] / loc["num"].sum()).tolist()}
    def overlap(a, b):   # fraction of b's experts (per token) also chosen in a
        return (a[:, :, None] == b[:, None, :]).any(-1).float().mean().item()
    agree = {k: overlap(routes[k][0], routes["ref"][0]) for k in xs if not VARIANTS[k].get("topk")}
    return out, agree, local

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
        ys, agree, local = moe_multi(L, xs)
        lp = os.path.join(OUT, "local_err.json"); le = json.load(open(lp)) if os.path.exists(lp) else {}
        le[str(L)] = local; json.dump(le, open(lp, "w"))
        e = torch.tensor(local["expert_rel"]); top = torch.tensor(local["expert_share"]).topk(3)
        print(f"  local L{L}: layer {local['layer_rel']:.3f}, expert median {e[e > 0].median():.3f} max {e.max():.3f}, "
              f"top damage shares " + " ".join(f"e{i}={v:.0%}" for v, i in zip(top.values.tolist(), top.indices.tolist())), flush=True)
    for k in VARIANTS: h[k] = h[k] + ys[k].view(S, T, -1)
    torch.save({"h": h, "layer": L}, state + ".tmp"); os.replace(state + ".tmp", state)
    dev = {k: ((h[k] - h["ref"]).norm() / h["ref"].norm()).item() for k in VARIANTS if k != "ref"}
    print(f"layer {L} {time.time()-t0:.0f}s hidden-dev " + " ".join(f"{k}={v:.3f}" for k, v in dev.items())
          + " route-agree " + " ".join(f"{k}={v:.3f}" for k, v in agree.items() if k != "ref"), flush=True)

memo.clear(); G.memo = None
# Final metrics, computed in token chunks so the 154,880-wide logits never all sit in memory at once.
wn, wh = W("model.norm.weight"), W("lm_head.weight")
tgt = torch.from_numpy(ids[:, 1:].reshape(-1))
hs = {k: h[k][:, :-1].reshape(-1, h[k].shape[-1]) for k in VARIANTS}      # positions that have a next token
acc = {k: {"nll": 0.0, "kl": [], "top1": 0} for k in VARIANTS}
for i in range(0, len(tgt), 128):
    lr = None
    for k in VARIANTS:
        lp = torch.log_softmax(G.mm(G.rms(hs[k][i:i + 128], wn), wh), -1)
        if k == "ref": lr = lp
        acc[k]["nll"] += -lp.gather(1, tgt[i:i + 128, None]).sum().item()
        acc[k]["kl"].append((lr.exp() * (lr - lp)).sum(-1))
        acc[k]["top1"] += (lp.argmax(-1) == lr.argmax(-1)).sum().item()
n = len(tgt); res = {}
for k, a in acc.items():
    kl = torch.cat(a["kl"])
    res[k] = {"ppl": float(np.exp(a["nll"] / n)), "kl_mean": kl.mean().item(), "kl_p99": kl.quantile(0.99).item(),
              "top1_agree": a["top1"] / n}
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
