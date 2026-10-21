"""Reference GLM-5.2 forward pass that streams weights layer by layer from Hugging Face.

Follows transformers 5.12 `modeling_glm_moe_dsa.py`. For sequences <= index_topk (2048) tokens the DSA
indexer selects every causal key, so plain causal attention is exact and the indexer is skipped.

Weights are fetched per tensor via HTTP range requests (tools/hf_tensors.py), converted to float32,
used, and dropped, so peak memory is about one attention block or one expert at a time.
"""
import json, os, threading, queue
import numpy as np, torch
import hf_tensors as H

torch.set_grad_enabled(False)
CFG = json.load(open(os.environ.get("GLM_CFG", os.path.join(os.path.dirname(__file__), "glm52_config.json"))))
D = CFG["hidden_size"]; NH = CFG["num_attention_heads"]; NOPE = CFG["qk_nope_head_dim"]
ROPE = CFG["qk_rope_head_dim"]; VD = CFG["v_head_dim"]; KVR = CFG["kv_lora_rank"]
EPS = CFG["rms_norm_eps"]; E = CFG["n_routed_experts"]; TOPK = CFG["num_experts_per_tok"]
SCALE = (NOPE + ROPE) ** -0.5

_wm = H.weight_map() if "GLM_CFG" not in os.environ else None

LOCAL = None   # optional {name: tensor} source, used by the self-test against transformers

def W(name):
    if LOCAL is not None:
        return LOCAL[name].float()
    return torch.from_numpy(H.fetch(name, _wm, cache=False))

def rms(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))

def rope_cs(T):
    inv = 1.0 / (CFG["rope_parameters"]["rope_theta"] ** (torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE))
    f = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    return f.cos().float(), f.sin().float()          # T, ROPE/2

def rot(x, cos, sin):                                 # interleaved pairs, output [even', odd'] like HF
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

def attention(L, x, cs):
    p = f"model.layers.{L}.self_attn."
    T = x.shape[0]; cos, sin = cs
    q = rms(x @ W(p + "q_a_proj.weight").T, W(p + "q_a_layernorm.weight")) @ W(p + "q_b_proj.weight").T
    q = q.view(T, NH, NOPE + ROPE).transpose(0, 1)                        # NH,T,256
    qn, qr = q.split([NOPE, ROPE], -1)
    ckv = x @ W(p + "kv_a_proj_with_mqa.weight").T
    kp, kr = ckv.split([KVR, ROPE], -1)
    kv = (rms(kp, W(p + "kv_a_layernorm.weight")) @ W(p + "kv_b_proj.weight").T).view(T, NH, NOPE + VD).transpose(0, 1)
    kn, v = kv.split([NOPE, VD], -1)
    qr = rot(qr, cos, sin); kr = rot(kr, cos, sin).unsqueeze(0).expand(NH, T, ROPE)
    qq = torch.cat([qn, qr], -1); kk = torch.cat([kn, kr], -1)
    o = torch.nn.functional.scaled_dot_product_attention(qq.unsqueeze(0), kk.unsqueeze(0), v.unsqueeze(0),
                                                         is_causal=True, scale=SCALE)[0]
    return o.transpose(0, 1).reshape(T, NH * VD) @ W(p + "o_proj.weight").T

def mlp(prefix, x):
    g = x @ W(prefix + "gate_proj.weight").T; u = x @ W(prefix + "up_proj.weight").T
    return (torch.nn.functional.silu(g) * u) @ W(prefix + "down_proj.weight").T

def route(L, x):
    p = f"model.layers.{L}.mlp.gate."
    s = (x @ W(p + "weight").T).sigmoid()
    idx = (s + W(p + "e_score_correction_bias")).topk(TOPK, -1).indices
    w = s.gather(1, idx); w = w / (w.sum(-1, keepdim=True) + 1e-20) * CFG["routed_scaling_factor"]
    return idx, w

class ExpertStream:
    """Fetch expert weights with several parallel downloads (bounded lookahead) so download overlaps compute."""
    def __init__(self, L, experts, ahead=24):
        import concurrent.futures as cf, collections
        self.L, self.experts, self.ahead = L, list(experts), ahead
        self.ex = cf.ThreadPoolExecutor(ahead); self.pend = collections.deque()
    def _get(self, e):
        p = f"model.layers.{self.L}.mlp.experts.{e}."
        return [W(p + n + "_proj.weight") for n in ("gate", "up", "down")]
    def __iter__(self):
        it = iter(self.experts)
        for e in it:
            self.pend.append((e, self.ex.submit(self._get, e)))
            if len(self.pend) >= self.ahead: break
        while self.pend:
            e, f = self.pend.popleft()
            nxt = next(it, None)
            if nxt is not None: self.pend.append((nxt, self.ex.submit(self._get, nxt)))
            yield e, f.result()
        self.ex.shutdown()

def moe(L, x, hook=None):
    """Routed experts + shared expert. `hook(L, e, xe, (g,u,d))` sees each expert's inputs and weights."""
    idx, w = route(L, x)
    out = mlp(f"model.layers.{L}.mlp.shared_experts.", x)
    used = sorted(set(idx.flatten().tolist()))
    for e, (g, u, d) in ExpertStream(L, used):
        tok, slot = (idx == e).nonzero(as_tuple=True)
        xe = x[tok]
        if hook: hook(L, e, xe, (g, u, d))
        ye = (torch.nn.functional.silu(xe @ g.T) * (xe @ u.T)) @ d.T
        out.index_add_(0, tok, ye * w[tok, slot, None])
    return out, idx

def layer(L, h, cs, hook=None):
    p = f"model.layers.{L}."
    h = h + attention(L, rms(h, W(p + "input_layernorm.weight")), cs)
    x = rms(h, W(p + "post_attention_layernorm.weight"))
    if CFG["mlp_layer_types"][L] == "dense":
        return h + mlp(p + "mlp.", x), x, None
    y, idx = moe(L, x, hook)
    return h + y, x, idx

def embed(ids):
    """Gather only the embedding rows for the tokens in use."""
    if LOCAL is not None:
        return LOCAL["model.embed_tokens.weight"].float()[ids]
    rows = H.fetch_rows("model.embed_tokens.weight", sorted(set(ids)))
    return torch.from_numpy(np.stack([rows[i] for i in ids]))

def head(h):
    return rms(h, W("model.norm.weight")) @ W("lm_head.weight").T
