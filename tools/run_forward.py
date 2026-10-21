"""Resumable layer-by-layer forward pass of real GLM-5.2 over the calibration set.
Saves: hidden state after each layer (only the latest kept), and for layers in SAVE the MoE input x
and routing indices, to OUT. Usage: python run_forward.py OUT last_layer"""
import os, sys, time, numpy as np, torch
import glm_ref as G
OUT = sys.argv[1]; LAST = int(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
SAVE = {3, 4, 10, 20, 30, 40, 50, 60, 70, 77}
torch.set_num_threads(os.cpu_count())
ids = np.load(os.path.join(os.environ["GLM_CACHE"], "calib_8x1024.npy")); S, T = ids.shape
cs = G.rope_cs(T)
start = 0; hp = os.path.join(OUT, "h.npy")
if os.path.exists(hp):
    start = int(open(os.path.join(OUT, "h.layer")).read()) + 1; h = torch.from_numpy(np.load(hp))
else:
    h = G.embed(ids.flatten().tolist()).view(S, T, -1)
for L in range(start, LAST + 1):
    t0 = time.time(); p = f"model.layers.{L}."
    wn = G.W(p + "input_layernorm.weight")
    h = h + torch.stack([G.attention(L, G.rms(h[s], wn), cs) for s in range(S)])
    x = G.rms(h, G.W(p + "post_attention_layernorm.weight")).view(S * T, -1)
    if G.CFG["mlp_layer_types"][L] == "dense":
        y = G.mlp(p + "mlp.", x); idx = None
    else:
        y, idx = G.moe(L, x)
    h = h + y.view(S, T, -1)
    if L in SAVE:
        np.save(os.path.join(OUT, f"x{L}.npy"), x.numpy().astype(np.float16))
        if idx is not None: np.save(os.path.join(OUT, f"idx{L}.npy"), idx.numpy().astype(np.int16))
    np.save(hp + ".tmp.npy", h.numpy()); os.replace(hp + ".tmp.npy", hp)
    open(os.path.join(OUT, "h.layer"), "w").write(str(L))
    print(f"layer {L} done in {time.time()-t0:.0f}s |h| {h.norm(dim=-1).mean():.1f}", flush=True)
