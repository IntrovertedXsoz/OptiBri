"""Self-test: glm_ref.py must match transformers' GlmMoeDsaForCausalLM on a tiny random model."""
import json, os, tempfile, torch
os.environ.setdefault("GLM_BF16", "0")   # exactness test runs the fp32 path
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "glm52_config.json")))
n = 5
cfg.update(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4,
           qk_head_dim=16, head_dim=12, v_head_dim=16, kv_lora_rank=8, q_lora_rank=16, num_hidden_layers=n,
           first_k_dense_replace=1, n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=8,
           intermediate_size=32, vocab_size=100, pad_token_id=0, eos_token_id=[1], index_head_dim=8, index_n_heads=2,
           mlp_layer_types=["dense"] + ["sparse"] * (n - 1), indexer_types=["full"] * n, num_nextn_predict_layers=0)
p = tempfile.mktemp(suffix=".json"); json.dump(cfg, open(p, "w")); os.environ["GLM_CFG"] = p
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
torch.manual_seed(0)
c = GlmMoeDsaConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "transformers_version")})
c._attn_implementation = "eager"
m = GlmMoeDsaForCausalLM(c).float().eval()
with torch.no_grad():
    for name, t in m.named_parameters():
        t.normal_(0, 0.2)
    for L in range(1, n):
        m.model.layers[L].mlp.gate.e_score_correction_bias.normal_(0, 0.1)
sd = {k: v.detach() for k, v in m.state_dict().items()}
for L in range(1, n):   # HF stores experts fused; expand to the per-expert names used in the checkpoint
    pre = f"model.layers.{L}.mlp.experts."
    gu = sd.pop(pre + "gate_up_proj"); dn = sd.pop(pre + "down_proj"); I = cfg["moe_intermediate_size"]
    for e in range(cfg["n_routed_experts"]):
        sd[pre + f"{e}.gate_proj.weight"] = gu[e, :I]; sd[pre + f"{e}.up_proj.weight"] = gu[e, I:]
        sd[pre + f"{e}.down_proj.weight"] = dn[e]
import glm_ref as G
G.LOCAL = sd
ids = torch.randint(0, 100, (1, 37))
with torch.no_grad():
    ref = m(ids).logits[0]
    h = G.embed(ids[0].tolist()); cs = G.rope_cs(37)
    for L in range(n):
        h, _, _ = G.layer(L, h, cs)
    out = G.head(h)
err = (out - ref).abs().max().item() / ref.abs().max().item()
print("max rel err", err); assert err < 1e-4, "MISMATCH"; print("PASS")
