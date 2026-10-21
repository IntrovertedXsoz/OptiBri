# PROGRESS

## Session 0 — Tue Oct 20, 22:35 → Wed Oct 21, ~01:35 ET (one-off 3-hour session requested by the user)

### Built
- `tools/hf_tensors.py`: fetches any single GLM-5.2 tensor (or embedding rows) by HTTP range request. Measured 122 MB/s with 16 parallel fetches.
- `tools/glm_ref.py`: reference forward pass that streams weights layer by layer. **Verified** against transformers 5.12 `GlmMoeDsaForCausalLM` on a tiny random model (`tools/test_glm_ref.py`, max rel err 5.5e-5). For ≤2048-token sequences, DSA is exactly full causal attention.
- `tools/calib_data.py`: 8 × 1024-token sequences (6 wikitext-103, 2 Python source) tokenized with GLM-5.2's tokenizer.
- `tools/run_forward.py`: resumable layer-by-layer pass over all 8,192 tokens. Reached layer 12 (~200 s/layer on 4 cores). Saved MoE inputs/routes for layers 3, 4 and 10 (container scratch only).
- `tools/quant.py`: block-Hadamard rotation, MSE-optimal uniform RTN, layer-shared vector quantization.
- `tools/run_e2e.py`: **end-to-end** quality of compressed variants vs the original. Every variant keeps its own hidden-state stream and routing. Outputs KL, top-1 agreement and perplexity.

### Measured (all on real GLM-5.2 weights)
1. **Experts don't share raw structure** (exp1, layer 40). Pairwise cosine is ≈ 0.001, the shared mean explains chance-level energy, and per-expert rank is near full. → Shared base + delta is dropped.
2. **Shared activation subspace looked lossless but was overfitting** (exp2, exp3, layer 3). In-sample, k=2048 gives 0.034 error. Held-out it's 0.228 (Wikipedia) and 0.353 (code). → Dropped unless far more calibration data shows a plateau.
3. **Expert usage** (layer 3, 8,192 tokens): all 256 used, top-64 take 41% of routes. Usage is not very skewed at this depth.
4. **Low-bit frontier, expert output error** (exp4, layer 3, 8 experts):
   | bpw | method | mean err |
   |---|---|---|
   | 2.00 | Hadamard + 4-dim VQ, 256-entry layer codebook | 0.369 |
   | 2.00 | Hadamard + optimal RTN | 0.414 |
   | 2.50 | Hadamard + 4-dim VQ, 1024 entries | 0.253 |
   | 2.50 | plain RTN g64 | 0.506 |
   | 3.00 | Hadamard + optimal RTN | 0.210 |
   | 4.25 | plain RTN g128 | 0.106 |

### Running at session end
- `run_e2e.py` on 2 × 512 tokens with variants ref / hrtn3 (3.0 bpw experts) / hrtn2 (2.0 bpw experts). See the result section below if it finished.

### Notes for the next session
- The container is ephemeral. If `scratchpad/glmcache` is gone, rerun `calib_data.py` (seconds). `run_e2e.py` must restart from layer 0 (~2 h for 78 layers).
- Avoid `pkill -f <pattern>` inside a Bash call: it matches and kills the calling shell. Use `pgrep -af "^python3 ..."`.
- The repo has no `main` branch yet, so no PR could be opened (a PR needs a base branch).

## Next goal (session 1, Wed Oct 21, 18:02 ET)
Finish the end-to-end curve: get KL / top-1 / perplexity for 2- and 3-bit experts. This sets the per-expert error budget that "97% quality" allows. Then test the strongest ~2.2 bpw quantizer that fits the 221 GB disk (activation-aware rounding plus higher-dimensional codebooks/trellis) end to end.
