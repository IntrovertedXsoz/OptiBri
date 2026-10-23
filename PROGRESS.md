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

5. **Routing locality** (exp5, layers 3/4/10): 4 consecutive tokens need 82–93% of 32 distinct experts, so expert reads barely amortize across drafted tokens. A RAM cache of the last 64 tokens' experts would hit 87%, but that is nearly every expert (tens of GB). → With ~4 GB of RAM, the speed limit is bytes/token. See PLAN.md "Speed physics".

### Running at session end (left running in the background; it uses no Claude usage)
- `run_e2e.py` on 2 × 512 tokens (Wikipedia + code). Variants: ref, hrtn3 (3.0 bpw experts), hrtn2 (2.0 bpw experts), top4 (bf16 experts, only 4 of 8 routed). ~255 s/layer, reached layer 11 by 01:09; needs ~4.5 h more.
  Log: `scratchpad/glmcache/e2e.log`, state `scratchpad/glmcache/e2e/state.pt`, final numbers in `e2e/result.json`.
- Observations so far (per-layer, not final):
  - Layer 8: the quantized variants' drift doubled in one layer (hrtn3 0.069 → 0.130, hrtn2 0.149 → 0.284) while top4 didn't. Suspect a quantization-sensitive "super weight" expert in layer 8. **Check it:** find the expert/row with the largest output error there; try keeping a few weights in bf16.
  - From layer 9 the hidden norm grows sharply (massive activations), so relative hidden drift stops being informative. Judge only by the final KL / top-1 / perplexity.
  - The `route-agree` metric compares sorted index lists position by position and **understates** the true overlap. Replace it with set overlap.

### Notes for the next session
- The container is ephemeral. If `scratchpad/glmcache` is gone, rerun `calib_data.py` (seconds). `run_e2e.py` must restart from layer 0 (~5 h for 78 layers with 4 variants).
- Avoid `pkill -f <pattern>` inside a Bash call: it matches and kills the calling shell. Use `pgrep -af "^python3 ..."`.
- The repo has no `main` branch yet, so no PR could be opened (a PR needs a base branch).

## Next goal (session 1, Wed Oct 21, 18:02 ET)
First: if the container survived, read `e2e/result.json` (or let `run_e2e.py` resume). If it was reset, restart `run_e2e.py` immediately (~5 h; it will span sessions).
Finish the end-to-end curve: get KL / top-1 / perplexity for 2- and 3-bit experts. This sets the per-expert error budget that "97% quality" allows. Then test the strongest ~2.2 bpw quantizer that fits the 221 GB disk (activation-aware rounding plus higher-dimensional codebooks/trellis) end to end.

## Session 1 — Wed Oct 21, 18:03 ET
**Goal:** get the end-to-end numbers (KL / top-1 / perplexity vs the original) for 3-bit experts, 2-bit experts and top-4 routing. This is the first measurement that ties compression to final quality.
- 18:03: the container survived, but the overnight run had stopped after layer 12 (background processes don't survive the session going idle). Resumed from saved state.
- Download speed today: 226 MB/s with 16 parallel fetches, 304 MB/s with 32, 221 MB/s with 64. Expert prefetch raised from 8 to 24 in flight.
- Speedups: bf16 matmuls on the cloud CPU's AMX unit; the profile showed 96% of time in my quantizer, so I moved the rotation to bf16 and made rounding in-place (same error, 0.3443 vs 0.3442). 255 s → ~110 s per layer.
- Hit Hugging Face HTTP 429 at layer 68; added backoff honouring Retry-After. The final step ran out of memory (4 variants × full logits ≈ 14 GB); metrics are now computed in 128-token chunks.

### Result: first end-to-end quality numbers (`results_e2e_v1.json`, 1,022 tokens: Wikipedia + Python)
Only the routed experts are changed; everything else is the original bf16.
| Variant | Perplexity | Δ vs original | KL mean | KL p99 | Top-1 agreement |
|---|---|---|---|---|---|
| original | 1.788 | — | 0 | 0 | 100% |
| hrtn3: experts 3.0 bpw (Hadamard + optimal RTN, one scale per row) | 1.982 | +10.8% | 0.142 | 1.98 | 91.7% |
| hrtn2: experts 2.0 bpw | 2.756 | +54% | 0.555 | 5.01 | 76.2% |
| top4: bf16 experts, 4 of 8 routed | 2.189 | +22% | 0.276 | 3.62 | 86.5% |

**What it means (honest):**
- None of these is near the 97% target. Good 4-bit-class quantization of large models usually lands around KL 0.01–0.03, so this simple quantizer is far from good enough even at 3 bits.
- Halving the experts per token costs about as much as dropping experts to ~2.5 bits. "Fewer experts" isn't a free speed lever. It might still work *adaptively* (drop only low-weight experts).
- Per-layer drift was very non-uniform (layer 8 doubled the error in one step). Most of the damage may come from a few sensitive layers, experts or rows. That's the most promising lead: protecting a small part of the model costs few bytes.
- The quantizer here is the crudest reasonable one: one scale per 6144-wide row, no activation awareness. Group scales, activation-aware rounding (GPTQ/LDLQ) and codebooks/trellis each give known, large error reductions. They haven't been applied end to end yet.

## Next goal (session 2, Thu Oct 22, 18:02 ET)
**Find where the damage comes from.** Run one end-to-end pass with variants that each quantize (3-bit) only one slice of layers: 3–20, 21–40, 41–60, 61–77. Log the per-expert output error per layer to find the worst experts and rows. If a few layers/experts dominate, test keeping them at higher precision. Use 2 × 512 tokens as before. Run time ~2 h at the current ~110 s/layer.

## Session 2 — Thu Oct 22, 18:03 ET
**Goal:** find where the 3-bit damage comes from. One end-to-end pass with 3-bit experts applied to one layer slice per variant (3–20, 21–40, 41–60, 61–77), plus a per-layer/per-expert local damage log (`local_err.json`).
- Run took ~100 s/layer. The final step hit out-of-memory again (5 variants after a long run); rerunning in a fresh process from the saved state completed it.

### Result: where the 3-bit damage comes from (`results_e2e_slices.json`, `results_local_err_3bit.json`)
3-bit experts (same quantizer as v1) applied to one layer slice at a time:
| Slice quantized | Perplexity | KL mean | KL p99 | Top-1 agreement |
|---|---|---|---|---|
| none (ref) | 1.785 | 0 | 0 | 100% |
| layers 3–20 | 1.782 | 0.024 | 0.25 | 97.0% |
| layers 21–40 | 1.790 | 0.049 | 0.64 | 96.1% |
| layers 41–60 | 1.822 | 0.063 | 0.94 | 94.0% |
| layers 61–77 | 1.820 | 0.039 | 0.62 | 95.1% |
- Slice KLs add up roughly to the all-layer KL (0.175 vs 0.142). **The damage is spread across all depths** (middle layers worst), not concentrated in a few layers.
- Within layers (local 3-bit error of each expert on the original model's own inputs): layer local error 0.15–0.32 (median 0.27). **The worst 5% of experts carry 34% of a layer's damage, the worst 10% carry 47%.**
- **Super-weight experts confirmed:** a single expert carries 53% of the damage in layer 8, 35% in layer 9 and 55% in layer 77. This explains the layer-8 jump seen in v1.
- So protecting the worst ~10% of experts per layer could remove about half the damage for little disk. But every layer is damaged, so the base quantizer must also improve (v1 uses one scale per 6144-wide row and no activation awareness).
- Caution: the worst experts were identified on the same tokens that are evaluated. Re-select them on *different* text before trusting a protection result.

## Next goal (session 3, Fri Oct 23, 18:02 ET)
**Combine a better base quantizer with expert protection and measure it end to end.**
1. Pick the protected experts from *different* calibration text (calib seqs 0 and 6, not the eval seqs 3 and 7): a cheap ref-only pass logging local damage, or reuse the local log machinery.
2. End-to-end variants (all experts compressed): (a) 3-bit with group scales (g=128) after rotation; (b) (a) + worst 10% experts per layer kept at 4-bit; (c) a ~2.3 bpw mix: 2-bit base with group scales + worst ~15% at 4-bit. Report bits/weight including scales.
