# OptiBri plan: GLM-5.2 on a 4 GB-RAM laptop

## Target machine
- Windows 11 laptop, AMD Ryzen 7 7730U (Zen 3, 8C/16T, AVX2+FMA, no AVX-512, 16 MB L3)
- Integrated Radeon (shares system RAM, no extra memory)
- 16 GB RAM, but only ~3.5–4 GB usable
- 221 GB free disk on a WD PC SN560 1 TB NVMe (DRAM-less, also holds the page file)
  - Sequential read: ~3.2 GB/s at QD1, ~4.9 GB/s at QD32 (Notebookcheck averages)
  - Random 4K read: ~56 MB/s at QD1, ~650 MB/s at QD32
  - So: reads must be large (MB-sized) and many in flight at once. Small random reads are about 60× slower.

## Hard requirements
- Quality ≥ 97% of the original GLM-5.2, measured against the real bf16 weights.
- No training of new models (calibration on existing weights is fine).
- All evidence is produced in the cloud container; the user cannot run tests.

## The model, from the real config (zai-org/GLM-5.2)
- `GlmMoeDsaForCausalLM`, bf16, 1.507 TB on Hugging Face, 282 shards
- 78 layers plus 1 MTP layer (layer 78). Layers 0–2 are dense MLP, layers 3–77 are MoE.
- Hidden size 6144, vocabulary 154,880, untied embedding and lm_head
- MLA attention: q_lora 2048, kv_lora 512, 64 heads, qk 256 (192 nope + 64 rope), v 256
- DSA sparse attention indexer: top-2048 keys. Prompts up to 2048 tokens behave exactly like full attention.
- MoE: 256 routed experts, 8 active, plus 1 shared expert. Each expert is 3 × 6144×2048 = 37.75M params. Sigmoid router, noaux_tc, routed_scaling_factor 2.5.

### Parameter budget (computed in tools; see PROGRESS.md)
| Part | Params |
|---|---|
| Total (incl. MTP) | 753.7B |
| Active per token | 40.7B |
| Always-on per token (attention 13.6B, dense, shared experts, lm_head) | 18.1B |
| Routed experts per token | 22.6B |
| Attention per layer | 174.4M (o_proj alone is 100.7M) |

### What this means
| Avg bits/weight | Disk | Read per token if nothing is cached | Time at 3.2 GB/s |
|---|---|---|---|
| 2.0 | 188 GB | 10.2 GB | 3.2 s |
| 2.2 | 207 GB | 11.2 GB | 3.5 s |
| 2.5 | 236 GB | 12.7 GB | 4.0 s |
| 4.0 | 377 GB | 20.4 GB | 6.4 s |

- **Disk forces an average of ≤ ~2.2 bits/weight**, unless the model's representation changes, for example by sharing structure across experts.
- **Always-on weights are 45% of per-token reads.** Most of that is attention, not experts. Experts aren't the only problem.
- **RAM (~4 GB) can't even hold the always-on part** (18.1B params ≈ 5 GB at 2.2 bpw). Everything streams. So the winning move is to make each byte read from disk serve many tokens.

## Strategy (each lever gated by measured quality)
1. **Evaluation harness first.** Layer-by-layer offline forward pass of the real model in the container. It streams each layer's weights by HTTP range requests (122 MB/s measured), runs many calibration tokens, and keeps hidden states (not weights). Several compressed variants run side by side in the same pass, so one download serves many experiments. Metrics are KL divergence and top-1 agreement versus the original, plus logit-scored multiple-choice items.
2. **Representation change for experts.** Find shared structure (shared subspaces, dictionaries, cross-expert codebooks) so the disk stores far fewer unique bytes. See IDEAS.md.
3. **Extreme but careful quantization.** Trellis/codebook quantization at ~2 bits where the measured sensitivity allows it, with more bits where it doesn't.
4. **Amortize every read across many tokens.** Prompt processing is already batched. For generation, the MTP head and speculative token trees are verified in one pass, so one read of attention, shared and expert weights serves k tokens.
5. **I/O engine built for this SSD.** Large aligned reads, many in flight (Windows overlapped I/O / IoRing), unbuffered reads so the page file and page cache don't fight us, and decompression on the idle CPU cores.
6. **Engine.** Start from Colibri (C, Apache 2.0) and port it to Windows with AVX2 kernels for our formats.

## Speed physics (from exp5)
- With ~4 GB of RAM, almost nothing stays resident. Every token reads its 8 routed experts per layer plus the always-on weights from the SSD.
- Neighbouring tokens share few experts. Verifying k drafted tokens per pass mainly amortizes the **always-on** 18.1B, not the experts.
- So time/token ≈ [routed 22.6B × bpw_e / (tokens per expert read ≈ 1.1)] + [always-on 18.1B × bpw_a / accepted tokens per pass], divided by SSD bandwidth (~4.9 GB/s with many large reads in flight).
- Example: bpw_e = 2.3 and bpw_a = 3 with 3 accepted tokens per pass gives ≈ 5.9 + 2.3 = 8.2 GB/token, about **1.7 s/token**. Using top-4 experts would cut the first term roughly in half, to ≈ 1.1 s/token, *if* the quality holds (being measured).
