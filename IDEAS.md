# IDEAS: ways to change how GLM-5.2 runs

Status values: untested, testing, measured (keep), measured (drop), parked. Every idea is judged by measured output error on real GLM-5.2 activations, never by intuition.

| # | Idea | Why it could matter | Status / evidence |
|---|---|---|---|
| 1 | Shared base + per-expert delta (raw weights) | Store the common part once, so the disk holds only small deltas | **measured (drop)**. Layer 40: experts are near-orthogonal (cos ≈ 0.001); the shared mean explains 3.1%, which is exactly chance for 32 random experts. Per-expert rank is near full (90% energy needs 1458/2048). |
| 2 | Shared *activation* subspace per layer | All 256 experts read the same input stream. If inputs occupy a k-dim subspace, gate/up shrink from 2048×6144 to 2048×k, with one 6144×k projection kept in RAM. | **measured (drop, parked)**. In-sample (exp2, layer 3): k=2048 gives 0.034 expert-output error vs 0.094 for 4-bit RTN. **But held-out (exp3) shows it was overfitting:** fitted on 3,072 tokens, error on unseen Wikipedia is 0.228 and on unseen code 0.353 at k=2048, still 0.175/0.256 at k=3072. The output-side subspace fails the same way. Could only be revived with ≥10× more calibration tokens showing a plateau. |
| 3 | Dictionary of atoms across experts (product/vector quantization with a codebook shared by a whole layer, held in RAM) | Disk stores only indices plus corrections | untested |
| 4 | Trade idle CPU for I/O: entropy-coded weights (ANS) decoded on the fly | Quantized weights aren't uniform; about 10–20% fewer bytes for free | untested |
| 5 | Weights regenerated from PRNG seeds + residuals | Extreme compression ratio if any structure fits | untested (low prior given #1) |
| 6 | Conversation-level expert pre-pass: predict the experts a reply needs, then lay them out contiguously | Turns random reads into sequential ones (SSD: 56 MB/s random vs 3.2 GB/s sequential) | untested |
| 7 | Speculative token trees (MTP head) verified in one pass | One weight read serves k tokens. This attacks the always-on 18.1B too, not only experts. | untested |
| 8 | Confidence-based expert/layer skipping | Fewer bytes on easy tokens | untested |
| 9 | Training-free expert merging via calibration | Fewer unique experts | untested (low prior given #1) |
| 10 | Cache-aware routing substitution | Higher RAM hit rate | untested |
| 11 | Attention is 13.6B of the 18.1B always-on params (o_proj alone is 100.7M/layer) | Compress attention as hard as experts, or amortize it via batching | untested |
| 12 | Hidden-state delta streaming: process the prompt with all layers resident one layer at a time (layer-major instead of token-major) | Prompt processing reads each layer once for the whole prompt | untested; it's how the eval harness already runs |
