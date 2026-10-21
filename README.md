# OptiBri

Research toward running GLM-5.2 (753.7B-parameter MoE) on a Windows laptop with ~4 GB of usable RAM and 221 GB of free disk, at ≥ 97% of the original quality.

- `PLAN.md`: target machine, hard limits, budget derived from the real model config, strategy
- `IDEAS.md`: backlog of ideas for changing how the model runs, each with measured evidence
- `PROGRESS.md`: session log and the next goal
- `tools/`: measurement tooling
  - `hf_tensors.py`: fetches single tensors from Hugging Face by HTTP range request
  - `glm_ref.py`: reference GLM-5.2 forward pass; verified against transformers by `test_glm_ref.py`
  - `run_forward.py`: resumable layer-by-layer pass over the calibration set
  - `exp*.py`: experiments
