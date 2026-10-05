## Project title

Robust Reasoning RL Under Imperfect Rewards

## One-line description

A reproducible GRPO research framework that measures reasoning correctness separately from
corrupted verifier rewards.

## Resume bullet

Built and validated standard and uncertainty-aware GRPO with TRL/LoRA, completing 27 real
pretrained GSM8K pilot conditions across three seeds with independent verification and pass@k;
exposed reward inflation without correctness gains and qualified a stronger Qwen model while
preserving the null result and reproducible experiment evidence.

## Technical skills demonstrated

Python, PyTorch, Transformers, TRL, PEFT/LoRA, Hugging Face Datasets, GRPO, verifier design,
seeded corruption, statistical aggregation, Pandas, Matplotlib, pytest, Ruff, mypy, and GitHub CI.

## Interview explanation

Imperfect rewards can encourage optimization that does not improve task correctness. I built
matched standard and uncertainty-aware GRPO paths and judged held-out answers independently.
The hardest engineering challenge was separating real optimizer movement, parsing errors,
sampling effects, and corrupted reward observations while retaining auditable raw evidence.
The small CPU pilot found no robust-method advantage: correctness stayed at zero while reward
rose under corruption. I diagnosed truncation and model weakness, corrected real final-answer
formats, and qualified Qwen on separate validation data. The larger frozen comparison is
blocked by unavailable CUDA, so I make no benchmark or superiority claim. The main lesson was
that reward growth and a completed training loop are insufficient evidence of capability.
