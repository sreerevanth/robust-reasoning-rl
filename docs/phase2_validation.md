# Phase 2 validation

Completed 5 October 2026 (Asia/Calcutta). This supplements the historical Phase 1 record.

## Quality gates

`python scripts/check.py` passed: Ruff lint, formatting (44 files), mypy (30 source files),
and 111 tests. Two expected PEFT warnings adjust `fan_in_fan_out` for the local test model's
Conv1D modules. `python -m pip check` reported no broken requirements.
Tests include real local GRPO/save/reload integration and SHA256 verification of the archived
evidence and deterministic dataset pools. These test fixtures are separate from model results.

## Executed evidence

- Real pretrained validation smoke: three conditions, no failures, two one-update LoRA runs
  with saved checkpoints independently reloaded for evaluation.
- Real GSM8K CPU pilot: all 27 conditions completed, no failures or missing planned conditions.
  Three seeds, three corruption levels, and base/standard/robust methods were evaluated.
- Eighteen two-update training runs saved checkpoints. Twelve noisy-reward runs have nonzero
  LoRA B norms; six clean-reward runs have zero gradients and unchanged adapters.
- Each evaluation has four held-out questions and four samples per question: 432 scored
  records from 336 new generations and 96 rejudged base responses. Replayed base outputs at additional corruption levels are flagged and are not
  additional independent generations. Training encountered two unique questions per run.
- Automatically generated per-seed tables, seed means/sample standard deviations, paired
  differences, training diagnostics, five requested plot types, and deterministic failure
  analysis are archived alongside the original records. Archive manifests record byte counts
  and SHA256 hashes. Model weights remain local and are excluded from Git.

See the [pilot report](../experiments/phase2/artifacts/cpu-pilot/report.md) and
[smoke report](../experiments/phase2/artifacts/pretrained-smoke/report.md).

## Interpretation and limits

All pilot conditions have zero pass@1 and pass@4. The study found no improvement in
single-attempt correctness, repeated sampling success, or held-out verifier performance.
The robust method did not reduce the observed reward gap. Injected corruption also inflates
the untrained base model's reward, so the gap alone cannot establish learned exploitation.
No significance, convergence, or benchmark-superiority claim is supported.

The hardware has no available CUDA device. The 10%/60% levels, pass@8, and larger CUDA
campaign were not executed. The checked-in GPU configuration and exact commands are in
[Phase 2 documentation](phase2.md). Shared answer parsing limits judge independence,
pretraining contamination is unknown, and the very small test set limits generalization.
