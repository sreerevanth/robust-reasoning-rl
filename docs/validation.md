# Final validation — 5 October 2026

This is the current validation record. Historical records remain in
[Phase 1](validation-phase1.md) and [Phase 2](phase2_validation.md).

## Environment

Windows, Python 3.11.9, PyTorch 2.14.1+cpu, CUDA unavailable, CUDA version null.
The host has eight physical cores, sixteen logical processors, and approximately 16 GB RAM;
only roughly 3 GB was available during the final pass. Integrated AMD graphics are not a
CUDA device. See [current hardware](../experiments/final/hardware.json).

## Executed checks

| Check | Result |
| --- | --- |
| Full training environment, `python scripts/check.py` | 120 tests passed |
| Ruff lint | Passed |
| Ruff formatting | 49 files formatted |
| mypy | Passed, 30 source files |
| `python -m pip check` | No broken requirements |
| Fresh `python -m venv` and `pip install -e '.[dev]'` | Installed successfully |
| Fresh core environment quality gates | 118 passed, 2 optional training tests skipped |
| Fresh core dependency check | No broken requirements |
| Evaluation CLI | Completed on fixture data |
| Experiment CLI | 15 explicitly labelled fixture conditions, no failures |
| Plot CLI | 13 nonempty artifacts per campaign |
| Training CLI | Real local tiny-model optimizer/save/reload integration passed |
| Real pretrained Qwen experiment CLI | Three smoke conditions completed, no failures |
| Aggregation and failure analysis | Generated from actual stored summaries and records |
| Final CUDA preflight | Externally blocked; no final-study condition launched |

The full suite displays one expected PEFT Conv1D layout warning. The standard local training
path runs in a subprocess and captures its corresponding warning. These are local random-model
integration tests, not pretrained-model performance evidence. Fresh installation was tested in
a new environment against this checkout, rather than in a separately cloned checkout.

## Measured pretrained evidence

The preserved SmolLM2 CPU pilot completed all 27 planned conditions across three seeds,
with zero failed conditions. All pass@1 and pass@4 values were zero. The original pretrained
SmolLM2 validation smoke completed three conditions. The final pass ran two candidate-model
qualifications on sixteen validation questions each, preserving original and rejudged records.
Qwen then completed three CPU smoke conditions with two real one-update GRPO runs.

The Qwen training checkpoints contain 4,399,104 trainable parameters. Both standard and robust
runs changed 168 trainable tensors according to before/after SHA256, saved adapters, and
independently reloaded them for generation. A small smoke success is not a method advantage.
The CPU train loops are resource-bounded instrumentation checks, not the full study budget.

No final 128-question corruption comparison ran. All 45 conditions in the frozen CUDA protocol
remain unexecuted. No failed condition, missing seed, or unexecuted result is represented as success.

## Evidence paths

- [CPU pilot report](../experiments/phase2/artifacts/cpu-pilot/report.md)
- [SmolLM2 pretrained smoke](../experiments/phase2/artifacts/pretrained-smoke/report.md)
- [Candidate qualification table](../experiments/final/qualification/report.md)
- [Original and audited Qwen outputs](../experiments/final/artifacts/qualification-qwen/)
- [Qwen CPU smoke report](../experiments/final/artifacts/qwen-cpu-smoke/report.md)
- [Frozen protocol](../experiments/final/protocol.yaml)
- [Final research report](final-report.md)

Archive-integrity tests verify recorded byte counts/SHA256 and reject weight extensions.
Dataset-integrity tests verify SHA256, scalar references, train/test/validation separation,
and exclusion of the previously inspected pilot test pool. Sweep-resume tests verify skipped
completed conditions, rejection of changed configs/evidence, and preservation of retried failures.
Atomic-write tests cover transient and persistent Windows file locks without losing old results.
No weights, caches, virtual environments, credentials, or result ZIPs are intentionally tracked.

## GitHub CI and reproduction

The remote CI run for commit `290fdff` completed successfully:
[verified Actions run](https://github.com/sreerevanth/robust-reasoning-rl/actions/runs/37266019236).
This statement refers to that executed run, not a promise that a later commit has passed remotely.
Remote core CI subsequently exposed Windows separators in the new final dataset manifest.
The preparation script now writes portable POSIX paths, with a regression assertion; regeneration
preserves the exact dataset bytes and hashes. This was a portability failure, not a result change.
All three remote jobs (core Python 3.11, core Python 3.12, and training API) passed
on commit `2c3d629`, as confirmed by the
[final code Actions run](https://github.com/sreerevanth/robust-reasoning-rl/actions/runs/37277697081).

```bash
python scripts/check.py
python -m src.cli evaluate --config configs/evaluation.yaml
python -m src.cli experiment --config configs/smoke.yaml
python -m src.cli plot --results results/fixture-sweep
```

For the final pretrained study on a configured CUDA host:

```bash
python scripts/run_final_study.py --resume
```

See [GPU setup and resume behavior](gpu-execution.md). GPU installation, memory fit, and the
full final experiment are externally unvalidated. Classification: **C — Framework + pilot
complete, final GPU study externally blocked**.
