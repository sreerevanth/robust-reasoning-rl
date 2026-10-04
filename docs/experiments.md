# Running experiments

## Offline software verification

From the repository root, after installing `.[dev]`:

```bash
python scripts/check.py
python -m src.cli evaluate --config configs/evaluation.yaml
python -m src.cli experiment --config configs/smoke.yaml
python -m src.cli plot --results results/fixture-sweep
```

The fixture sweep uses three seeds and five configurable levels, producing 15 verifier audits.
The underlying fixture policy is held constant across levels for a seed; only the verifier changes.
It does not create policy-training comparisons. Plots explicitly say that they are software fixtures.

## Real training

Install `.[dev,training]`. TRL 0.26.2 is pinned, with Transformers 4.x and PEFT support. The
training-extra tests create a random local tiny GPT-2 and train one step on CPU for both methods;
they require no downloaded model. This checks the actual optimizer, reward-call alignment,
KL configuration, LoRA, logging, and checkpoint save. It does not test convergence.

```bash
python -m src.cli train --config configs/baseline.yaml
python -m src.cli train --config configs/robust.yaml
python -m src.cli train --config configs/robust.yaml --resume checkpoints/robust/checkpoint-50
python -m src.cli experiment --config configs/reward_corruption.yaml
```

The full sweep runs the base policy, standard GRPO, and robust GRPO at each seed and level.
Train and evaluation datasets must have disjoint IDs and questions. Each trained run starts from
the same base checkpoint. Baseline and robust runs use identical data, inference, corruption
seeds, batch/learning settings, LoRA, and KL coefficient; reward shaping is the changed variable.
An exception is persisted in the sweep summary; default behavior stops at the failure. Set
`experiment.continue_on_error: true` to continue and retain failures. CLI exits nonzero if any
run fails. Completed intermediate runs remain reviewable.

The provided tiny split is a runnable demonstration, not an adequate research dataset. To load
a standard mathematical dataset configure `source: huggingface`, `name`, `subset`, `split`,
`revision`, `question_column`, `answer_column`, and optionally `answer_separator: '####'`.
Adapters assign row-index IDs when none exist; supply distinct train/evaluation ID columns for
a sweep to avoid accidental overlap. Dataset-specific reference extraction belongs in this adapter.

For pretrained evaluation set `model.backend: huggingface`, `model.name`, and optional
`model.tokenizer`, `model.revision`, `model.device`. To evaluate a saved LoRA checkpoint add
`model.adapter: checkpoints/robust/final`; for full finetuning set `model.name` to the saved
checkpoint. Use the same tokenizer as training. `evaluation_verifier` controls the independent
judge and `training_verifier` retains the reward source for transfer diagnostics.

## Sampling and attribution protocol

Hold n, temperature, top-p, max output tokens, prompt template, dataset, and seed schedule
constant across policy comparisons. Set `evaluation.k` to values <= n. Repeat evaluation with
temperature zero or low temperature and with stochastic sampling. Keep separate output folders.
An increased pass@k with unchanged pass@1 primarily supports a sampling explanation.
Compare transferred verifier correctness with training-source reward to identify reward overfitting.
Confidence or teacher effects need separate ablations; the framework does not attribute causality
automatically. Report actual compute and generated lengths when publishing experiments.

## Outputs and plotting

`evaluation.json` contains provenance, metrics, and complete generation records;
`generations.jsonl` supports per-sample audits. `config.json` stores resolved configuration.
`training.json` records actual trainer metrics/history, reward attenuation counts, and final checkpoint.
`run.json` marks training initiation; `failure.json` marks training failures inside the trainer loop.
Sweeps produce `manifest.json`, `summary.json` (including failed runs), and `summary.csv`.

Metadata records UTC timestamp, Git SHA/dirty state, installed numerical/model packages,
model/data/generation/training/verifier configuration, and seed. Output directories are overwritten
when reused; choose a unique output_dir for each experimental campaign. Retain failed runs
alongside successes. Generated artifacts are ignored by Git to avoid committing model weights.

Plotting consumes summary CSVs only. It rejects mixtures of fixture and research runs or
different corruption kinds. Do not point it at campaigns with different model/data or training
settings: compare one controlled campaign per folder. PNG and PDF robustness/gap/reward
curves and PNG reward/correctness, disagreement, and sampling panels use only recorded values.

## Evidence status

The repository makes no claim that robust GRPO outperforms standard GRPO. CPU fixture audits
and local random-model optimizer tests are software validation. Phase 2 adds measured pretrained
SmolLM2/GSM8K smoke and preliminary CPU experiments: see [the evidence protocol](phase2.md).
Larger benchmark evaluation, convergence studies, and meaningful statistical comparisons still
require substantially more compute and data.
