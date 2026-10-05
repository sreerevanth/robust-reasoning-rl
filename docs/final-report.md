# Abstract

We implemented standard and uncertainty-aware GRPO with LoRA, controlled verifier corruption,
independent scalar-answer evaluation, and reproducible experiment reporting. A real SmolLM2
CPU pilot completed 27 conditions across three seeds, but all conditions had zero independently
judged correctness and pass@4. Corruption raised observed reward without improving correctness,
validating instrumentation rather than the primary hypothesis. Subsequent validation qualification
selected Qwen2.5-0.5B-Instruct: four of sixteen original greedy responses were accepted, and eight
were accepted after documented parser corrections to those same responses. The frozen larger
pretrained corruption study requires unavailable CUDA compute and has not run. The repository
provides the framework, pilot evidence, qualified model, and execution package; it does not
establish that robust GRPO preserves reasoning better than standard GRPO.

# Research Question

Does uncertainty-aware policy optimization preserve held-out answer correctness better than
standard GRPO under imperfect rewards? Can evaluation distinguish single-attempt gains from
sampling effects and divergence between reward and correctness?

# Motivation

Reward is a measurement supplied to an optimizer. A corrupted measurement can favor incorrect
answers. Successful optimization of that measurement need not improve actual task performance.
Separate answer judging, per-member reward records, and matched sampling budgets make this
divergence inspectable.

# Experimental Setup

**Status C — Framework + pilot complete, final GPU study externally blocked.**
The [CPU protocol](../experiments/phase2/protocol.md) used three seeds (42, 123, 456),
0/20/40% corruption, base/standard/robust methods, two updates per trained condition, and four
held-out test questions with four samples each. Its records remain unchanged.
The [frozen final protocol](../experiments/final/protocol.yaml) specifies 256 training questions,
128 fresh test questions, three seeds, five corruption levels, eight samples, and 200 updates.
All 45 final conditions are unexecuted. The [preflight record](../experiments/final/execution_status.json)
records unavailable CUDA, not a failed or completed model experiment.

# Models

SmolLM2-135M-Instruct was the original real-model pipeline target. Longer greedy validation
generation still gave zero accepted answers, so it was not selected for the final study.
Qwen2.5-0.5B-Instruct was the smallest additional candidate tested and cleared the capability
gate before parser corrections. Its public model card specifies Apache-2.0 licensing and native
Transformers support: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
Pinned revisions are stored in configs and manifests; weights are excluded from Git.

The [automatically generated qualification table](../experiments/final/qualification/report.md)
retains both candidates, original metrics, corrected-parser metrics, runtime, truncation,
decoding, and device. This sixteen-question selection result is not a benchmark estimate.
The original Qwen and SmolLM2 generations are archived alongside separately rejudged records.
Qwen's subsequent CPU smoke completed base/standard/robust evaluation and both one-update
training runs, saving and independently reloading actual LoRA adapters. Both training runs
changed 168 tensors among 4,399,104 trainable parameters, verified by before/after hashes.

# Dataset

GSM8K comes from `openai/gsm8k`, revision
`740312add88f781978c0658806c59bc2815b9866`, subset `main`; the upstream MIT notice is retained.
References are the scalar following `####`, and only questions enter policy prompts.
The original train/validation/test pools were deterministic and disjoint.
Final training excludes all qualification questions. The fresh final test subset excludes the
entire previously inspected 32-question pilot test pool. Selection uses a fixed shuffle and
does not filter by model success. Counts and SHA256 hashes are in the
[final dataset manifest](../experiments/final/dataset_manifest.json).

# Reward Corruption

The measured CPU sweep flips each of three rule verifiers independently using deterministic
hash-seeded corruption. Changing a response can change its injected corruption, even if it
remains wrong. Observed ensemble reward is the member mean. The final protocol freezes the
same mechanism at 0/10/20/40/60%; these final conditions have no measured results.
Other implemented corruption mechanisms are software capabilities, not additional executed studies.

# Standard GRPO

Standard GRPO optimizes observed ensemble reward with TRL 0.26.2 and LoRA. The pilot records
real optimizer histories and independently reloaded checkpoints. Six clean-reward runs had
zero gradients and unchanged adapters; completing the trainer loop did not imply learning.
Twelve noisy-reward runs had nonzero LoRA B norms. Two updates encountered two unique
training questions per condition, despite an eight-question candidate pool.

# Robust GRPO

The combined shaping rule is `attenuation(U) * (reward * confidence - 0.2 * U)`, with
`U = 4 * population variance`. Standard and robust methods share model initialization,
data, seeds, optimizer, inference, and update budgets. Moderate disagreement attenuates reward.
The inherited severe threshold 0.9 exceeds the maximum 8/9 for three binary members, so that
branch is inactive; it is disclosed rather than tuned to final-test outcomes.
Group centering means zero shaped reward is not a loss-level sample mask.

# Independent Evaluation

A separate uncorrupted verifier object judges scalar rational equality against held-out gold.
Reward and correctness are paired on the same available-reward samples. Conditional error
rates without eligible examples remain undefined. The judges share parsing code; independence
is therefore limited. Exact final answers do not certify correct reasoning steps.
New extraction metrics count nonnumeric final-answer sentences as failures for scalar references.
Historical results preserve the previous definition and are not silently rewritten.

# Capability vs Sampling

Pass@1 is the mean single-sample success estimate; pass@k uses the unbiased combinatorial
estimator under a fixed generation budget. All original pilot pass@1 and pass@4 values were
zero. Neither single-attempt improvement nor improved repeated sampling was observed.
Candidate qualification used one greedy attempt, so it provides no pass@4/8 estimate.
The final execution package measures pass@1/4/8 with equal budgets across methods.

# Results

The [generated CPU report](../experiments/phase2/artifacts/cpu-pilot/report.md),
[aggregate CSV](../experiments/phase2/artifacts/cpu-pilot/aggregate.csv), and
[paired comparisons](../experiments/phase2/artifacts/cpu-pilot/paired_method_differences.csv)
contain all measured conditions. No seed or unfavorable result is omitted.
There are 432 scored pilot records: 336 new generations plus 96 rejudged base responses.
All 27 pilot conditions completed with no failures. Seed means and sample standard deviations
are descriptive; there is no significance test or comparative reasoning claim.

Qualification accepted 0/16 SmolLM2 answers and 4/16 Qwen answers originally. Corrections
to explicit final-answer formats recovered four additional Qwen answers from unchanged text.
The resulting 8/16 is a corrected evaluation of existing validation generations, not newly
learned capability. Original and revised values are both visible in the generated table.

The [Qwen CPU smoke report](../experiments/final/artifacts/qwen-cpu-smoke/report.md) records
zero pass@1 for base and standard and 0.125 for robust (one accepted answer among eight
samples), with robust pass@2 of 0.25. Its four validation questions, one seed, one update,
and 128-token limit make this pipeline evidence only. Most outputs lack an extractable final
answer. These outcomes are not pooled with the 512-token qualification or original test pilot.

# Reward-Hacking Analysis

Corruption increased observed reward while all pilot correctness remained zero. Robust GRPO
did not reduce that divergence: its reward gap was equal to baseline for seeds 42 and 123,
and slightly higher for seed 456 at both noisy levels. These unfavorable comparisons are retained.
The untrained base model also shows reward inflation, demonstrating that injected false
positives alone can produce a gap. The experiment does not establish learned exploitation.
At clean reward, all methods remain at the same correctness floor, so the clean-reward
capability tradeoff cannot be estimated.

# Failure Analysis

The [failure analysis](failure-analysis.md) distinguishes wrong reasoning, missing final markers,
truncation, recoverable final formatting, and corrupted rewards. A deterministic selection
procedure displays baseline and robust failures as well as the base model.
Correct answers discovered during Qwen validation are included; absent categories are explicitly
marked unavailable. No illustrative output is fabricated.

# Limitations

The CPU pilot has only four test questions and two updates per trained condition. The candidate
qualification has sixteen validation questions and uses greedy decoding. Neither supports
general benchmark performance or significance. The full 128-question, multi-seed comparison
is blocked by unavailable CUDA. Its protocol and runner are implemented but not GPU-validated.
The original 192-token pilot limit truncated many responses; qualification increased it to 512.
Longer generation did not rescue SmolLM2's validation correctness.

# Threats to Validity

GSM8K may have appeared in model pretraining or instruction tuning. A held-out finetuning split
does not eliminate prior exposure. Shared parser bugs can affect both reward and independent
judging. Validation-driven parser changes are documented before final-test evaluation, and
ambiguous equations or arbitrary reasoning numbers remain rejected. Three identical deterministic
rule verifiers have no intrinsic epistemic diversity; their disagreement arises from injected
corruption. Synthetic flips need not resemble learned reward-model errors. Historical inference
and training top-k defaults differed; the final protocol explicitly disables top-k in both.
Windows, library versions, decoding batches, and hardware can change exact reproducibility.

# Conclusion

The completed work validates a real pretrained GRPO pipeline and exposes reward/correctness
divergence under corruption. It does not support a robust-GRPO advantage. A more capable
small model now clears an explicit validation gate, legitimate parsing failures are repaired,
and a frozen, resumable final execution package preserves the untested primary hypothesis.
The larger pretrained comparison remains externally blocked; no missing result is implied.
