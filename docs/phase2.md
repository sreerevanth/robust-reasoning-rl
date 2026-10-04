# Phase 2: pretrained experiments on GSM8K

## Evidence and scope

This phase uses real pretrained **SmolLM2-135M-Instruct**, real GSM8K questions, TRL optimizer
updates, actual LoRA checkpoints, and checkpoint reloads for held-out generation. It does not
replace the Phase 1 framework or treat synthetic arithmetic fixtures as model evidence.

The [fixed protocol](../experiments/phase2/protocol.md) was committed before the held-out pilot.
The [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) specifies a chat
template; training and inference now both use that template. The model revision is
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The
[GSM8K source](https://huggingface.co/datasets/openai/gsm8k) revision is
`740312add88f781978c0658806c59bc2815b9866`. GSM8K's upstream MIT notice is retained in
`data/GSM8K_LICENSE`; no model weights are committed.

## Hardware and practical budget

The recorded host is Windows, Python 3.11.9, PyTorch 2.14.1+cpu, eight physical CPU cores,
16 logical processors, and 16,486,756,352 bytes of RAM. CUDA is unavailable and its version
is null; there is no CUDA GPU or VRAM. About 3 GB RAM was available at inspection time.
See [hardware.json](../experiments/phase2/hardware.json). CPU experiments use four PyTorch
threads. Model loading, training activations, and concurrent host applications affect runtime.

The smoke uses two validation questions, eight candidate training questions, seed 42, one update
per method, four generations, 192 output tokens, temperature 0.7, top-p 0.95, and 20% flips.
The CPU pilot uses four held-out official-test questions, eight candidate training questions,
two updates per trained condition, four generations, and the same decoding parameters.
The pilot protocol specifies seeds 42/123/456 and corruption levels 0/20/40%.
These extremely small budgets cannot establish convergence, benchmark performance, or
statistical significance. The unexecuted GPU configuration includes 0/10/20/40/60%,
three seeds, 200 updates, 256 training questions, 128 test questions, and pass@1/4/8.
Even that configuration remains a subset study, not a claim of a complete GSM8K benchmark.
Temperature and top-p are explicit. In this pinned integration, pretrained inference additionally
inherits Transformers' default top-k 50, whereas TRL training has top-k disabled by its default.
This difference is shared by both trained methods; inference comparisons remain matched across
all policies. It should not be confused with identical training and inference sampling distributions.

## Splits and preprocessing

The official train split is shuffled at seed 20261004, with its first 32 questions saved as a
training pool and the next 16 as validation. The official test split is independently shuffled
with the same seed and its first 32 saved as a test pool. CPU configs use fixed prefixes of
these pools. There is no difficulty filtering or selection by model accuracy. The preparation
script asserts disjoint question hashes across all three saved pools. IDs are split-qualified
hashes; raw file SHA256s and original split/selection indices are in the
[dataset manifest](../experiments/phase2/dataset_manifest.json).

References are scalar text after the final GSM8K `####`. The question alone is given to the
model; references and original worked solutions do not enter policy prompts. Saved records
include references for auditing. The larger GPU config uses the adapter's seeded shuffle and
namespaced IDs directly on the official dataset; its sampling algorithm is Python's shuffle
and differs from the Hugging Face shuffle used by the saved CPU pools. Do not pool them.

## Parsing audit and independent judging

The original validation smoke exposed explicit final answers such as `5 steps forward` and
`96 second graders`. A regression-tested whitelist strips those count units when the prefix
is a valid scalar. Explanatory equations (`2 + 3 = 5`) or alternative numbers (`42 or 43`)
are not converted into a conveniently chosen answer. No last-number fallback is added.
The original smoke artifacts retain their original parsing and sequential decoding results.
The pilot uses the fixed parser for every method and batches four inference samples per call.
Validation and test outcomes are not pooled.

Evaluation judges are separate, uncorrupted verifier objects. However, they share scalar parsing
with the reward verifiers. This is weak independence and cannot detect all common implementation
errors or prove faithful reasoning. The audit labels **answer correctness**, not reasoning-step
correctness. A high reward/correctness gap can be caused directly by injected false positives;
it does not establish that a policy learned to exploit the verifier.

## Reproducible commands

From the repository root with `.[dev,training]` installed:

```bash
python scripts/prepare_phase2.py
python -m src.cli --verbose experiment --config configs/pretrained_smoke.yaml
python -m src.cli --verbose experiment --config configs/gsm8k_cpu_pilot.yaml
python scripts/report_results.py --results results/gsm8k-cpu-pilot
python -m src.cli plot --results results/gsm8k-cpu-pilot
python scripts/archive_results.py --results results/gsm8k-cpu-pilot --destination experiments/phase2/artifacts/cpu-pilot
```

Preparation pins the published revisions and reconstructs the deterministic pools. Do not run
it just to analyze already archived evidence: it refreshes the inspection hardware file and may
need network access. Config paths, sample counts, and seeds are sufficient for reproduction.
Use a new output directory for independent replications rather than overwriting historical runs.

On a compatible CUDA host, with a CUDA-enabled PyTorch installation:

```bash
python -m src.cli --verbose experiment --config configs/gsm8k_gpu.yaml
python scripts/report_results.py --results results/gsm8k-gpu
python -m src.cli plot --results results/gsm8k-gpu
```

The GPU config enables bfloat16; disable it for hardware without bfloat16 support. No GPU run
is claimed here. Checkpoint directories stay on the execution host and are excluded from Git.
Recorded checkpoint paths identify actual saves; portability requires rerunning training or
separately transporting those weights.

## Reading the evidence

The [pretrained smoke report](../experiments/phase2/artifacts/pretrained-smoke/report.md)
is generated from three measured conditions and is integration evidence only. It includes
raw generations, training audits, actual loss/KL/gradient histories, and adapter-update norms.
The pilot's automatically generated report and aggregate CSV use seed means and sample
standard deviations. Undefined conditional error rates remain undefined. Plots show empirical
standard deviations, not confidence intervals.

Failure cases are selected by a deterministic first-eligible rule for each method and category.
Robust-policy failures are retained. Categories without examples are explicitly reported; no
correct example is manufactured to complete the display. Long generations are excerpted in
Markdown while full text remains in JSON/JSONL. Replayed base responses are flagged and do
not count as additional independent model-generation evidence.
The official test split is held out from this finetuning, not proven absent from model pretraining
or instruction tuning. Prior benchmark exposure and teacher effects cannot be excluded by this study.

Use training diagnostics to distinguish real optimization from a completed trainer loop with no
effective update: inspect reward variance, fraction of zero-variance groups, gradient norms,
and LoRA B norm (initialized at zero by default). If pass@1 stays at zero or a large fraction of
outputs is truncated, the experiment cannot discriminate robust reasoning capability. Even
improved pass@4 would only be suggestive sampling evidence under this small budget.

The existing severe threshold 0.9 is above the maximum binary disagreement 8/9 achievable
by three members. Consequently the severe suppression branch is inactive in this specific
campaign; moderate attenuation is the operative safety mechanism. This was not retuned after
observing outputs. The generated reports include paired seed-level method differences and an
explicit list of any uncompleted planned conditions, so incomplete campaigns are visible.
