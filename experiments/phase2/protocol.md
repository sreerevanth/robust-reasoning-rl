# Phase 2 fixed protocol

Frozen before pretrained inference or optimization. The CPU smoke uses validation examples,
not the final test subset. The CPU pilot uses the first four examples of a seeded, unfiltered
official GSM8K test shuffle. Training uses eight examples from a disjoint official train shuffle.
Selection seed is 20261004; question hashes, full selected data, and pinned source revisions
are in the dataset manifest. No examples are selected according to model accuracy or difficulty.

The smoke executes base evaluation, one standard-GRPO step, and one robust-GRPO step at
20% corruption, then saves/reloads/evaluates real LoRA adapters. It establishes integration only.
It is permissible to fix parsing defects observed on validation generations, with regression tests,
before running the final pilot. The parser is shared across every policy condition.

The preliminary CPU pilot is fixed at two optimizer updates per trained condition, four sampled
generations per prompt, temperature 0.7, top-p 0.95, 192 maximum output tokens, four held-out
questions, seeds 42/123/456, and corruption probabilities 0/0.2/0.4. It compares untouched
SmolLM2-135M-Instruct, standard GRPO, and the existing combined robust strategy. All other
parameters match. No learning-rate, penalty, threshold, or prompt tuning follows test results.
Missing 10%/60% conditions are deferred to the GPU configuration, not silently omitted.

Use the model's chat template consistently for training and inference. This was identified from
the model card before collecting outcomes. Plain-text prompt experiments from Phase 1 are
software tests and are not pooled with this campaign.

The judge is a separate uncorrupted rule-verifier instance. It shares scalar answer parsing
with the training verifier, so this is weak verifier independence, not an external semantic judge.
Correctness is answer correctness, not proof of faithful or valid intermediate reasoning.
Corruption is keyed by exact response and seed, and repeated identical responses share signals.

The principal question is not answerable conclusively with this budget. Report all results,
training-zero-advantage diagnostics, response truncation, and extraction failures. Positive gaps
can arise from deliberate synthetic false positives and alone do not prove learned exploitation.
Report seed means and sample standard deviations, no significance tests. If correctness remains
at a floor or adapters receive no effective reward gradient, state that explicitly.
