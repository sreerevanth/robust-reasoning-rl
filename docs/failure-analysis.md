# Failure analysis from actual generations

All examples below link to retained model outputs. Category labels concern final answers;
they do not automatically prove valid or invalid reasoning. The generated per-method reports
select the first eligible record in sorted artifact order, including standard and robust failures.
Markdown excerpts remove trailing line whitespace; full original text remains in JSON/JSONL.

## Correct answer with visible arithmetic

The Qwen robust CPU smoke response for `gsm8k-validation-a22eeeb1e7375c0a` writes
`3x = 2 × 51 - 3`, then `3x = 99` and `x = 33`, ending with
`Final answer: The number is 33.` The reference is 33 and the separate judge accepts it.
This is consistent with correct arithmetic on that question. It is one sample after one
update on validation data, not evidence of a reliable robust-GRPO advantage.
See the [Qwen smoke examples](../experiments/final/artifacts/qwen-cpu-smoke/failure_analysis.md).

## Correct text rejected by the original parser

Qwen's original greedy qualification output on the same number question ended with
`Final answer: The number is 33`, but the old parser retained the sentence instead of a scalar.
Other actual responses ended with `12 teaspoons`, `<answer>9</answer>`, and an emphasized
escaped currency answer `**\$2220**` after an explicit conclusion cue. All matched their
gold references after bounded formatting repairs. These repairs recover existing answers;
they do not improve the model. Both original and rejudged responses are in the
[qualification archive](../experiments/final/artifacts/qualification-qwen/).

The parser still rejects ambiguous equations, multiple answers, and arbitrary numbers inside
reasoning. Real wrong answers such as `49,500 televisions` (reference 477) and
`0.0025 pounds per square inch` (reference 4) remain wrong after formatting normalization.
Nonnumeric or ambiguous final strings now count as extraction failures on scalar tasks.

A manual audit of the historical SmolLM2 pilot also found an unmarked final sentence
with the reference number: seed 42, base, sample 0 for `gsm8k-test-6dbed0240798dcb4`
ends "John's profit on his car is $250." The historical parser returned no final answer.
Its visible payment and earnings calculations are incorrect despite the matching final number.
This illustrates a limitation of the historical judged floor; it is not a rescored metric
or evidence of valid reasoning. The original scores and outputs remain unchanged, and the
parser was not expanded using this held-out case. See the
[original evaluation](../experiments/phase2/artifacts/cpu-pilot/seed-42/level-0/base/evaluation/evaluation.json).

## Incorrect answer with false-positive reward and high disagreement

The original held-out pen-reuse question has reference 31: 25 original pens, five refills,
then one final refill. A real response ends `The answer is 5` while describing inconsistent
pen counts. Two corrupted members reward it at 20% corruption, producing reward 2/3 and
disagreement 8/9; the uncorrupted judge rejects it. The same failure occurs for base, standard,
and robust policies. See the [detailed audit](failure_analysis.md) and
[all pilot method examples](../experiments/phase2/artifacts/cpu-pilot/failure_analysis.md).

This is a false-positive reward and a potential exploitation opportunity. Since the untouched
base model also produces it, the observation does not establish that RL learned exploitation.
The robust method did not remove this failure in the measured pilot.

## Truncation and genuine model weakness

The [original base-output diagnosis](../experiments/final/pilot_diagnosis.json) classifies
48 real pilot responses: 20 truncated, seven missing a marker after EOS, sixteen with
nonscalar final strings, and five parsed wrong scalars. Truncation takes precedence in this
classification, so these are disjoint diagnostic categories rather than a revised accuracy.
Gold-answer acceptance checks passed for all four questions. The model receives the intended
chat template; sampled decoding uses positive temperature and the configured token limit.
No reference or worked solution appears in policy prompts.

Longer greedy validation generation reduced truncation for SmolLM2 but still yielded no
correct answer. This supports a capability problem in addition to formatting and budget issues.
Qwen's qualification shows nonzero ability on the same fixed validation pool. Its remaining
failures include truncated Python-like reasoning and incorrect numerical setups, not merely
recoverable final formatting.

## Standard and robust failures

Both Qwen smoke policies still fail most attempts under the deliberately short 128-token
pipeline budget. Base and standard score zero, while robust has one accepted answer among
eight samples. This tiny smoke cannot separate stochastic differences from policy improvement.
The [generated report](../experiments/final/artifacts/qwen-cpu-smoke/report.md) preserves these
outcomes and the [parameter diagnostics](../experiments/final/artifacts/qwen-cpu-smoke/training_diagnostics.csv)
show that both adapters changed. Weight movement is not proof of capability improvement.

## Unavailable false-negative illustration

The original pilot has no accepted correct answers under its recorded judge, so its
recorded false-negative rate is undefined. In the
Qwen smoke, the sole accepted answer receives positive ensemble reward; no correct-answer /
false-negative example was observed. The candidate qualification uses uncorrupted rewards.
No false-negative illustration is manufactured. The framework supports and tests that corruption
mechanism, but those unit tests are not pretrained-model evidence.
