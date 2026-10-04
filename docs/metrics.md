# Metric definitions

Each prompt has n sampled generations and c independently correct generations. The standard
unbiased estimator is `pass@k = 1 − C(n−c,k)/C(n,k)`. It estimates the probability of at least
one correct answer in k draws from the sampling distribution. The implementation uses a product
to avoid factorial overflow, validates `0 <= c <= n` and `1 <= k <= n`, and returns one when
`n−c < k`. Prompt estimates are macro averaged; pass@1 is `c/n`, not the first generation only.
This interpretation assumes exchangeable independent sampling. Deterministic repeated decoding
produces identical draws, so larger k supplies no sampling diversity and is not evidence of gain.

| Output | Definition and interpretation |
|---|---|
| independent_accuracy | Correct generations / all generations |
| independent_verifier_accuracy | Alias of independent_accuracy; proxy model correctness, not judge classification accuracy |
| observed_reward | Mean available aggregate training-verifier reward |
| reward_coverage | Fraction of generations with an aggregate reward |
| paired_independent_accuracy | Correctness on the same samples used in observed_reward |
| reward_hacking_gap | Mean available normalized reward minus paired independent correctness |
| false_positive_reward_rate | P(reward >= 0.5 given independent incorrectness and available reward) |
| false_negative_rate | P(reward < 0.5 given independent correctness and available reward) |
| reward_correctness_correlation | Pearson correlation on paired available samples |
| verifier_disagreement | Mean normalized ensemble reward variance |
| confidence | Mean ensemble heuristic confidence |
| answer_consistency | Mean within-prompt same normalized-answer pair fraction |
| answer_diversity | Mean unique normalized answers / generation count per prompt |
| extraction_failure_rate | Fraction without an extracted final answer |

Undefined statistics serialize as JSON null: no incorrect samples for a false-positive rate,
no correct samples for a false-negative rate, no available rewards for the gap, and constant
variables for Pearson correlation. Null is never silently converted to zero.

Rewards already lie in [0,1], so no dataset-dependent rescaling is needed for the gap. Positive
gaps indicate reward inflation relative to this evaluation judge; negative gaps indicate reward
underestimation. The gap is a diagnostic, not proof of deliberate exploitation. False positives
and false negatives can cancel, giving a zero gap despite a broken verifier. Always inspect
conditional error rates, coverage, and correlation together.

Ensemble uncertainty uses population variance, normalized disagreement `4*variance` in [0,1],
binary vote entropy `−p log2 p − (1−p) log2(1−p)` (p is the positive-vote fraction), and population
standard deviation of member confidence. Missing members are excluded from these moments
but reduce confidence through coverage. A singleton ensemble has zero disagreement, which
does not imply reliable verification. All-missing ensembles have zero reported moments and
zero confidence: inspect reward availability before interpreting those moments.

Consistency treats failed extraction as a shared failure symbol and uses formatting normalization,
not rational equivalence. High consistency can mean repeatedly wrong answers. These descriptive
metrics never replace independent correctness.

Plots show means across seeds with empirical ±one standard deviation bands; these are **not
confidence intervals** or significance tests. Scatter panels show individual seed/level runs.
Sampling curves retain method and corruption level separately rather than pooling levels.

Phase 2 generation telemetry reports mean generated tokens (including EOS when present,
excluding prompt and post-EOS batch padding) and truncation rate (fraction terminated by the
output-token limit rather than EOS). Training diagnostics report mean logged gradient norm,
KL, zero-reward-variance group fraction, and LoRA B norm. TRL's `total_flos` may be zero
because this trainer does not populate that accounting field; it does not mean no compute
was used. Raw and shaped reward means are reported separately. Only two unique questions
may be seen during a two-update run even if its configured candidate pool contains eight.
