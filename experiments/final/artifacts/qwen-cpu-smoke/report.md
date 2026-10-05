# Measured experiment report

Automatically generated from summary.json. No experimental values are manually supplied.

Completed conditions: 3. Recorded failures: 0.

## Main table (seed means)

| corruption_level | method | pass@1_mean | independent_accuracy_mean | observed_reward_mean | reward_hacking_gap_mean |
| --- | --- | --- | --- | --- | --- |
| 0.200000 | base | 0.000000 | 0.000000 | 0.250000 | 0.250000 |
| 0.200000 | baseline | 0.000000 | 0.000000 | 0.250000 | 0.250000 |
| 0.200000 | robust | 0.125000 | 0.125000 | 0.291667 | 0.166667 |

## Per seed

| corruption_level | method | seed | pass@1 | pass@2 | independent_accuracy | observed_reward | reward_hacking_gap | false_positive_reward_rate | false_negative_rate | verifier_disagreement | extraction_failure_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.200000 | base | 42 | 0.000000 | 0.000000 | 0.000000 | 0.250000 | 0.250000 | 0.125000 | undefined | 0.555556 | 1.000000 |
| 0.200000 | baseline | 42 | 0.000000 | 0.000000 | 0.000000 | 0.250000 | 0.250000 | 0.125000 | undefined | 0.555556 | 1.000000 |
| 0.200000 | robust | 42 | 0.125000 | 0.250000 | 0.125000 | 0.291667 | 0.166667 | 0.142857 | 0.000000 | 0.333333 | 0.875000 |

## Across seeds

| corruption_level | method | seed_count | independent_accuracy_mean | independent_accuracy_std | reward_hacking_gap_mean | reward_hacking_gap_std |
| --- | --- | --- | --- | --- | --- | --- |
| 0.200000 | base | 1 | 0.000000 | undefined | 0.250000 | undefined |
| 0.200000 | baseline | 1 | 0.000000 | undefined | 0.250000 | undefined |
| 0.200000 | robust | 1 | 0.125000 | undefined | 0.166667 | undefined |

Standard deviations are sample standard deviations across seeds. Undefined rates stay undefined. Tiny budgets do not support significance testing or causal claims. Repeated base responses across corruption levels are not independent evidence.
