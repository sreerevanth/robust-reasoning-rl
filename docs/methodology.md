# Methodology and research integrity

## Questions and hypotheses

Does training against an imperfect verifier improve observed reward while independent correctness
stagnates? Can confidence and disagreement based shaping reduce this gap at matched compute?
Does a gain persist under a transferred evaluation verifier and at pass@1, or arise only when
multiple attempts are permitted? These are hypotheses to test, not findings of this repository.

## Rewards and optimization

All observed verifier rewards are bounded in [0,1]. Missing rewards are `None`, not negative labels.
An ensemble aggregates available rewards by mean, majority (ties are 0.5), confidence-weighted
mean, or minimum. Zero total confidence under weighted aggregation yields a missing reward.

Let `r` be aggregate reward, `U=4 Var(r_i)` normalized disagreement, and `C` be confidence.
The implemented methods are:

| Strategy | Shaped reward before attenuation |
|---|---|
| standard | r |
| confidence | r C |
| penalized | r − lambda U |
| combined | r C − lambda U |
| conservative | minimum available member reward |

Robust strategies then multiply by a scale set by disagreement thresholds: one at low
disagreement, `moderate_scale` at moderate disagreement, and `severe_scale` at severe
disagreement. Threshold equality enters the higher suppression tier. Negative penalty-shaped
rewards are allowed. Standard rewards bypass attenuation to preserve the baseline definition.

Ensemble confidence is mean member confidence times available-member fraction times `(1-U)`.
This is a heuristic reliability indicator, not a calibrated probability of correctness. Correlated
verifiers can agree and still be wrong. Confidence degradation scales member confidences
independently of whether an individual corruption event occurs, modelling unreliable reporting.

TRL applies group-relative centering and a reference-policy KL term controlled by `beta`.
Reward standard-deviation scaling is disabled (`scale_rewards="none"`) for both methods so
attenuation is not immediately canceled by group standardization. Centering still couples samples.
**Zero shaped reward does not imply zero optimization weight or sample filtering.** Suppression
counters describe reward attenuation only; the framework does not claim a gradient mask. Missing
rewards map to zero shaped reward and are counted separately. Such samples can still receive
negative centered advantages. This imputation is an explicit limitation, especially at high missingness.

Use nonzero `beta` to apply TRL's reference-policy penalty. With PEFT, TRL uses the base policy
with adapters disabled as its reference. A frozen baseline alone is not a learned independent judge.

## Corruption design

Random flips complement a reward; false positives and negatives alter the appropriate side of
the binary threshold (0.5); missingness removes signals; Gaussian noise is clipped to [0,1];
systematic bias targets a metadata key/value and assigns `bias_reward`. Probability controls
event application, and `noise_std` controls magnitude conditional on the event. At probability
zero, reward is unchanged. Confidence scaling can still change reported confidence.

SHA256 of `(seed, example_id, exact response)` seeds each event. This makes corruption stable
under evaluation reordering and repeatable for identical responses. Repeated identical responses
receive the same corrupted signal; these are not independent Bernoulli trials. Separate ensemble
seeds create independent member draws conditional on the same response. Changing textual
reasoning may change corruption; this synthetic design is not a learned verifier threat model.

## Honest interpretation

The rule verifier recognizes exact rational scalars and limited formatting. It does not validate
reasoning steps, symbolic proofs, code, units, approximate equality, or semantic equivalence of
general expressions. Its accuracy is proxy correctness, not an oracle for genuine reasoning.
The exact and rule verifiers share parsing code; transfer between them is weak independence.
Future independent evaluators should differ in implementation, information source, and failure modes.

No teacher supplies generated reasoning in this implementation. A model's prior instruction tuning
may nevertheless contain teacher effects or benchmark exposure. Record base-model provenance,
dataset revision, and train/evaluation split. The tiny local dataset proves software execution only.
Do not use its results for benchmark, significance, or superiority claims.

PyTorch deterministic algorithms warn on unsupported nondeterministic operations. CUDA kernels,
hardware, library versions, dataset/model revisions, and distributed ordering can still affect results.
Pin revisions before research runs and retain the recorded package/configuration manifest.
