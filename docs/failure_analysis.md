# Qualitative failure audit

The automatically selected examples and full records are the authoritative evidence. Selection
is deterministic (first eligible example per method/category in sorted artifact order), rather
than a search for cases favorable to robust GRPO. The narrative below inspects an actual
completed seed-42 example; it is not a newly generated or hand-edited model response.

## Recursive pen reuse: incorrect answer with positive corrupted reward

GSM8K question `gsm8k-test-029ef1d7713b4f7c` asks how many pens are available in total
when 25 original pens are used and the ink remnants of five empty pens can form one full pen.
The reference is 31: the 25 originals yield five refills, whose remnants yield one more refill.

The recorded response concludes `The answer is 5`, while earlier text describes adding
25 pens and another 25 pens. Its final answer counts neither all original pens nor the last
recursive refill, and its visible explanation is internally inconsistent. This is a genuine
incorrect scalar answer, not merely an extraction failure caused by units or a missing marker.

At 20% corruption, two ensemble members assign positive rewards to that incorrect answer.
The aggregate reward is 2/3 with normalized disagreement 8/9. The separate uncorrupted judge
rejects it. The same false-positive response is present for the base, standard-GRPO, and
robust-GRPO seed-42 evaluations. Robust training therefore has not removed this failure in
the measured two-update pilot. Nonzero adapter updates do not guarantee changed answers.

This example is from the held-out test pool and was not supplied as a training prompt. Its
positive observed reward is computed during evaluation. It demonstrates a reward error and
a potential exploitation opportunity; it does **not** show that any policy learned to exploit
this question or intentionally became dishonest. The untouched base policy already exhibits it.

## Unavailable categories and limited inference

The complete seed-42 pilot contains no independently correct answer, so no correct-answer /
correct-reward or correct-answer / false-negative illustration can be drawn from that seed.
The generated failure report explicitly marks such categories unavailable. Missing examples
are not replaced with synthetic successes. Incorrect, falsely rewarded, high-disagreement,
and robust-policy failures are all retained.

Other outputs stop at the configured 192-token limit without an explicit final marker. These
are answer failures under the shared evaluation protocol, but they do not establish that all
intermediate reasoning was wrong. The final campaign report separates extraction failure and
truncation rates from independent final-answer correctness. Larger inference budgets could
change these observations; they have not been substituted into the fixed pilot.

See the generated `failure_analysis.md` and `failure_analysis.json` in the archived campaign
for exact source paths, all method/category selections, and full transcripts. No causal or
statistically significant advantage is claimed from this qualitative audit.
