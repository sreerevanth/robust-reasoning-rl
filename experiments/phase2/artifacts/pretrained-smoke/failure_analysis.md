# Qualitative examples

Deterministic first eligible example per method/category. All methods are included; unavailable categories are explicitly listed. These labels concern final answers, not reasoning validity. Base duplicates are rescoring of the same responses.

## base/answer_correct_reward_positive

No qualifying measured example.

## base/answer_incorrect_reward_negative

Source: `seed-42/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: 25; observed reward: 0.0; disagreement: 0.0.

```text
To solve this problem, let's break it down step by step.

Frank starts with a starting point of 5 steps back. He then takes 5 steps back again, so he is 5 steps back and then 10 steps forward, which is 5 + 10 = 15 steps forward.

Next, Frank takes 5 steps back again, so he is 5 steps back and then 20 steps forward, which is 5 + 20 = 25 steps forward.

Finally, Frank takes 5 steps back again, so he is 5 steps back and then 5 + 20 = 25 steps forward.

Therefore, Frank is 25 steps forward from his original starting point.

Final answer: 25.
```

## base/answer_incorrect_false_positive

No qualifying measured example.

## base/answer_correct_false_negative

No qualifying measured example.

## base/high_disagreement

Source: `seed-42/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: 26 steps forward; observed reward: 0.3333333333333333; disagreement: 0.888888888888889.

```text
Frank starts with 5 steps back. 10 steps forward = 5 * 5 = 25 steps forward.
Frank starts from his original starting point. 25 steps forward from his original starting point is 25 + 5 = 26 steps forward.
Final answer: 26 steps forward.
```

## baseline/answer_correct_reward_positive

No qualifying measured example.

## baseline/answer_incorrect_reward_negative

Source: `seed-42/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: 25; observed reward: 0.0; disagreement: 0.0.

```text
To solve this problem, let's break it down step by step.

Frank starts with a starting point of 5 steps back. He then takes 5 steps back again, so he is 5 steps back and then 10 steps forward, which is 5 + 10 = 15 steps forward.

Next, Frank takes 5 steps back again, so he is 5 steps back and then 20 steps forward, which is 5 + 20 = 25 steps forward.

Finally, Frank takes 5 steps back again, so he is 5 steps back and then 5 + 20 = 25 steps forward.

Therefore, Frank is 25 steps forward from his original starting point.

Final answer: 25.
```

## baseline/answer_incorrect_false_positive

No qualifying measured example.

## baseline/answer_correct_false_negative

No qualifying measured example.

## baseline/high_disagreement

Source: `seed-42/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: 26 steps forward; observed reward: 0.3333333333333333; disagreement: 0.888888888888889.

```text
Frank starts with 5 steps back. 10 steps forward = 5 * 5 = 25 steps forward.
Frank starts from his original starting point. 25 steps forward from his original starting point is 25 + 5 = 26 steps forward.
Final answer: 26 steps forward.
```

## robust/answer_correct_reward_positive

No qualifying measured example.

## robust/answer_incorrect_reward_negative

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: 25; observed reward: 0.0; disagreement: 0.0.

```text
To solve this problem, let's break it down step by step.

Frank starts with a starting point of 5 steps back. He then takes 5 steps back again, so he is 5 steps back and then 10 steps forward, which is 5 + 10 = 15 steps forward.

Next, Frank takes 5 steps back again, so he is 5 steps back and then 20 steps forward, which is 5 + 20 = 25 steps forward.

Finally, Frank takes 5 steps back again, so he is 5 steps back and then 5 + 20 = 25 steps forward.

Therefore, Frank is 25 steps forward from his original starting point.

Final answer: 25.
```

## robust/answer_incorrect_false_positive

No qualifying measured example.

## robust/answer_correct_false_negative

No qualifying measured example.

## robust/high_disagreement

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: 26 steps forward; observed reward: 0.3333333333333333; disagreement: 0.888888888888889.

```text
Frank starts with 5 steps back. 10 steps forward = 5 * 5 = 25 steps forward.
Frank starts from his original starting point. 25 steps forward from his original starting point is 25 + 5 = 26 steps forward.
Final answer: 26 steps forward.
```
