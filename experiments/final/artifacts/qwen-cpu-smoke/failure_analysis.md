# Qualitative examples

Deterministic first eligible example per method/category. All methods are included; unavailable categories are explicitly listed. These labels concern final answers, not reasoning validity. Base duplicates are rescoring of the same responses.

## base/extraction_failure

Source: `seed-42/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## base/truncated

Source: `seed-42/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## base/answer_correct_reward_positive

No qualifying measured example.

## base/answer_incorrect_reward_negative

Source: `seed-42/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## base/answer_incorrect_false_positive

Source: `seed-42/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
To solve this problem, we need to track Frank's movements step-by-step to determine his final position relative to his original starting point.

Let's break down Frank's movements:

1. **First Movement (backward):**
   - Start at the origin.
   - Take 5 steps back.
   - Ending position after this movement: \( x \)

2. **Second Movement (forward):**
   - Move 10 steps forward from the previous location.
   - Ending position after this movement: \( x + 10 \)

3. **Third Movement (backward):**
   - From the previous ending position, take 
```

## base/answer_correct_false_negative

No qualifying measured example.

## base/high_disagreement

Source: `seed-42/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
To solve this problem, we need to track Frank's movements step-by-step to determine his final position relative to his original starting point.

Let's break down Frank's movements:

1. **First Movement (backward):**
   - Start at the origin.
   - Take 5 steps back.
   - Ending position after this movement: \( x \)

2. **Second Movement (forward):**
   - Move 10 steps forward from the previous location.
   - Ending position after this movement: \( x + 10 \)

3. **Third Movement (backward):**
   - From the previous ending position, take 
```

## baseline/extraction_failure

Source: `seed-42/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## baseline/truncated

Source: `seed-42/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## baseline/answer_correct_reward_positive

No qualifying measured example.

## baseline/answer_incorrect_reward_negative

Source: `seed-42/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## baseline/answer_incorrect_false_positive

Source: `seed-42/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
To solve this problem, we need to track Frank's movements step-by-step to determine his final position relative to his original starting point.

Let's break down Frank's movements:

1. **First Movement (backward):**
   - Start at the origin.
   - Take 5 steps back.
   - Ending position after this movement: \( x \)

2. **Second Movement (forward):**
   - Move 10 steps forward from the previous location.
   - Ending position after this movement: \( x + 10 \)

3. **Third Movement (backward):**
   - From the previous ending position, take 
```

## baseline/answer_correct_false_negative

No qualifying measured example.

## baseline/high_disagreement

Source: `seed-42/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
To solve this problem, we need to track Frank's movements step-by-step to determine his final position relative to his original starting point.

Let's break down Frank's movements:

1. **First Movement (backward):**
   - Start at the origin.
   - Take 5 steps back.
   - Ending position after this movement: \( x \)

2. **Second Movement (forward):**
   - Move 10 steps forward from the previous location.
   - Ending position after this movement: \( x + 10 \)

3. **Third Movement (backward):**
   - From the previous ending position, take 
```

## robust/extraction_failure

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## robust/truncated

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## robust/answer_correct_reward_positive

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-a22eeeb1e7375c0a`.

Armand is playing a guessing game with his dad where he has to guess a number his dad is thinking of. His dad tells him that the number, when multiplied by 3, is three less than twice 51. What is the number?

Reference: 33; extracted: 33; observed reward: 1.0; disagreement: 0.0.

```text
Let's denote the unknown number as \( x \).

According to the problem statement:
\[ 3x = 2 \times 51 - 3 \]

First, let's solve for \( x \):
\[ 3x = 102 - 3 \]
\[ 3x = 99 \]
\[ x = \frac{99}{3} \]
\[ x = 33 \]

Therefore, the number Armand needs to think of is 33.

Final answer: The number is 33.
```

## robust/answer_incorrect_reward_negative

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To determine how many steps forward Frank is from his original starting point after performing each movement, we need to follow these steps:

1. **Starting Position**: Frank's original position is at zero steps forward.
2. **First Movement**:
   - Take 5 steps back.
   - New position = \(0 + (-5) = -5\) steps forward.
3. **Second Movement**:
   - Take 10 steps forward.
   - New position = \(-5 + 10 = 5\) steps forward.
4. **Third Movement**:
   - Take 2 steps back.
   - New position
```

## robust/answer_incorrect_false_positive

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
To solve this problem, we need to track Frank's movements step-by-step to determine his final position relative to his original starting point.

Let's break down Frank's movements:

1. **First Movement (backward):**
   - Start at the origin.
   - Take 5 steps back.
   - Ending position after this movement: \( x \)

2. **Second Movement (forward):**
   - Move 10 steps forward from the previous location.
   - Ending position after this movement: \( x + 10 \)

3. **Third Movement (backward):**
   - From the previous ending position, take 
```

## robust/answer_correct_false_negative

No qualifying measured example.

## robust/high_disagreement

Source: `seed-42/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-validation-5e6375912f4d30ca`.

Frank is practicing a new dance move.  It starts with him take 5 steps back, and then 10 steps forward, and then 2 steps back, and then double that amount forward.  How many steps forward is Frank from his original starting point?

Reference: 7; extracted: None; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
To solve this problem, we need to track Frank's movements step-by-step to determine his final position relative to his original starting point.

Let's break down Frank's movements:

1. **First Movement (backward):**
   - Start at the origin.
   - Take 5 steps back.
   - Ending position after this movement: \( x \)

2. **Second Movement (forward):**
   - Move 10 steps forward from the previous location.
   - Ending position after this movement: \( x + 10 \)

3. **Third Movement (backward):**
   - From the previous ending position, take 
```
