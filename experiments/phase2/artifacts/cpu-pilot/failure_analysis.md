# Qualitative examples

Deterministic first eligible example per method/category. All methods are included; unavailable categories are explicitly listed. These labels concern final answers, not reasoning validity. Base duplicates are rescoring of the same responses.

## base/extraction_failure

Source: `seed-123/level-0/base/evaluation/evaluation.json`; ID: `gsm8k-test-78dd8c8b20f38c37`.

Michael is traveling on the interstate at an average speed of 50km/hr (taking into account all the necessary stops for gas, relaxation, etc.). If he goes on like this for 3 days, and the distance between Alaska and Texas is 6000 km, what percentage of this distance has he covered?

Reference: 60; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To solve this problem, we need to first convert the distance from miles to kilometers, and then find the percentage of the distance covered.

Distance from Alaska to Texas is 6000 km. Since the distance is in kilometers, we need to convert it to miles. 6000 km * 1.5 = 9000 miles.

Now, we can use the formula for percentage to find the percentage of the distance covered: ((distance in miles / distance from Alaska) * 100) * 1000 = (9000 * 100) / 9000 * 1000 = 9%.

So, Michael has covered a total distance of 9% of the distance from Alaska to Texas, which is 9000 miles.
```

## base/truncated

Source: `seed-123/level-0/base/evaluation/evaluation.json`; ID: `gsm8k-test-6dbed0240798dcb4`.

John rents his car out 10 times a month for 3 hours each time.  He gets paid $25 an hour.  If his car payment is $500, how much profit does he make on his car?

Reference: 250; extracted: John makes $3250 per month on his car; observed reward: 0.0; disagreement: 0.0.

```text
John rents his car out 10 times a month for 3 hours each time and earns $25 per hour.

Total payment = 10 x 3 = 30 hours

After payment, John's car loan is 10 x 3 = 30 hours of payment + 500 hours of payment = 3000 hours of payment

Total profit = 3000 hours of payment + 25 hours of payment = 3250 hours of payment

So, John makes a profit of 3250 hours of payment.

Final answer: John makes $3250 per month on his car.

Preliminaries:

John's car rental is paid by credit card, and the loan is made in monthly installments.

John pays $25 an hour for the car loan.


```

## base/answer_correct_reward_positive

No qualifying measured example.

## base/answer_incorrect_reward_negative

Source: `seed-123/level-0/base/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.0; disagreement: 0.0.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## base/answer_incorrect_false_positive

Source: `seed-123/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## base/answer_correct_false_negative

No qualifying measured example.

## base/high_disagreement

Source: `seed-123/level-0.2/base/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## baseline/extraction_failure

Source: `seed-123/level-0/baseline/evaluation/evaluation.json`; ID: `gsm8k-test-78dd8c8b20f38c37`.

Michael is traveling on the interstate at an average speed of 50km/hr (taking into account all the necessary stops for gas, relaxation, etc.). If he goes on like this for 3 days, and the distance between Alaska and Texas is 6000 km, what percentage of this distance has he covered?

Reference: 60; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To solve this problem, we need to first convert the distance from miles to kilometers, and then find the percentage of the distance covered.

Distance from Alaska to Texas is 6000 km. Since the distance is in kilometers, we need to convert it to miles. 6000 km * 1.5 = 9000 miles.

Now, we can use the formula for percentage to find the percentage of the distance covered: ((distance in miles / distance from Alaska) * 100) * 1000 = (9000 * 100) / 9000 * 1000 = 9%.

So, Michael has covered a total distance of 9% of the distance from Alaska to Texas, which is 9000 miles.
```

## baseline/truncated

Source: `seed-123/level-0/baseline/evaluation/evaluation.json`; ID: `gsm8k-test-6dbed0240798dcb4`.

John rents his car out 10 times a month for 3 hours each time.  He gets paid $25 an hour.  If his car payment is $500, how much profit does he make on his car?

Reference: 250; extracted: John makes $3250 per month on his car; observed reward: 0.0; disagreement: 0.0.

```text
John rents his car out 10 times a month for 3 hours each time and earns $25 per hour.

Total payment = 10 x 3 = 30 hours

After payment, John's car loan is 10 x 3 = 30 hours of payment + 500 hours of payment = 3000 hours of payment

Total profit = 3000 hours of payment + 25 hours of payment = 3250 hours of payment

So, John makes a profit of 3250 hours of payment.

Final answer: John makes $3250 per month on his car.

Preliminaries:

John's car rental is paid by credit card, and the loan is made in monthly installments.

John pays $25 an hour for the car loan.


```

## baseline/answer_correct_reward_positive

No qualifying measured example.

## baseline/answer_incorrect_reward_negative

Source: `seed-123/level-0/baseline/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.0; disagreement: 0.0.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## baseline/answer_incorrect_false_positive

Source: `seed-123/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## baseline/answer_correct_false_negative

No qualifying measured example.

## baseline/high_disagreement

Source: `seed-123/level-0.2/baseline/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## robust/extraction_failure

Source: `seed-123/level-0/robust/evaluation/evaluation.json`; ID: `gsm8k-test-78dd8c8b20f38c37`.

Michael is traveling on the interstate at an average speed of 50km/hr (taking into account all the necessary stops for gas, relaxation, etc.). If he goes on like this for 3 days, and the distance between Alaska and Texas is 6000 km, what percentage of this distance has he covered?

Reference: 60; extracted: None; observed reward: 0.0; disagreement: 0.0.

```text
To solve this problem, we need to first convert the distance from miles to kilometers, and then find the percentage of the distance covered.

Distance from Alaska to Texas is 6000 km. Since the distance is in kilometers, we need to convert it to miles. 6000 km * 1.5 = 9000 miles.

Now, we can use the formula for percentage to find the percentage of the distance covered: ((distance in miles / distance from Alaska) * 100) * 1000 = (9000 * 100) / 9000 * 1000 = 9%.

So, Michael has covered a total distance of 9% of the distance from Alaska to Texas, which is 9000 miles.
```

## robust/truncated

Source: `seed-123/level-0/robust/evaluation/evaluation.json`; ID: `gsm8k-test-6dbed0240798dcb4`.

John rents his car out 10 times a month for 3 hours each time.  He gets paid $25 an hour.  If his car payment is $500, how much profit does he make on his car?

Reference: 250; extracted: John makes $3250 per month on his car; observed reward: 0.0; disagreement: 0.0.

```text
John rents his car out 10 times a month for 3 hours each time and earns $25 per hour.

Total payment = 10 x 3 = 30 hours

After payment, John's car loan is 10 x 3 = 30 hours of payment + 500 hours of payment = 3000 hours of payment

Total profit = 3000 hours of payment + 25 hours of payment = 3250 hours of payment

So, John makes a profit of 3250 hours of payment.

Final answer: John makes $3250 per month on his car.

Preliminaries:

John's car rental is paid by credit card, and the loan is made in monthly installments.

John pays $25 an hour for the car loan.


```

## robust/answer_correct_reward_positive

No qualifying measured example.

## robust/answer_incorrect_reward_negative

Source: `seed-123/level-0/robust/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.0; disagreement: 0.0.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## robust/answer_incorrect_false_positive

Source: `seed-123/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```

## robust/answer_correct_false_negative

No qualifying measured example.

## robust/high_disagreement

Source: `seed-123/level-0.2/robust/evaluation/evaluation.json`; ID: `gsm8k-test-029ef1d7713b4f7c`.

Ram uses a lot of pens. He discovered that he can save money by mixing the ink from five empty pens to make one full pen. If he buys 25 pens and then uses them to make new pens when the ink runs low, how many total pens does he get to have?

Reference: 31; extracted: 6; observed reward: 0.6666666666666666; disagreement: 0.888888888888889.

```text
Ram uses a lot of pens.
He discovered that he can save money by mixing the ink from five empty pens to make one full pen.
He uses this new pen to make new pens when the ink runs low.
So, he ends up with 5 empty pens + 1 full pen = 6 pens.
So, he gets to have 6 pens.
Final answer: 6
```
