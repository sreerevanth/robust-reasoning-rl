# Measured model qualification

Fixed 16-question validation pool; no final-test selection. Greedy decoding, one attempt.
This is model selection evidence, not an RL result or a statistically strong benchmark.
Original outputs are retained. Parser-audit metrics rejudge those same actual outputs
with the corrected scalar extraction-failure definition; no new generations are implied.

| model | advertised_parameter_scale | model_revision | examples | pass@1 | original_pass@1 | extraction_failure_rate | original_extraction_failure_rate | truncation_rate | seconds | device | max_tokens | temperature | qualification_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HuggingFaceTB/SmolLM2-135M-Instruct | 135M | 12fd25f77366fa6b3b4b768ec3050bf629380bac | 16 | 0.000000 | 0.000000 | 0.312500 | 0.312500 | 0.187500 | 220.725691 | cpu | 512 | 0 | False |
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | 7ae557604adf67be50417f59c2c2f167def9a775 | 16 | 0.500000 | 0.250000 | 0.250000 | 0.312500 | 0.125000 | 811.542725 | cpu | 512 | 0 | True |
