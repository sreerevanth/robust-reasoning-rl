# Validation record — 4 October 2026

Validation was executed locally on Windows using Python 3.11.9 and a project-local virtual
environment. CUDA was unavailable. This record describes software execution, not benchmark results.

| Check | Executed result |
|---|---|
| `python -m pytest -q` | 98 passed; 2 expected PEFT Conv1D layout warnings |
| `python -m ruff check src tests scripts` | Passed |
| `python -m ruff format --check src tests scripts` | 38 files already formatted |
| `python -m mypy src` | No issues in 29 source files |
| `python -m pip check` | No broken requirements |
| Evaluation CLI | Completed; 4 fixture questions, 16 generated responses |
| Corruption experiment CLI | Completed; 15 fixture audit runs, zero failures |
| Plot CLI | Completed; 9 nonempty plot artifacts |
| Real training integration | Both baseline and robust paths executed one CPU optimizer step |
| Checkpoint/inference integration | Real LoRA save/load and stochastic/greedy generation passed |

The local training test builds a randomly initialized tiny GPT-2 and tokenizer entirely on disk.
It invokes the actual TRL trainer, PEFT, and PyTorch optimizer. No pretrained model download is
needed for these checks. The two warnings are PEFT automatically adapting GPT-2 Conv1D
fan-in/fan-out layout. They do not indicate test failure.

The fixture campaign uses three seeds and corruption probabilities 0, 0.1, 0.2, 0.4, and 0.6.
It tests corrupt reward observation for a fixed arithmetic fixture policy. It does not train policies
or provide evidence of reasoning improvement. Generated files live under ignored `results/`.

The validated training stack includes TRL 0.26.2, Transformers 4.57.6, PEFT 0.18.1,
Accelerate 1.15.0, Datasets 4.8.5, and PyTorch 2.14.1. These are the observed installed versions,
not a promise of identical behavior on other hardware or future dependency versions.

CI configuration is included, but GitHub Actions was not run here: no Git remote is configured.
Meaningful pretrained-policy sweeps, large held-out evaluation, convergence studies, and
statistical comparisons were not executed. They require external training compute.

Reproduce the local checks from the repository root:

```bash
python scripts/check.py
python -m src.cli evaluate --config configs/evaluation.yaml
python -m src.cli experiment --config configs/smoke.yaml
python -m src.cli plot --results results/fixture-sweep
```
