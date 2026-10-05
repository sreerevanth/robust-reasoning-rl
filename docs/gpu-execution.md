# Frozen pretrained study execution

The local machine has no available CUDA device. No result from the final 45-condition protocol
is claimed. Qwen qualified on separate validation data and its real CPU save/reload smoke ran.
The execution package is checked locally for configuration, imports, CPU preflight, and
resumable orchestration; GPU execution itself remains unvalidated.

On a CUDA-enabled Python 3.11 Linux environment (Colab, Kaggle, or a standard NVIDIA host):

```bash
git clone --branch codex/robust-reasoning https://github.com/sreerevanth/robust-reasoning-rl.git
cd robust-reasoning-rl
bash scripts/setup_gpu.sh
python scripts/run_final_study.py --resume
```

Colab/Kaggle users can run these shell commands in a notebook after selecting a GPU runtime
and enabling network access. Keep the research implementation in the repository. A standard
Linux host needs an NVIDIA driver and a matching CUDA-enabled PyTorch installation; the setup
script verifies actual CUDA availability. The checked-in final protocol uses float16 and fp16
training, compatible with T4-class hardware; its actual memory fit has not been measured locally.

One execution command qualifies the base model on validation using the study's sampling
settings, checks pass@1 >= 0.1, runs the full sweep, aggregates metrics, generates plots, and
exports a ZIP of lightweight evidence. If qualification fails, it retains that evidence and
does not launch RL. Greedy CPU qualification is model selection evidence; the GPU gate checks
the actual study distribution. No threshold is relaxed automatically.

The frozen protocol covers 256 training questions, 128 fresh official-test questions, three
seeds, corruption 0/10/20/40/60%, base/standard/robust methods, 200 updates, and pass@1/4/8.
The parser rules were chosen from validation outputs before any final-test generation.
All hyperparameters and pinned model/data revisions are in
[protocol.yaml](../experiments/final/protocol.yaml). The deterministic data files are committed;
`scripts/prepare_final_data.py` regenerates them and their manifest from the pinned source.

## Resume and artifacts

Use persistent storage for `results/final-study` and its checkpoints before ending a hosted
notebook runtime. `--resume` verifies the same resolved config and qualification evidence,
then skips completed conditions whose stored metrics and evidence hash match. Changed configs
are rejected. Incomplete or failed conditions are rerun; partial optimizer checkpoints are
not silently treated as completed results. The training CLI separately supports explicit
`--resume checkpoint-path` for individual training runs.

Failures retain seed, method, level, model, configuration, and error. Retried failures remain in
`failure_history`; active failures remain in `failures`. Successful conditions are never dropped.
Interruptions may leave an unfinished condition; rerunning retries it rather than fabricating
an outcome. Completed evidence and local checkpoint files must be kept together for resume.

Outputs include `summary.json`, `report.md`, `aggregate.csv`, `per_seed.csv`, paired effects,
training diagnostics, raw generations, failure analysis, and plots. The runner creates
`results/final-study-evidence.zip`; weights are excluded. After successful execution, download
the ZIP before terminating Colab/Kaggle. Local checkpoint paths in metadata identify saves but
are not portable weights. Share adapters separately if required.

Exit states distinguish unavailable CUDA, failed capability qualification, incomplete conditions,
and completion. `execution_status.json` records the reason, hardware, and protocol hash.
Use a new output directory for an independent replication; do not change the frozen study to
favor a method after inspecting final-test results.
