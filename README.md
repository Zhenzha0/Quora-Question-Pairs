# Quora Question Pairs

Duplicate question detection on the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset using dense embeddings and classical ML classifiers.

## Overview

Questions are embedded with [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) and stored in a [Zarr](https://zarr.dev/) store (`embeddings.zarr`). Pairwise features derived from the embeddings (cosine similarity, Euclidean/Manhattan distance, lexical overlap, etc.) are fed into downstream classifiers.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

> **What is uv?**
> `uv` is a modern Python package and project manager (written in Rust) that replaces the `pip` + `venv` workflow. It is dramatically faster than `pip`, automatically creates and manages virtual environments for you, and pins dependencies in a lockfile (`uv.lock`) so everyone gets exactly the same versions. You never have to manually run `python -m venv` or worry about environment activation — `uv run` handles it all in one step.

Installing uv is easy and painless:

```bash
# Download and install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, when you run a script, uv will automatically download and set up the necessary virtual environments. Never worry about venvs ever again!

You run a script with 

```bash
uv run <script_name.py>
```

*Dependencies include `torch` (CUDA 12.6 wheel on Linux), `sentence-transformers`, `catboost`, `zarr`, `dvc[s3]`, and `scikit-learn`. See `pyproject.toml` for the full list.*

---

## Getting the Embeddings (DVC)
    
> **What is DVC?**
> DVC (Data Version Control) is like Git, but for large data files. Git isn't suited for storing gigabytes of binary data, so instead DVC stores a small pointer file (`embeddings.zarr.dvc`) in the repo and keeps the actual data on a separate remote storage (e.g. S3). Running `dvc pull` downloads the real data from that remote. This keeps the repo lightweight while still letting everyone reproduce the exact same data.

The pre-computed `embeddings.zarr` store (~3.7 GB, 18 k files) is tracked with DVC. The DVC remote config files will be shared separately. If you want to just get the embeddings, the same applies to you too!

All you have to do is:

```bash
# Place the provided .dvc/config (and credentials) in the repo, then:
uv run dvc pull
```

This pulls `embeddings.zarr` from the configured remote. There is no need to re-run `embed_quora.py` unless you want to regenerate the embeddings.
Once the pull finishes, `embeddings.zarr/` will be present and ready to use.

`embeddings.zarr` is necessary for the other scripts that rely on this to function.

### Zarr store layout

> **What is Zarr?**
> Zarr is a format for storing large numerical arrays on disk (or in the cloud). Think of it like `.npy`, but designed for data that won't fit comfortably in memory. Instead of one monolithic file, Zarr splits the array into many small compressed chunks, meaning you can load only the rows you need rather than the entire dataset at once. This makes random access fast and efficient, which matters when you have hundreds of thousands of embeddings.

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `ids` | `(N,)` | int64 | Unique question IDs, sorted ascending |
| `texts` | `(N,)` | str | Question text, aligned with `ids` |
| `embeddings` | `(N, 2560)` | float32 | Qwen3-Embedding-4B vectors |

Look up a question by ID with `np.searchsorted(store["ids"], qid)`.

---

## Scripts

| Script | What it does |
|--------|-------------|
| `embed_quora.py` | Downloads the Quora dataset via `kagglehub`, embeds every unique question with Qwen3-Embedding-4B (SDPA, batches of 128), and writes the result to `embeddings.zarr`. Logs progress every 30 s. |
| `catboost_thresh.py` | *(Legacy)* Monolithic script: loads embeddings, builds features, trains cosine baseline + Logistic Regression + CatBoost, evaluates, and saves errors. Kept as a reference. Use `experiments/` for new work. |
| `experiments/run_experiment.py` | **New entry point.** Plug-and-play experiment runner — see the [Experiment Harness](#experiment-harness) section below. |

---

## Running on the NUS SOC Cluster (Slurm)
*Follow this to do what I did!*

> **What is Slurm?**
> Slurm is a job scheduler used by HPC (high-performance computing) clusters. Because many users share the same machines, you don't run scripts directly — you submit a *job* describing what resources you need (GPUs, RAM, time limit) and Slurm queues it up and runs it when those resources are free. `slurmscript.sh` is that job description: the `#SBATCH` lines at the top are directives to Slurm (not regular shell comments), and the last line is the actual command to execute.
>
> Useful Slurm commands:
> | Command | What it does |
> |---------|-------------|
> | `sbatch slurm_gpu.sh <script.py> [args...]` | Submit a GPU job to the queue |
> | `sbatch slurm_cpu.sh <script.py> [args...]` | Submit a CPU-only job to the queue |
> | `squeue -u $USER` | Check the status of your queued/running jobs, $USER being your e1111.. username |
> | `scancel <jobid>` | Cancel a job |
> | `sinfo` | See available partitions and node status |

I mainly figured things out using this link:
https://www.comp.nus.edu.sg/~cs3210/student-guide/accessing/

The basic sequence of events are:
1. Create a SOC account from https://mysoc.nus.edu.sg/~newacct
2. Turn on The SOC Compute Cluster here: https://mysoc.nus.edu.sg/~myacct/services.cgi
3. Get the Forticlient VPN and connect to it
4. ssh into the cluster (the password is the password you created for the SOC account)
```bash
ssh <your e0111.. login>@xlogin.comp.nus.edu.sg
```

Once on the cluster, download uv, clone the repo and install dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/reallyjustalan/Quora-Question-Pairs.git
cd Quora-Question-Pairs
```
Notice how we didn't even have to bother installing dependencies? uv will do that automatically at runtime!

Pull the embeddings with DVC (place the provided config files first - you may copy files over using `cp`):

```bash
uv run dvc pull
```

There are two Slurm scripts (GPU and CPU-only) and a convenience wrapper `submit.sh` that handles log naming and folder organisation for you. **Use `submit.sh` — it is the recommended way to submit jobs.**

### Using `submit.sh` (recommended)

`submit.sh` automatically:
- derives the job name from the script filename (e.g. `embed_quora` from `embed_quora.py`)
- routes stdout/stderr to `logs/<script>_<jobid>.log` / `logs/<script>_<jobid>.err`
- creates the `logs/` directory if it doesn't exist

```bash
# GPU job (H200 GPU, 64 GB RAM, 8 CPUs, 2 h wall time)
./submit.sh gpu embed_quora.py

# CPU-only job (64 GB RAM, 16 CPUs, 2 h wall time)
./submit.sh cpu catboost_thresh.py

# Extra arguments are forwarded to the script
./submit.sh gpu embed_quora.py --batch-size 64
./submit.sh cpu catboost_thresh.py --threshold 0.5
```

Log files will appear under `logs/` and be named after the script, e.g.:
```
logs/embed_quora_12345.log
logs/embed_quora_12345.err
```

### Calling sbatch directly (advanced)

If you prefer to call `sbatch` manually, `%x` in the output path resolves to the job name, so you'll want to set `--job-name` yourself:

```bash
sbatch --job-name=embed_quora --output=logs/embed_quora_%j.log --error=logs/embed_quora_%j.err slurm_gpu.sh embed_quora.py
```

---

## Experiment Harness

All new model experiments live in `experiments/`. The key idea is that **model logic, feature building, and the evaluation pipeline are completely separate files** — you snap them together.

### How it works

```
embeddings.zarr  ──►  data.py  ──►  model.build_features()  ──►  run_experiment.py  ──►  report.py
                      (loads)        (each model picks its          (fixed pipeline:        (metrics.txt
                                      own feature set from           split → fit →           errors.csv
                                      features.py primitives)        predict)                all_experiments.csv)
```

- **`data.py`** — loads the zarr store + CSV into a list of `PairRecord` objects. No model logic here.
- **`features.py`** — pure feature functions (`embedding_features`, `lexical_features`, `all_features`, `matryoshka_embedding_features`, `matryoshka_all_features`). Each returns a `dict[str, float]` per pair. Models import whatever they need.
- **`models/<name>.py`** — each model owns its feature selection and any internal pre-processing (e.g. `StandardScaler` in `logreg_model.py`). Every model exposes three methods: `build_features()`, `fit()`, and `predict_proba()`.
- **`run_experiment.py`** — the fixed pipeline entry point. Change 3 lines to swap models.
- **`report.py`** — takes predictions and writes the full report (see below).

### Fixed split — fair comparison

The first run saves `experiments/splits/default_split.npz`. Every subsequent run loads those exact train/test indices. All models are therefore evaluated on **identical test rows**, making comparisons meaningful.

### Running an experiment

```bash
cd experiments
uv run run_experiment.py
```

To switch to a different model, edit the three lines in the `EXPERIMENT CONFIG` block at the top of `run_experiment.py`:

```python
from models.logreg_model import LogRegModel   # ← swap
MODEL = LogRegModel()                          # ← swap
EXPERIMENT_NAME = "logreg_all_features"        # ← unique name
```

For quick smoke-tests, set `MAX_ROWS = 50_000` in the same file.

### Report output

Each run produces:

```
experiments/results/
├── all_experiments.csv               ← one row per run (accuracy, precision, recall, F1, TP/FP/TN/FN)
└── <experiment_name>/
    ├── metrics.txt                   ← full metrics block + classification report
    ├── errors.csv                    ← every FP and FN with question text and predicted probability
    └── feature_importance.txt        ← ranked feature importances (if the model supports it)
```

### Adding a new model

1. Create `experiments/models/my_model.py` — copy any existing model as a template.
2. Implement `build_features(records)`, `fit(X_train, y_train)`, and `predict_proba(X_test)`.
3. Optionally implement `feature_importances() → dict[str, float]` for automatic importance reporting.
4. Import and instantiate it in `run_experiment.py`.

### Adding a new feature set

Add a function to `features.py` that takes a `PairRecord` and returns `dict[str, float]`, then reference it from your model's `_feature_fn`.

---

## Project Structure

```
.
├── embed_quora.py              # Step 1: embed all questions → embeddings.zarr
├── catboost_thresh.py          # (Legacy) monolithic classifier script
├── experiments/
│   ├── run_experiment.py       # Step 2: entry point for all model experiments
│   ├── data.py                 # Shared loader: zarr + CSV → list[PairRecord]
│   ├── features.py             # Primitive feature functions (embedding, lexical)
│   ├── report.py               # Metrics printer + results writer
│   ├── models/
│   │   ├── catboost_model.py
│   │   ├── logreg_model.py
│   │   ├── cosine_baseline.py
│   │   └── xgboost_model.py
│   ├── splits/                 # Auto-created; holds the fixed train/test split
│   └── results/                # Auto-created; one subfolder per experiment run
├── embeddings.zarr.dvc         # DVC pointer to the Zarr store
├── slurm_gpu.sh                # Slurm job script for GPU tasks (e.g. embedding)
├── slurm_cpu.sh                # Slurm job script for CPU-only tasks (e.g. classifiers)
├── submit.sh                   # Convenience wrapper: auto-names logs and routes to logs/
├── logs/                       # Created automatically; contains <script>_<jobid>.log/.err
├── pyproject.toml
└── uv.lock
```
