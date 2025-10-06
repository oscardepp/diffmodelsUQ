This repository contains implementations and scripts for evaluating uncertainty quantification (UQ) methods on UCI regression datasets. The codebase includes:
- Conditional Diffusion Models (CDM): `cdm.py`, `cdm_split_cqr.py`
- Deep Ensembles: `deep_ensemble.py`
- Monte Carlo Dropout: `mcdropout.py` and related scripts
- Neural net model code used by these methods: `net.py`
- Example experiment/job scripts for Northwestern QUEST HPC: `scripts/*.sh`
- Prepared dataset splits under `UCI_Datasets/` (one folder per dataset)

This README explains how to set up the environment, run experiments locally, and submit jobs to the Northwestern QUEST cluster. The Python scripts include baked-in parameters (see each file for exact defaults). The run scripts in `scripts/` demonstrate recommended SLURM settings for QUEST.

## Quick links

- Main training scripts: `cdm.py`, `cdm_split_cqr.py`, `deep_ensemble.py`, `mcdropout.py`
- Inference / helper code: `net.py`
- Job submission examples: `scripts/run_cdm.sh`, `scripts/run_job_cdm.sh`, `scripts/run_jobde.sh`, `scripts/run_jobmcd.sh`
- Datasets: `UCI_Datasets/` (contains data/ subfolders for each dataset)

## Repository contract (inputs / outputs)

- Inputs: dataset folders under `UCI_Datasets/<dataset>/data/` and the Python scripts. Most scripts accept CLI args (see `--help`) but include baked defaults in the code and the `scripts/` wrappers.
- Outputs: model checkpoints, logs, and results written relative to the working directory or dataset root (see each script for exact paths). SLURM job scripts redirect stdout/stderr to `/home/otd8990/logs/*.out` and `.err` in examples — customize those paths for your account.

## Environment setup

These experiments were run with Conda environments. Example environment YAMLs are provided in `envs/`:

- `envs/env-de.yaml` (deep ensemble)
- `envs/env-mcdropout.yaml` (mc-dropout)
- `envs/env-sc-cqr-cdm.yaml` (CDM / conditional diffusion)

Recommended steps (local or on QUEST):

1. Install Miniconda / Anaconda on your machine or ensure Conda is available on QUEST.
2. Create and activate the environment that matches the experiment you want to run. Example:

```bash
# create the CDM environment
conda env create -f envs/env-sc-cqr-cdm.yaml
conda activate card_cdm

# or create the deep ensemble env
conda env create -f envs/env-de.yaml
conda activate deep_ensemble

# or the mc-dropout env
conda env create -f envs/env-mcdropout.yaml
conda activate mcdropout
```

If you do not use Conda you can manually install the main dependencies (PyTorch or TensorFlow as required, numpy, scipy, pandas, scikit-learn). Use the environment YAMLs as a starting point.

Notes for QUEST:

- QUEST provides modules and GPUs; the `scripts/*.sh` job scripts assume you will activate a Conda environment inside the job (they use `eval "$(conda shell.bash hook)"` then `conda activate <env>`).
- Adjust `#SBATCH` headers (account/project, output paths, partition, time) to match your allocation.

## Dataset layout

Each dataset under `UCI_Datasets/<name>/data/` contains:

- `data.txt` — the full dataset (features + target) in a whitespace-separated format
- `index_train_*.txt`, `index_test_*.txt` — pre-split train/test indices for cross-validation folds
- `index_features.txt`, `index_target.txt` — column indices for features and target
- `n_epochs.txt`, `n_hidden.txt`, `n_splits.txt`, `tau_values.txt`, `dropout_rates.txt` — utility files with default hyperparameters used by earlier experiments

If you wish to use your own dataset, create the same structure (or modify the scripts to point to a different path). The training scripts accept a `--root` and `--dataset` CLI args which point to the dataset root and dataset name.

## Running locally (interactive / development)

You can run each experiment script directly from the command line. Example using CDM:

```bash
# from repository root
conda activate card_cdm
python -u cdm.py --root /absolute/path/to/UCI_Datasets --dataset power-plant --epochs 100 --dropout 0.2 --run_cdm
```

For split/CQR variant:

```bash
conda activate card_cdm
python -u cdm_split_cqr.py --root /absolute/path/to/UCI_Datasets --dataset YearPredictionMSD --epochs 200 --dropout 0.15 --run_cdm
```

Deep ensemble example:

```bash
conda activate deep_ensemble
python -u deep_ensemble.py --dataset concrete
```

Monte Carlo dropout example (script names vary; see `scripts/run_jobmcd.sh` for usage):

```bash
conda activate mcdropout
python -u experiment_oscar.py --dir naval-propulsion-plant --epochx 15 --hidden 2
```

Notes:

- Use absolute paths in `--root` when running on cluster jobs to avoid ambiguity.
- Use `python -u` to get unbuffered stdout so logs appear live in job output files.
- Check the top of each Python file to see baked-in defaults. You can pass CLI args to override them when provided.

## Submitting jobs on Northwestern QUEST (SLURM)

The `scripts/` directory contains example SLURM submission scripts which were used to run experiments on QUEST. They include recommended resource requests and environment activation steps. Customize them for your account and dataset paths.

Example: submit the CDM job

```bash
# make the script executable (once)
chmod +x scripts/run_cdm.sh

# submit it to SLURM
sbatch scripts/run_cdm.sh
```

What the job scripts do (high level):

- Load a clean module environment with `module purge` and set UTF-8
- Activate a Conda environment inside the batch script
- Set helpful thread-related env vars: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
- Run the desired training script using `srun` and request GPUs with `--gres=gpu:a100:1`
- Redirect stdout/stderr to log files (paths are set in the `#SBATCH --output` and `--error` directives)

Customize these fields before running on QUEST:

- `#SBATCH --account=` — set to your QUEST allocation/project
- `#SBATCH --output` and `--error` — set to a directory you own (example uses `/home/otd8990/logs/`)
- `--partition`, `--constraint`, and `--gres` — match the hardware you want (A100-SXM shown in examples)

Inspecting GPU in job logs

If you need to verify the job sees GPUs, the job scripts include a commented-out diagnostic snippet that prints CUDA availability and device names. Enable that snippet (remove comment markers) to make the job write GPU details to the .out log.

## Where parameters live

- Most script-level hyperparameters (epochs, dropout rates, dataset names) are provided via argparse CLI flags in the Python files but have defaults set in the source. Check the top of each Python file (`cdm.py`, `cdm_split_cqr.py`, `deep_ensemble.py`, `mcdropout.py`, `experiment_oscar.py`) to see the baked-in defaults.
- The `UCI_Datasets/*/data/` folders include ancillary files like `n_epochs.txt` and `dropout_rates.txt` that were used by experiments; those help reproduce previously reported runs.

## Logging, outputs and checkpoints

- SLURM scripts use `#SBATCH --output` and `--error` to capture stdout/stderr; change those to point to a logs folder in your home on QUEST (or a shared project folder).
- The Python scripts typically print progress to stdout and may save model checkpoints to disk. Search the Python files for `save`, `checkpoint`, or `torch.save` / `tf.train.Checkpoint` to find where models are persisted.

## Troubleshooting

- Conda activation fails in batch jobs: ensure your login shell supports conda and that `eval "$(conda shell.bash hook)"` is present in the script before `conda activate`.
- GPU not visible in job: check `--gres` and `--constraint` in your SBATCH header, and confirm your partition supports GPUs. Add the CUDA diagnostic snippet to the script to print device info.
- Missing packages: create the Conda environment from the provided YAMLs under `envs/` or install packages via pip inside the activated environment.

## Reproducing results

1. Pick a dataset under `UCI_Datasets/` and confirm the `--root`/`--dataset` path you will use.
2. Create/activate the matching Conda environment from `envs/`.
3. Run the script locally for a quick smoke test with reduced epochs or submit a job to QUEST using a `scripts/*.sh` wrapper.

## Files changed / key files

- `cdm.py`, `cdm_split_cqr.py`: conditional diffusion model training and split/CQR experiments
- `deep_ensemble.py`: deep ensemble training
- `mcdropout.py`: MC-dropout training
- `net.py`: neural network model definitions used by the above scripts
- `scripts/*.sh`: SLURM job templates for QUEST

## Next steps and optional improvements

- Add explicit output/checkpoint directory CLI args to each Python script to make experiment artifacts easier to collect.
- Add a small wrapper to collect results and summarize metrics after a job finishes.
- Add unit tests / smoke tests that run one epoch on a tiny synthetic dataset so users can validate the installation quickly.

---

If you want, I can now:

1. Update the SLURM scripts to use placeholders (e.g., ${USER}, ${HOME}/logs) so they're easier to reuse.
2. Add short examples of how to change baked-in parameters inside the Python files.
3. Create a small smoke-test script and add it to CI.

Tell me which of these you'd like and I'll implement it.
