#!/bin/bash
#SBATCH --job-name=SQR_naval
#SBATCH --partition=gengpu
#SBATCH --constraint=sxm                 # A100-SXM nodes on your cluster
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=p32827
#SBATCH --output=/home/otd8990/logs/SQR_naval.out
#SBATCH --error=/home/otd8990/logs/SQR_naval.err

# (1) Clean modules
module purge

# UTF-8 so arrows/symbols render correctly
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# (2) Activate Conda
eval "$(conda shell.bash hook)"
conda activate card_cdm   # or another env where sqr_uci.py + torch + sklearn live

# (3) Helpful perf/env knobs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_ENABLE_ONEDNN_OPTS=0    # harmless if TF not installed
export PYTHONUNBUFFERED=1

# (4) Optional: quick GPU sanity check (commented out)
# python - <<'PY'
# import torch
# print("cuda.is_available:", torch.cuda.is_available())
# print("torch:", torch.__version__, "cuda:", torch.version.cuda)
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
# PY

# (5) Run SQR (no comments after backslashes)
srun --gres=gpu:a100:1 --ntasks=1 \
  python -u bootstrap/sqr.py \
    --root /home/otd8990/UCI_Datasets \
    --dataset naval-propulsion-plant \
    --epochs 2000 \
    --n_hidden_layers 1 \
    --n_hidden_units 64 \
    --batch 64 \
    --n_ens 1 \
    --alpha 0.05
