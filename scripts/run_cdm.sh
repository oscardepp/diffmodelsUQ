#!/bin/bash
#SBATCH --job-name=CDM_power
#SBATCH --partition=gengpu
#SBATCH --constraint=sxm                 # A100-SXM nodes on your cluster
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=p32827
#SBATCH --output=/home/otd8990/logs/CDM_power.out
#SBATCH --error=/home/otd8990/logs/CDM_power.err

# (1) Clean modules and load the cluster's CUDA driver-visible environment
module purge

# UTF-8 so arrows/symbols render correctly
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# (2) Activate Conda
eval "$(conda shell.bash hook)"
conda activate card_cdm

# (3) Helpful perf/env knobs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_ENABLE_ONEDNN_OPTS=0    # only matters if TF is present; safe otherwise
export PYTHONUNBUFFERED=1

# (4) Confirm GPU visibility in-job (will print to .out)
# python - <<'PY'
# import torch
# print("cuda.is_available:", torch.cuda.is_available())
# print("torch:", torch.__version__, "cuda:", torch.version.cuda)
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
# PY

# (5) Run your experiment (NO comments after backslashes!)
THEANO_FLAGS='allow_gc=False,device=gpu,floatX=float32' \
srun --gres=gpu:a100:1 --ntasks=1 \
  python -u cdm.py \
    --root /home/otd8990/UCI_Datasets \
    --dataset power-plant \
    --epochs 100 \
    --dropout 0.2 \
    --run_cdm
