#!/bin/bash
#SBATCH --job-name=DE_concrete
#SBATCH --partition=gengpu
#SBATCH --constraint=sxm
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --account=p32827
#SBATCH --output=/home/otd8990/logs/DE_concrete.out
#SBATCH --error=/home/otd8990/logs/DE_concrete.err

set -euo pipefail

# 1) Clean env and load matching CUDA/cuDNN for TF 2.12
module purge

# 2) (Optional) other HPC toolchains if required by your site
# module load gcc/XX

# 3) Activate your conda env
eval "$(conda shell.bash hook)"
conda activate deep_ensemble

# 4) Useful diagnostics in logs
echo "which python: $(which python)"
python -c "import tensorflow as tf; 
print('TF:', tf.__version__); 
print('Visible GPUs:', tf.config.list_physical_devices('GPU'))"

THEANO_FLAGS='allow_gc=False,device=gpu,floatX=float32' \
cd /home/otd8990

# 5) Run under srun (no stray backslashes / comments!)
srun --gres=gpu:a100:1 --ntasks=1 \
  python -u deep_ensemble.py --dataset concrete