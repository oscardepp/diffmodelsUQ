#!/bin/bash
#SBATCH --job-name=MC_naval                        # Job name
#SBATCH --partition=gengpu                          # Use the general GPU partition
#SBATCH --constraint=sxm                            # Use SXM-form GPUs (e.g., A100-SXM)
#SBATCH --gres=gpu:a100:1                           # Request 1 A100 GPU
#SBATCH -N 1                                        # Use 1 node
#SBATCH --cpus-per-task=32                          # 10 CPU threads
#SBATCH --mem=50G                                   # 50 GB memory
#SBATCH --time=8:00:00                             # Max wall time: 48 hours
#SBATCH --account=p32827                            # Your actual Quest project allocation
#SBATCH --output=/home/otd8990/logs/MC_naval.out     # Stdout log file path (customize)
#SBATCH --error=/home/otd8990/logs/MC_naval.err      # Stderr log file path (customize)


module purge

# Make sure Conda activates properly in batch mode
eval "$(conda shell.bash hook)"

# Activate your actual environment (use absolute path if needed)
conda activate mcdropout

# Now run your training script
#python mcdropoutdeepensemblepipelinecolab3.py

THEANO_FLAGS='allow_gc=False,device=gpu,floatX=float32' \
cd /home/otd8990/
srun --gres=gpu:a100:1 --ntasks=1 \
  python -u experiment_oscar.py --dir naval-propulsion-plant --epochx 15 --hidden 2

