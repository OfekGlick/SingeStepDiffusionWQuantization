#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="imagenet_64_script"           # Name of the job

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:1                                  # Request n gpus
#SBATCH --cpus-per-task=20                            # number of cpus per task and per node

#SBATCH -A cs
#SBATCH -p public
#SBATCH -w plato2

#SBATCH -o logs/slurm_%N_%j_out.txt      # stdout goes here
#SBATCH -e logs/slurm_%N_%j_err.txt      # stderr goes here

#SBATCH --mail-type=fail                              # send email if job fails
#SBATCH --mail-user=ofek.glick@campus.technion.ac.il

nvidia-smi
echo "Running from $(pwd)"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run fine tuning on SST-2 with Depth, half precision
python3 generate_image_64.py