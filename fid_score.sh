#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="fid_score"           # Name of the job

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:1                                  # Request n gpus
#SBATCH --cpus-per-task=5                            # number of cpus per task and per node

#SBATCH -A cs
#SBATCH -p public
#SBATCH -w plato2

#SBATCH -o fid_scores/quantized_model_config_w8_a8_actTrue_t60%N_%j_out.txt      # stdout goes here
#SBATCH -e fid_scores/quantized_model_config_w8_a8_actTrue_t60%N_%j_err.txt      # stderr goes here

#SBATCH --mail-type=fail                              # send email if job fails
#SBATCH --mail-user=ofek.glick@campus.technion.ac.il

nvidia-smi
echo "Running from $(pwd)"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

fidelity --gpu 0 --fid --input1 /home/ofek.glick/BRECQ/images_dataset --input2 /home/ofek.glick/BRECQ/quantized_model_config_w8_a8_actTrue_t60