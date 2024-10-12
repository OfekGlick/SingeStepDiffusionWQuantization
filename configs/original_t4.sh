#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="original_t4"           # Name of the job

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:1                                  # Request n gpus
#SBATCH --cpus-per-task=4                            # number of cpus per task and per node

#SBATCH -A cs
#SBATCH -p public
#SBATCH -w bruno2


#SBATCH -o logs/original_t4_%N_%j_out.txt      # stdout goes here
#SBATCH -e logs/original_t4_%N_%j_err.txt      # stderr goes here

#SBATCH --mail-type=fail                              # send email if job fails
#SBATCH --mail-user=ofek.glick@campus.technion.ac.il

nvidia-smi
echo "Running from $(pwd)"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run fine tuning on SST-2 with Depth, half precision
python3 ../main_imagenet.py \
--data_path '' \
--arch cd_imagenet64 \
--n_bits_w 8 \
--channel_wise \
--n_bits_a 8 \
--act_quant \
--base_path /home/ofek.glick/BRECQ \
--t_for_diffuser 4 \
--original
            