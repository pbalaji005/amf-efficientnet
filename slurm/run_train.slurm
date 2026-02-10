#!/bin/bash
#SBATCH --job-name=amf_train
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pb13070@uga.edu

# Load modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate your environment if using conda
# source activate myenv

# Run training
cd $SLURM_SUBMIT_DIR
python src/train.py \
    --data ./data \
    --run-name effnet_b3_baseline \
    --epochs 20 \
    --batch-size 32
