#!/bin/bash
#SBATCH --job-name=yourmt3_inference
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --nodelist=landonia11
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Activate virtual environment
cd /home/s2286943/YourMT3
source venv/bin/activate

# Create output directories
mkdir -p logs
mkdir -p batch_output

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run batch inference
python batch_inference.py \
    --input-dir examples/ \
    --output-dir batch_output/ \
    --device cuda \
    --skip-existing

echo "Job finished at: $(date)"
