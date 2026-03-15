#!/bin/bash
#SBATCH --job-name=pop909_inference
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --nodelist=landonia11
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/pop909_%j.out
#SBATCH --error=logs/pop909_%j.err

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
mkdir -p /home/s2286943/pop909_midi_output

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run batch inference on POP909 dataset
python batch_inference.py \
    --input-dir /disk/scratch/s2286943/mlp_dataset/POP909-Dataset/POP909/ \
    --output-dir /home/s2286943/pop909_midi_output/ \
    --device cuda \
    --skip-existing \
    --model-name "YMT3+"

echo "Job finished at: $(date)"
