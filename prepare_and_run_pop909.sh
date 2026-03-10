#!/bin/bash
#SBATCH --job-name=prep_pop909
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --nodelist=landonia11
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/prep_pop909_%j.out
#SBATCH --error=logs/prep_pop909_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate virtual environment
cd /home/s2286943/YourMT3
source venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p /disk/scratch/s2286943/mlp_dataset
mkdir -p /home/s2286943/pop909_midi_output

# Step 1: Prepare POP909 dataset
echo "================================================"
echo "Step 1: Preparing POP909 dataset..."
echo "================================================"
python prepare_datasets.py

# Check if dataset preparation succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed!"
    exit 1
fi

echo "Dataset preparation completed at: $(date)"

# Step 2: Check GPU is available
nvidia-smi

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 3: Run batch inference on POP909 dataset
echo "================================================"
echo "Step 2: Running inference on POP909..."
echo "================================================"
python batch_inference.py \
    --input-dir /disk/scratch/s2286943/mlp_dataset/POP909-Dataset/POP909/ \
    --output-dir /home/s2286943/pop909_midi_output/ \
    --device cuda \
    --skip-existing \
    --model-name "YMT3+"

# Check if inference succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed!"
    exit 1
fi

echo "Job finished at: $(date)"
echo "All tasks completed successfully!"
