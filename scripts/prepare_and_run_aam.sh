#!/bin/bash
#SBATCH --job-name=prep_aam
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --exclude=damnii[07-12],landonia[01-08,21-25]
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/prep_aam_%j.out
#SBATCH --error=logs/prep_aam_%j.err

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate virtual environment
cd /home/s2286943/YourMT3
source venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p /disk/scratch/s2286943/mlp_dataset/AAM
mkdir -p /home/s2286943/aam_midi_output

# Step 1: Prepare AAM dataset (download, extract, resample)
echo "================================================"
echo "Step 1: Preparing AAM dataset..."
echo "================================================"
echo "This will download ~several GB from Zenodo"
echo "May take 1-2 hours depending on network speed"

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

# Step 3: Run batch inference on AAM dataset
echo "================================================"
echo "Step 2: Running inference on AAM..."
echo "================================================"
python batch_inference.py \
    --input-dir /disk/scratch/s2286943/mlp_dataset/AAM/ \
    --output-dir /home/s2286943/aam_midi_output/ \
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
