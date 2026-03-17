#!/bin/bash
#SBATCH --job-name=test_pop909_aam_ffnn
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/test_pop909_aam_ffnn_%j.out
#SBATCH --error=logs/test_pop909_aam_ffnn_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

cd /home/s2286943/YourMT3
source venv/bin/activate
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/disk/scratch/s2286943/tmp
mkdir -p $TMPDIR

cd amt/src

python test.py \
    "train_pop909_aam@epoch=49-step=187500.ckpt" \
    -p 2024 \
    -d pop909_aam \
    -tk mt3_full_plus \
    -dec ffnn \
    -pr bf16-mixed \
    -g 1 \
    -w True \
    -wb disabled

if [ $? -ne 0 ]; then
    echo "ERROR: Testing failed!"
    exit 1
fi

echo "Job finished at: $(date)"
echo "Testing complete!"
