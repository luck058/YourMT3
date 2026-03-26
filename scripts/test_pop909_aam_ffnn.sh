#!/bin/bash
#SBATCH --job-name=test_pop909_aam
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/test_pop909_aam_%j.out
#SBATCH --error=logs/test_pop909_aam_%j.err

DEC="${1:-ffnn}"
EXP_ID="train_pop909_aam_${DEC}"
echo "Decoder:       $DEC"
echo "Experiment ID: $EXP_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

cd /home/s2286943/YourMT3
source venv/bin/activate
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TMPDIR=/disk/scratch/s2286943/tmp
mkdir -p $TMPDIR

cd amt/src

python test.py \
    "${EXP_ID}@last.ckpt" \
    -p 2024 \
    -d pop909_aam \
    -tk mt3_full_plus \
    -enc perceiver-tf \
    -dec "$DEC" \
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
