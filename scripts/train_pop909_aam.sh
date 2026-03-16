#!/bin/bash
#SBATCH --job-name=train_pop909_aam
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/train_pop909_aam_%j.out
#SBATCH --error=logs/train_pop909_aam_%j.err

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


# ── Step 1: Prepare raw datasets (extract POP909, download+downsample AAM) ──
echo "================================================"
echo "Step 1: Preparing raw datasets..."
echo "================================================"

python prepare_datasets.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed!"
    exit 1
fi
echo "Raw dataset preparation complete at: $(date)"

# ── Diagnostic: show AAM directory structure ─────────────────────────────────
echo "--- AAM directory structure (first 30 entries) ---"
find /disk/scratch/s2286943/mlp_dataset/AAM -maxdepth 3 | sort | head -30
echo "--- WAV count: $(find /disk/scratch/s2286943/mlp_dataset/AAM -name '*.wav' | wc -l) ---"
echo "--- FLAC count: $(find /disk/scratch/s2286943/mlp_dataset/AAM -name '*.flac' | wc -l) ---"
echo "--- MID count: $(find /disk/scratch/s2286943/mlp_dataset/AAM -name '*.mid' | wc -l) ---"

# ── Step 2: Preprocess POP909 and AAM into yourmt3_indexes ──────────────────
echo "================================================"
echo "Step 2: Preprocessing datasets..."
echo "================================================"

DATA_HOME="/disk/scratch/s2286943/mlp_dataset"

cd amt/src
python -c "
from utils.preprocess.preprocess_pop909 import preprocess_pop909
from utils.preprocess.preprocess_aam import preprocess_aam
preprocess_pop909('$DATA_HOME')
preprocess_aam('$DATA_HOME')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed!"
    exit 1
fi
echo "Preprocessing complete at: $(date)"

# Verify audio segment cache was created
echo "--- POP909 validation audio_segments files: $(find "$DATA_HOME/POP909-Dataset/POP909" -name "*_audio_segments.npy" | wc -l) ---"
echo "--- AAM validation audio_segments files: $(find "$DATA_HOME/AAM" -name "*_audio_segments.npy" | wc -l) ---"

# Copy index files from scratch to the local data dir (training uses ../../data/yourmt3_indexes) or create symlinks if you prefer. This is necessary because the training script expects the index files to be in a specific location.
mkdir -p /home/s2286943/YourMT3/data/yourmt3_indexes
cp "$DATA_HOME/yourmt3_indexes/"*.json /home/s2286943/YourMT3/data/yourmt3_indexes/
echo "--- Index files ---"
ls /home/s2286943/YourMT3/data/yourmt3_indexes/*.json 2>/dev/null || echo "WARNING: no JSON index files found!"

# ── Step 3: Fine-tune YourMT3+ (frozen encoder, AdamW, equal POP909+AAM) ───
echo "================================================"
echo "Step 3: Training..."
echo "================================================"

# Load the pretrained YourMT3+ checkpoint via exp_id@checkpoint syntax.
# The checkpoint must exist at:
#   amt/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt
python train.py \
    "train_pop909_aam" \
    -p 2024 \
    -d pop909_aam \
    -tk mt3_full_plus \
    -dec ffnn \
    -o AdamW \
    -lr 1e-4 \
    -bsz 4 8 \
    -e 50 \
    -fe True \
    -ps 0 0 \
    -pr bf16-mixed \
    -nw 0 \
    -g 1 \
    -wb disabled

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit 1
fi

echo "Job finished at: $(date)"
echo "Training complete!"
