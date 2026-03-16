#!/bin/bash
#SBATCH --job-name=test_preprocessing
#SBATCH --partition=Teaching
#SBATCH --nodelist=damnii08
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/test_preprocessing_%j.out
#SBATCH --error=logs/test_preprocessing_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/s2286943/YourMT3
source venv/bin/activate
mkdir -p logs

export TMPDIR=/disk/scratch/s2286943/tmp
mkdir -p $TMPDIR

DATA_HOME="/disk/scratch/s2286943/mlp_dataset"

# ── Step 1: Prepare raw datasets ─────────────────────────────────────────────
echo "================================================"
echo "Step 1: Preparing raw datasets..."
echo "================================================"

python prepare_datasets.py
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed!"
    exit 1
fi
echo "Raw dataset preparation complete at: $(date)"

# ── Step 2: Preprocess POP909 and AAM (now with audio segment caching) ───────
echo "================================================"
echo "Step 2: Preprocessing datasets..."
echo "================================================"

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

# ── Step 3: Verify cached audio segment files were created ───────────────────
echo "================================================"
echo "Step 3: Verifying audio segment cache..."
echo "================================================"

echo "--- POP909 validation audio_segments files ---"
find "$DATA_HOME/POP909-Dataset/POP909" -name "*_audio_segments.npy" | wc -l
find "$DATA_HOME/POP909-Dataset/POP909" -name "*_audio_segments.npy" | head -5

echo "--- AAM validation audio_segments files ---"
find "$DATA_HOME/AAM" -name "*_audio_segments.npy" | wc -l
find "$DATA_HOME/AAM" -name "*_audio_segments.npy" | head -5

echo "--- Checking index JSONs contain audio_segments_file key ---"
python -c "
import json, glob
for path in glob.glob('$DATA_HOME/yourmt3_indexes/*_validation_file_list.json'):
    with open(path) as f:
        data = json.load(f)
    first = next(iter(data.values()))
    has_key = 'audio_segments_file' in first
    print(f'{path}: audio_segments_file present = {has_key}')
    if has_key:
        import os
        seg_file = first['audio_segments_file']
        exists = os.path.exists(seg_file)
        print(f'  -> {seg_file}')
        print(f'  -> file exists: {exists}')
        if exists:
            import numpy as np
            segs = np.load(seg_file, allow_pickle=False)
            print(f'  -> shape: {segs.shape}, dtype: {segs.dtype}')
"

# ── Step 4: Copy updated index files ─────────────────────────────────────────
echo "================================================"
echo "Step 4: Copying index files..."
echo "================================================"

mkdir -p /home/s2286943/YourMT3/data/yourmt3_indexes
cp "$DATA_HOME/yourmt3_indexes/"*.json /home/s2286943/YourMT3/data/yourmt3_indexes/
echo "--- Index files ---"
ls /home/s2286943/YourMT3/data/yourmt3_indexes/*.json 2>/dev/null || echo "WARNING: no JSON index files found!"

echo "================================================"
echo "Preprocessing test complete at: $(date)"
echo "================================================"
