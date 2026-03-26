#!/bin/bash
#SBATCH --job-name=baseline_chord_eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/baseline_chord_eval_%j.out
#SBATCH --error=logs/baseline_chord_eval_%j.err

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

# ── Step 1: Run test.py with t5 decoder to generate MIDI output ──────────────
echo "================================================"
echo "Step 1: Running test inference (t5 decoder)..."
echo "================================================"

cd amt/src

python test.py \
    "train_pop909_aam@model.ckpt" \
    -p 2024 \
    -d pop909_aam \
    -tk mt3_full_plus \
    -enc perceiver-tf \
    -dec t5 \
    -pr bf16-mixed \
    -g 1 \
    -w True \
    -wb disabled

if [ $? -ne 0 ]; then
    echo "ERROR: Test inference failed!"
    exit 1
fi
echo "Inference complete at: $(date)"

cd /home/s2286943/YourMT3

# ── Step 2: Find the MIDI output directories ──────────────────────────────────
# test.py writes MIDI to: ../logs/2024/train_pop909_aam/model_output_{dataset}_...
# We find the most recently created pop909 and aam output dirs.
LOG_DIR=/home/s2286943/YourMT3/amt/logs/2024/train_pop909_aam

POP909_MIDI_DIR=$(ls -td ${LOG_DIR}/model_output_pop909* 2>/dev/null | head -1)
AAM_MIDI_DIR=$(ls -td ${LOG_DIR}/model_output_aam* 2>/dev/null | head -1)

echo "POP909 MIDI dir: $POP909_MIDI_DIR"
echo "AAM MIDI dir:    $AAM_MIDI_DIR"

INDEX_DIR=/home/s2286943/YourMT3/data/yourmt3_indexes

# ── Step 3: Chord evaluation on POP909 ───────────────────────────────────────
echo "================================================"
echo "Step 3: Evaluating chords on POP909..."
echo "================================================"

if [ -z "$POP909_MIDI_DIR" ]; then
    echo "ERROR: No POP909 MIDI output directory found under $LOG_DIR"
    exit 1
fi

python evaluate_chords_pop909.py \
    --midi-dir "$POP909_MIDI_DIR" \
    --pop909-dir /disk/scratch/s2286943/mlp_dataset/POP909-Dataset/POP909 \
    --index-file "$INDEX_DIR/pop909_test_file_list.json" \
    --out "$POP909_MIDI_DIR/chord_eval_results_pop909_baseline.json"

if [ $? -ne 0 ]; then
    echo "ERROR: POP909 chord evaluation failed!"
    exit 1
fi
echo "POP909 evaluation complete at: $(date)"

# ── Step 4: Chord evaluation on AAM ──────────────────────────────────────────
echo "================================================"
echo "Step 4: Evaluating chords on AAM..."
echo "================================================"

if [ -z "$AAM_MIDI_DIR" ]; then
    echo "ERROR: No AAM MIDI output directory found under $LOG_DIR"
    exit 1
fi

python evaluate_chords_aam.py \
    --midi-dir "$AAM_MIDI_DIR" \
    --annotations-dir /disk/scratch/s2286943/mlp_dataset/AAM/annotations \
    --index-file "$INDEX_DIR/aam_test_file_list.json" \
    --out "$AAM_MIDI_DIR/chord_eval_results_aam_baseline.json"

if [ $? -ne 0 ]; then
    echo "ERROR: AAM chord evaluation failed!"
    exit 1
fi
echo "AAM evaluation complete at: $(date)"

echo "================================================"
echo "All done! Results saved to:"
echo "  POP909: $POP909_MIDI_DIR/chord_eval_results_pop909_baseline.json"
echo "  AAM:    $AAM_MIDI_DIR/chord_eval_results_aam_baseline.json"
echo "Job finished at: $(date)"
