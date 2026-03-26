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

INDEX_DIR=/home/s2286943/YourMT3/data/yourmt3_indexes
POP909_MIDI_DIR=/home/s2286943/baseline_chord_eval/pop909_midi
AAM_MIDI_DIR=/home/s2286943/baseline_chord_eval/aam_midi
mkdir -p "$POP909_MIDI_DIR"
mkdir -p "$AAM_MIDI_DIR"

# ── Step 1: Generate file lists from test index JSONs ────────────────────────
echo "================================================"
echo "Step 1: Generating test file lists from index JSONs..."
echo "================================================"

python -c "
import json
for dataset, key, out in [
    ('$INDEX_DIR/pop909_test_file_list.json', 'mix_audio_file', '/tmp/pop909_test_files.txt'),
    ('$INDEX_DIR/aam_test_file_list.json',    'mix_audio_file', '/tmp/aam_test_files.txt'),
]:
    with open(dataset) as f:
        d = json.load(f)
    with open(out, 'w') as f:
        for entry in d.values():
            f.write(entry[key] + '\n')
    print(f'Written {len(d)} files to {out}')
"

if [ $? -ne 0 ]; then
    echo "ERROR: File list generation failed!"
    exit 1
fi

# ── Step 2: Inference on POP909 test set ─────────────────────────────────────
echo "================================================"
echo "Step 2: Running inference on POP909 test set..."
echo "================================================"

cd /home/s2286943/YourMT3/amt/src

python ../../batch_inference.py \
    --input-dir /tmp \
    --file-list /tmp/pop909_test_files.txt \
    --output-dir "$POP909_MIDI_DIR" \
    --model-name "YMT3+" \
    --device cuda \
    --skip-existing

if [ $? -ne 0 ]; then
    echo "ERROR: POP909 inference failed!"
    exit 1
fi
echo "POP909 inference complete at: $(date)"

# ── Step 3: Inference on AAM test set ────────────────────────────────────────
echo "================================================"
echo "Step 3: Running inference on AAM test set..."
echo "================================================"

python ../../batch_inference.py \
    --input-dir /tmp \
    --file-list /tmp/aam_test_files.txt \
    --output-dir "$AAM_MIDI_DIR" \
    --model-name "YMT3+" \
    --device cuda \
    --skip-existing

if [ $? -ne 0 ]; then
    echo "ERROR: AAM inference failed!"
    exit 1
fi
echo "AAM inference complete at: $(date)"

# ── Step 4: Chord evaluation on POP909 ───────────────────────────────────────
echo "================================================"
echo "Step 4: Evaluating chords on POP909..."
echo "================================================"

cd /home/s2286943/YourMT3

python evaluate_chords_pop909.py \
    --midi-dir "$POP909_MIDI_DIR/model_output" \
    --pop909-dir /disk/scratch/s2286943/mlp_dataset/POP909-Dataset/POP909 \
    --index-file "$INDEX_DIR/pop909_test_file_list.json" \
    --out "$POP909_MIDI_DIR/chord_eval_results_pop909_baseline.json"

if [ $? -ne 0 ]; then
    echo "ERROR: POP909 chord evaluation failed!"
    exit 1
fi
echo "POP909 evaluation complete at: $(date)"

# ── Step 5: Chord evaluation on AAM ──────────────────────────────────────────
echo "================================================"
echo "Step 5: Evaluating chords on AAM..."
echo "================================================"

python evaluate_chords_aam.py \
    --midi-dir "$AAM_MIDI_DIR/model_output" \
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
