#!/bin/bash
#SBATCH --job-name=eval_chords_pop909
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/eval_chords_pop909_%j.out
#SBATCH --error=logs/eval_chords_pop909_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /home/s2286943/YourMT3
source venv/bin/activate
mkdir -p logs

python evaluate_chords_pop909.py \
    --midi-dir /home/s2286943/pop909_midi_output/model_output \
    --pop909-dir /disk/scratch/s2286943/mlp_dataset/POP909-Dataset/POP909 \
    --index-file /home/s2286943/YourMT3/data/yourmt3_indexes/pop909_test_file_list.json \
    --out /home/s2286943/pop909_midi_output/chord_eval_results_pop909.json

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo "Job finished at: $(date)"
echo "Results saved to /home/s2286943/pop909_midi_output/chord_eval_results_pop909.json"
