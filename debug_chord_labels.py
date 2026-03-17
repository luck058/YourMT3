"""debug_chord_labels.py

Debug script to inspect chord labels produced by evaluate_chords_aam.py.
"""
import sys
sys.path.insert(0, '/home/s2286943/YourMT3')

from evaluate_chords_aam import midi_to_chord_estimates, load_chord_annotations_arff

song_id = '0914'
midi_path = f'/home/s2286943/aam_midi_output/model_output/{song_id}_mix_16k.mid'
arff_path = f'/disk/scratch/s2286943/mlp_dataset/AAM/annotations/{song_id}_beatinfo.arff'

est_intervals, est_labels = midi_to_chord_estimates(midi_path)
print('EST intervals (first 20):')
for i, (iv, lb) in enumerate(zip(est_intervals[:20], est_labels[:20])):
    print(f'  {iv[0]:.6f} -> {iv[1]:.6f}  ({lb})')

# Check for overlaps
print('\nChecking for overlaps in EST intervals:')
for i in range(1, len(est_intervals)):
    if est_intervals[i][0] < est_intervals[i-1][1]:
        print(f'  OVERLAP at index {i}: [{est_intervals[i-1]}] -> [{est_intervals[i]}]')

ref_intervals, ref_labels = load_chord_annotations_arff(arff_path)
print('\nREF intervals (first 10):')
for iv, lb in zip(ref_intervals[:10], ref_labels[:10]):
    print(f'  {iv[0]:.6f} -> {iv[1]:.6f}  ({lb})')
