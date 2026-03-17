"""evaluate_chords_pop909.py

Evaluates YourMT3+ MIDI output against POP909 ground truth chord annotations
using mir_eval.chord metrics.

Usage:
    python evaluate_chords_pop909.py \
        --midi-dir /home/s2286943/pop909_midi_output/model_output \
        --pop909-dir /disk/scratch/s2286943/mlp_dataset/POP909-Dataset/POP909 \
        --index-file /home/s2286943/YourMT3/data/yourmt3_indexes/pop909_test_file_list.json \
        --out chord_eval_results.json

chord_midi.txt format (already time-indexed in seconds):
    start_sec    end_sec    chord_label
"""
import os
import json
import argparse
import numpy as np
import mir_eval
import music21


def load_chord_annotations(chord_midi_txt: str):
    """Load chord_midi.txt -> (intervals, labels) for mir_eval."""
    intervals = []
    labels = []
    with open(chord_midi_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            chord_label = parts[2]
            if end_sec <= start_sec:
                continue
            intervals.append([start_sec, end_sec])
            labels.append(chord_label)
    return np.array(intervals, dtype=float), labels


def music21_chord_to_mir_eval(root, quality: str) -> str:
    """Convert music21 root + quality string to mir_eval label e.g. 'C:maj'."""
    if root is None:
        return 'N'
    root_name = root.name.replace('-', 'b')  # music21 uses B- -> Bb
    quality_map = {
        'major': 'maj',
        'minor': 'min',
        'diminished': 'dim',
        'augmented': 'aug',
        'dominant-seventh': '7',
        'major-seventh': 'maj7',
        'minor-seventh': 'min7',
        'diminished-seventh': 'dim7',
        'half-diminished-seventh': 'hdim7',
    }
    mir_quality = quality_map.get(quality, 'maj')
    return f"{root_name}:{mir_quality}"


def midi_to_chord_estimates(midi_path: str):
    """
    Use music21 to chordify a MIDI file.
    Returns (intervals, labels) for mir_eval.
    """
    score = music21.converter.parse(midi_path)
    chordified = score.chordify()

    intervals = []
    labels = []

    for elem in chordified.flatten().getElementsByClass(['Chord', 'Rest']):
        try:
            start_sec = float(elem.getOffsetInHierarchy(chordified))
            end_sec = start_sec + float(elem.seconds)
        except Exception:
            continue

        if end_sec <= start_sec:
            continue

        if isinstance(elem, music21.chord.Chord):
            label = music21_chord_to_mir_eval(elem.root(), elem.quality)
        elif isinstance(elem, music21.note.Rest):
            label = 'N'
        else:
            continue

        intervals.append([start_sec, end_sec])
        labels.append(label)

    return np.array(intervals, dtype=float), labels


def evaluate_song(midi_path: str, pop909_song_dir: str):
    """Evaluate one song. Returns dict of mir_eval scores or None on failure."""
    chord_txt = os.path.join(pop909_song_dir, 'chord_midi.txt')

    if not os.path.exists(chord_txt):
        print(f"  WARNING: missing chord_midi.txt in {pop909_song_dir}, skipping.")
        return None

    ref_intervals, ref_labels = load_chord_annotations(chord_txt)
    est_intervals, est_labels = midi_to_chord_estimates(midi_path)

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        print(f"  WARNING: empty intervals for {os.path.basename(midi_path)}, skipping.")
        return None

    try:
        # Trim both to the shorter duration so intervals align
        duration = min(ref_intervals[-1, 1], est_intervals[-1, 1])
        ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
            ref_intervals, ref_labels, 0, duration, 'N', 'N')
        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, 0, duration, 'N', 'N')
        scores = mir_eval.chord.evaluate(ref_intervals, ref_labels,
                                         est_intervals, est_labels)
    except Exception as e:
        print(f"  WARNING: mir_eval failed for {os.path.basename(midi_path)}: {e}")
        return None

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi-dir', required=True,
                        help='Directory containing YourMT3+ output .mid files')
    parser.add_argument('--pop909-dir', required=True,
                        help='POP909 root directory (.../POP909-Dataset/POP909/)')
    parser.add_argument('--index-file', required=True,
                        help='pop909_test_file_list.json from preprocessing')
    parser.add_argument('--out', default='chord_eval_results_pop909.json',
                        help='Output JSON for per-song and aggregate results')
    args = parser.parse_args()

    with open(args.index_file) as f:
        index = json.load(f)

    song_ids = [entry['pop909_id'] for entry in index.values()]
    print(f"Evaluating {len(song_ids)} test songs...")

    metric_keys = ['root', 'thirds', 'triads', 'tetrads', 'majmin', 'mirex']
    all_scores = {}

    for song_id in song_ids:
        midi_path = os.path.join(args.midi_dir, f"{song_id}.mid")
        if not os.path.exists(midi_path):
            print(f"  WARNING: no MIDI found for song {song_id}, skipping.")
            continue

        song_dir = os.path.join(args.pop909_dir, song_id)
        print(f"  [{song_id}] evaluating...")
        scores = evaluate_song(midi_path, song_dir)
        if scores is not None:
            all_scores[song_id] = {k: float(scores[k]) for k in metric_keys if k in scores}

    if not all_scores:
        print("No songs evaluated successfully.")
        return

    # Macro-average across songs
    aggregate = {}
    for k in metric_keys:
        vals = [s[k] for s in all_scores.values() if k in s]
        if vals:
            aggregate[k] = float(np.mean(vals))

    print("\n=== Aggregate Chord Evaluation (POP909 test set) ===")
    for k, v in aggregate.items():
        print(f"  {k:10s}: {v:.4f}")

    results = {'per_song': all_scores, 'aggregate': aggregate}
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
