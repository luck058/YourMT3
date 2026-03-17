"""evaluate_chords_aam.py

Evaluates YourMT3+ MIDI output against AAM ground truth chord annotations
using mir_eval.chord metrics.

AAM chord annotations come from {id}_segments.arff files, which contain
segment-level key/chord labels (e.g. 'Fmaj', 'G#min').

Usage:
    python evaluate_chords_aam.py \
        --midi-dir /home/s2286943/aam_midi_output/model_output \
        --annotations-dir /disk/scratch/s2286943/mlp_dataset/AAM/annotations \
        --index-file /home/s2286943/YourMT3/data/yourmt3_indexes/aam_test_file_list.json \
        --out /home/s2286943/aam_midi_output/chord_eval_results_aam.json
"""
import os
import re
import json
import argparse
import numpy as np
import mir_eval
import music21


def parse_arff_key_to_mir_eval(key_str: str) -> str:
    """Convert AAM key string to mir_eval label. e.g. 'Fmaj' -> 'F:maj', 'G#min' -> 'G#:min'."""
    if not key_str or key_str == '':
        return 'N'
    # Match root (e.g. F, G#, Bb) and quality (maj/min)
    m = re.match(r"^([A-G][#b]?)(maj|min)$", key_str)
    if not m:
        return 'N'
    root, quality = m.group(1), m.group(2)
    return f"{root}:{quality}"


def load_chord_annotations_arff(beatinfo_arff: str):
    """
    Load {id}_beatinfo.arff -> (intervals, labels) for mir_eval.
    Format: start_time, bar, beat, chord_label
    End time of each beat = start time of the next beat.
    """
    entries = []  # list of (start_sec, chord_str)

    in_data = False
    with open(beatinfo_arff) as f:
        for line in f:
            line = line.strip()
            if line.upper() == '@DATA':
                in_data = True
                continue
            if line.startswith('@') or not line:
                continue
            if not in_data:
                # some arff files omit @DATA, start reading once we see numeric data
                if not line[0].isdigit():
                    continue
                in_data = True

            parts = [p.strip().strip("'") for p in line.split(',')]
            if len(parts) < 4:
                continue

            start_sec = float(parts[0])
            chord_str = parts[3]
            entries.append((start_sec, chord_str))

    if not entries:
        return np.array([], dtype=float).reshape(0, 2), []

    # Sort by start time (some beatinfo.arff files have out-of-order entries)
    entries.sort(key=lambda x: x[0])

    # Estimate duration of last beat from average beat length
    if len(entries) > 1:
        avg_beat = (entries[-1][0] - entries[0][0]) / (len(entries) - 1)
    else:
        avg_beat = 0.5
    last_end = entries[-1][0] + avg_beat

    # Build intervals, clipping each end to the next start to avoid overlaps
    intervals = []
    labels = []
    prev_end = None
    for i, (start_sec, chord_str) in enumerate(entries):
        end_sec = entries[i + 1][0] if i + 1 < len(entries) else last_end
        if prev_end is not None:
            start_sec = prev_end  # force contiguous
        end_sec = max(end_sec, start_sec + 1e-6)  # ensure strictly positive duration
        intervals.append([start_sec, end_sec])
        labels.append(parse_arff_key_to_mir_eval(chord_str))
        prev_end = end_sec

    return np.array(intervals, dtype=float), labels


def music21_chord_to_mir_eval(root, quality: str) -> str:
    """Convert music21 root + quality string to mir_eval label e.g. 'C:maj'."""
    if root is None:
        return 'N'
    root_name = root.name.replace('-', 'b')
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
    Returns (intervals, labels) for mir_eval, with contiguous intervals.
    """
    score = music21.converter.parse(midi_path)
    chordified = score.chordify()

    def offset_to_seconds(offset_qn, tempo_map):
        secs = 0.0
        prev_offset, prev_tempo = tempo_map[0]
        for (cur_offset, cur_tempo) in tempo_map[1:]:
            if offset_qn <= cur_offset:
                break
            secs += (min(offset_qn, cur_offset) - prev_offset) * (60.0 / prev_tempo)
            prev_offset, prev_tempo = cur_offset, cur_tempo
        secs += (offset_qn - prev_offset) * (60.0 / prev_tempo)
        return secs

    tempo_map = []
    for mm in chordified.flatten().getElementsByClass('MetronomeMark'):
        tempo_map.append((mm.offset, mm.number))
    if not tempo_map:
        tempo_map = [(0.0, 120.0)]

    entries = []
    for elem in chordified.flatten().getElementsByClass(['Chord', 'Rest']):
        offset_qn = float(elem.offset)
        dur_qn = float(elem.duration.quarterLength)
        if dur_qn <= 0:
            continue
        start_sec = offset_to_seconds(offset_qn, tempo_map)
        end_sec = offset_to_seconds(offset_qn + dur_qn, tempo_map)
        if isinstance(elem, music21.chord.Chord):
            label = music21_chord_to_mir_eval(elem.root(), elem.quality)
        else:
            label = 'N'
        entries.append((start_sec, end_sec, label))

    if not entries:
        return np.array([], dtype=float).reshape(0, 2), []

    entries.sort(key=lambda x: x[0])

    # Clip each entry's end to the next entry's start to remove overlaps,
    # then deduplicate and force contiguous
    clipped = []
    for i, (start, end, label) in enumerate(entries):
        if i + 1 < len(entries):
            end = min(end, entries[i + 1][0])
        clipped.append((start, end, label))

    intervals = []
    labels = []
    prev_end = None
    for start, end, label in clipped:
        if prev_end is not None:
            start = prev_end  # force contiguous
        if end - start < 1e-6:
            continue
        intervals.append([start, end])
        labels.append(label)
        prev_end = end

    return np.array(intervals, dtype=float), labels


def evaluate_song(midi_path: str, segments_arff: str):
    """Evaluate one song. Returns dict of mir_eval scores or None on failure."""
    if not os.path.exists(segments_arff):
        print(f"  WARNING: missing {segments_arff}, skipping.")
        return None

    ref_intervals, ref_labels = load_chord_annotations_arff(segments_arff)
    est_intervals, est_labels = midi_to_chord_estimates(midi_path)

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        print(f"  WARNING: empty intervals for {os.path.basename(midi_path)}, skipping.")
        return None

    try:
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
    parser.add_argument('--annotations-dir', required=True,
                        help='AAM annotations directory (contains {id}_segments.arff)')
    parser.add_argument('--index-file', required=True,
                        help='aam_test_file_list.json from preprocessing')
    parser.add_argument('--out', default='chord_eval_results_aam.json',
                        help='Output JSON for per-song and aggregate results')
    args = parser.parse_args()

    with open(args.index_file) as f:
        index = json.load(f)

    song_ids = [entry['aam_id'] for entry in index.values()]
    print(f"Evaluating {len(song_ids)} test songs...")

    metric_keys = ['root', 'thirds', 'triads', 'tetrads', 'majmin', 'mirex']
    all_scores = {}

    for song_id in song_ids:
        midi_path = os.path.join(args.midi_dir, f"{song_id}_mix_16k.mid")
        if not os.path.exists(midi_path):
            print(f"  WARNING: no MIDI found for song {song_id}, skipping.")
            continue

        segments_arff = os.path.join(args.annotations_dir, f"{song_id}_beatinfo.arff")
        print(f"  [{song_id}] evaluating...")
        scores = evaluate_song(midi_path, segments_arff)
        if scores is not None:
            all_scores[song_id] = {k: float(scores[k]) for k in metric_keys if k in scores}

    if not all_scores:
        print("No songs evaluated successfully.")
        return

    aggregate = {}
    for k in metric_keys:
        vals = [s[k] for s in all_scores.values() if k in s]
        if vals:
            aggregate[k] = float(np.mean(vals))

    print("\n=== Aggregate Chord Evaluation (AAM test set) ===")
    for k, v in aggregate.items():
        print(f"  {k:10s}: {v:.4f}")

    results = {'per_song': all_scores, 'aggregate': aggregate}
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()
