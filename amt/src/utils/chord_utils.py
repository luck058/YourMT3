"""
chord_utils.py — Frame-level chord classification utilities.

Vocabulary: 25 classes
  0–11  : major triads  (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
  12–23 : minor triads  (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
  24    : no chord / silence  ("N")
"""
import os
import re
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
N_CHORD_CLASSES = 25
NO_CHORD_IDX = 24

_ROOT_TO_IDX = {r: i for i, r in enumerate(ROOTS)}

# Enharmonic normalisation: flat → sharp equivalent
_ENHARMONIC = {
    'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
    'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
}

# Quality strings that collapse to a major triad
_MAJ_QUALITIES = {
    'maj', 'maj6', 'maj7', 'maj9', 'maj11', 'maj13',
    '7',           # dominant 7th has a major-triad base
    'sus2', 'sus4', '5', 'aug', 'add9', '6', '69',
}
# Quality strings that collapse to a minor triad
_MIN_QUALITIES = {
    'min', 'min6', 'min7', 'min9', 'min11', 'min13',
    'dim', 'dim7', 'hdim7', 'minmaj7',
}

# Human-readable names in class-index order (useful for logging)
CHORD_NAMES: List[str] = (
    [f'{r}:maj' for r in ROOTS] +
    [f'{r}:min' for r in ROOTS] +
    ['N']
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _normalize_root(root: str) -> Optional[str]:
    root = root.strip()
    root = _ENHARMONIC.get(root, root)
    return root if root in _ROOT_TO_IDX else None


def _quality_to_maj_or_min(quality: str) -> Optional[str]:
    q = quality.lower().strip()
    if q in _MAJ_QUALITIES:
        return 'maj'
    if q in _MIN_QUALITIES:
        return 'min'
    return None


def parse_chord_label(label: str) -> int:
    """
    Convert any chord label string to a class index (0–24).

    Handles two formats:
      POP909 : ``Root:quality``  e.g. ``B:maj7``, ``C#:min``, ``N``
      AAM    : ``Rootquality``   e.g. ``Fmaj``, ``Amin``, ``D#maj``

    Extended / altered chords are collapsed to the nearest triad:
      maj7, maj9, sus4, 7 (dominant) → maj
      min7, min9, dim, hdim7         → min
    Chords with unrecognised qualities map to ``NO_CHORD_IDX``.
    """
    label = label.strip().strip("'\"")
    if label in ('N', 'X', '', 'None', 'n'):
        return NO_CHORD_IDX

    # --- POP909 format: colon separator ---
    if ':' in label:
        root_str, quality = label.split(':', 1)
        quality = quality.split('/')[0]   # strip inversion, e.g. "maj/5" → "maj"
        root = _normalize_root(root_str)
        if root is None:
            return NO_CHORD_IDX
        maj_or_min = _quality_to_maj_or_min(quality)
        if maj_or_min is None:
            return NO_CHORD_IDX
        root_idx = _ROOT_TO_IDX[root]
        return root_idx if maj_or_min == 'maj' else root_idx + 12

    # --- AAM format: concatenated, e.g. "Fmaj", "A#min", "D#maj" ---
    m = re.match(r'^([A-G][#b]?)(.+)$', label)
    if m is None:
        return NO_CHORD_IDX
    root_str, quality = m.group(1), m.group(2)
    root = _normalize_root(root_str)
    if root is None:
        return NO_CHORD_IDX
    maj_or_min = _quality_to_maj_or_min(quality)
    if maj_or_min is None:
        return NO_CHORD_IDX
    root_idx = _ROOT_TO_IDX[root]
    return root_idx if maj_or_min == 'maj' else root_idx + 12


# ---------------------------------------------------------------------------
# Chord file loaders
# ---------------------------------------------------------------------------

def _load_pop909_chord_file(path: str) -> List[Tuple[float, float, int]]:
    """
    Parse a POP909 ``chord_audio.txt`` file.
    Format per line: ``start_sec  end_sec  label``
    Returns a list of ``(start_sec, end_sec, class_idx)`` tuples.
    """
    intervals: List[Tuple[float, float, int]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            intervals.append((start, end, parse_chord_label(label)))
    return intervals


def _load_aam_chord_file(path: str) -> List[Tuple[float, float, int]]:
    """
    Parse an AAM ``beatinfo.arff`` file.
    Data lines: ``start_time, bar, beat, 'chord_name'``
    Intervals are beat-to-beat; the last beat extends by 1 second.
    Returns a list of ``(start_sec, end_sec, class_idx)`` tuples.

    Note: AAM beatinfo.arff files omit the ``@DATA`` section marker;
    data lines begin immediately after the ``@ATTRIBUTE`` declarations.
    """
    entries: List[Tuple[float, str]] = []
    past_header = False

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            upper = line.upper()
            # Optional @DATA marker (not present in all AAM files)
            if upper == '@DATA':
                past_header = True
                continue
            # Skip ARFF header directives
            if upper.startswith('@'):
                past_header = False   # still in header
                continue
            # First non-@ non-empty line after header directives → data
            past_header = True
            parts = line.split(',')
            if len(parts) < 4:
                continue
            try:
                start = float(parts[0])
            except ValueError:
                continue
            chord_name = parts[3].strip().strip("'\"")
            entries.append((start, chord_name))

    if not entries:
        return []

    # Sort by start time (some beatinfo.arff files have out-of-order entries)
    entries.sort(key=lambda x: x[0])

    intervals: List[Tuple[float, float, int]] = []
    prev_end: Optional[float] = None
    for i, (start, chord_name) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else start + 1.0
        # Force contiguous: close any floating-point gap between beats
        if prev_end is not None:
            start = prev_end
        end = max(end, start + 1e-6)  # ensure strictly positive duration
        intervals.append((start, end, parse_chord_label(chord_name)))
        prev_end = end
    return intervals


def load_chord_intervals(path: str) -> List[Tuple[float, float, int]]:
    """
    Auto-detect dataset format by file extension and load chord intervals.
    ``.arff`` → AAM;  otherwise → POP909.
    Returns list of ``(start_sec, end_sec, class_idx)``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.arff':
        return _load_aam_chord_file(path)
    return _load_pop909_chord_file(path)


def chord_file_from_audio_file(audio_file: str) -> Optional[str]:
    """
    Derive the chord annotation file path from an audio file path.
    Returns ``None`` if the dataset is not recognised.

    POP909 : ``POP909/<id>/<id>.wav``  → ``POP909/<id>/chord_audio.txt``
    AAM    : ``AAM/<id>_mix_16k.wav``  → ``AAM/<id>_beatinfo.arff``
    """
    path = audio_file.replace('\\', '/')
    if 'POP909' in path:
        return os.path.join(os.path.dirname(path), 'chord_audio.txt')
    if 'AAM' in path:
        return path.replace('_mix_16k.wav', '_beatinfo.arff')
    return None


# ---------------------------------------------------------------------------
# Frame-level label generation
# ---------------------------------------------------------------------------

def chord_intervals_to_frame_labels(
    intervals: List[Tuple[float, float, int]],
    n_frames: int,
    duration: float,
    start_time: float = 0.0,
) -> np.ndarray:
    """
    Convert a list of chord intervals into a frame-level integer label array.

    Each frame is assigned the chord class with the greatest overlap in that
    frame's time window (majority-chord rule).  Frames with no coverage
    default to ``NO_CHORD_IDX``.

    Args:
        intervals:  ``(abs_start, abs_end, class_idx)`` tuples, absolute time
        n_frames:   number of encoder output frames for this segment
        duration:   segment duration in seconds
        start_time: absolute start time of this segment in seconds

    Returns:
        ``np.ndarray`` of shape ``(n_frames,)``, dtype ``int64``
    """
    frame_len = duration / n_frames
    seg_end = start_time + duration

    overlap = np.zeros((n_frames, N_CHORD_CLASSES), dtype=np.float64)

    for abs_start, abs_end, cls in intervals:
        # Clip interval to this segment
        clip_start = max(abs_start, start_time)
        clip_end = min(abs_end, seg_end)
        if clip_start >= clip_end:
            continue

        # Determine which frames this clipped interval touches
        f0 = int((clip_start - start_time) / frame_len)
        f1 = int((clip_end - start_time) / frame_len)
        f0 = max(0, min(f0, n_frames - 1))
        f1 = max(0, min(f1, n_frames - 1))

        for f in range(f0, f1 + 1):
            frame_start = start_time + f * frame_len
            frame_end = frame_start + frame_len
            ov = min(clip_end, frame_end) - max(clip_start, frame_start)
            if ov > 0.0:
                overlap[f, cls] += ov

    labels = np.full(n_frames, NO_CHORD_IDX, dtype=np.int64)
    has_coverage = overlap.sum(axis=1) > 0.0
    if has_coverage.any():
        labels[has_coverage] = np.argmax(overlap[has_coverage], axis=1)

    return labels


def frame_labels_to_intervals(
    frame_labels: np.ndarray,
    frame_duration: float,
    start_time: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a sequence of per-frame chord class indices into contiguous
    (intervals, labels) suitable for ``mir_eval.chord.evaluate``.

    Consecutive frames with the same class are merged into one interval.

    Args:
        frame_labels:   ``(n_frames,)`` int array of class indices (0–24)
        frame_duration: duration of each frame in seconds
        start_time:     absolute start time of the first frame in seconds

    Returns:
        intervals: ``np.ndarray`` of shape ``(N, 2)`` with ``[start, end]`` rows
        labels:    list of ``N`` mir_eval chord label strings (e.g. ``'C:maj'``, ``'N'``)
    """
    if len(frame_labels) == 0:
        return np.empty((0, 2), dtype=float), []

    intervals = []
    labels = []
    seg_start = start_time
    cur_cls = int(frame_labels[0])

    for f in range(1, len(frame_labels)):
        cls = int(frame_labels[f])
        if cls != cur_cls:
            seg_end = start_time + f * frame_duration
            intervals.append([seg_start, seg_end])
            labels.append(CHORD_NAMES[cur_cls])
            seg_start = seg_end
            cur_cls = cls

    # final segment
    seg_end = start_time + len(frame_labels) * frame_duration
    intervals.append([seg_start, seg_end])
    labels.append(CHORD_NAMES[cur_cls])

    return np.array(intervals, dtype=float), labels
