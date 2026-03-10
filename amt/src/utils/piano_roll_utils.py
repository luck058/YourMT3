# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""
piano_roll_utils.py

Utilities for converting YourMT3's note representations into binary piano roll
tensors suitable for training and evaluating the FFNNPianoRollDecoder.

The main entry point for training is notes_to_piano_roll(), which converts a
list of Note objects (from utils/note_event_dataclasses.py) into a binary
tensor of shape (n_frames, n_instruments, n_pitches).

Data pipeline integration:
    The cleanest place to call notes_to_piano_roll() is in the data module,
    so that labels are pre-computed and batched alongside audio segments.
    See the docstring of notes_to_piano_roll() for the required Note fields.

    During validation, notes_dict['notes'] already provides Note objects, so
    the conversion can also be called directly inside validation_step().
"""

from typing import Dict, List, Tuple
import torch
import numpy as np

from utils.note_event_dataclasses import Note


def notes_to_piano_roll(
    notes,                          # List[Note] from utils.note_event_dataclasses
    start_time: float,              # segment start time in seconds
    duration: float,                # segment duration in seconds
    programs: List[int],            # MIDI program numbers to track, in index order
    n_frames: int,                  # number of output frames (= T', encoder output length)
    pitch_min: int = 21,            # lowest MIDI pitch (piano A0)
    pitch_max: int = 108,           # highest MIDI pitch (piano C8)
) -> torch.Tensor:
    """
    Convert a list of Note objects into a binary piano roll tensor.

    Each cell [t, i, p] is 1.0 if instrument i is playing pitch p at frame t,
    and 0.0 otherwise.

    The frame rate is inferred from n_frames and duration, so it automatically
    matches whatever temporal resolution the encoder produces — no hardcoded
    hop length or sample rate is needed here.

    Args:
        notes:      List of Note objects. Each Note must have:
                        .onset   (float) — note start time in seconds
                        .offset  (float) — note end time in seconds
                        .pitch   (int)   — MIDI pitch number
                        .program (int)   — MIDI program number (0–127)
                        .is_drum (bool)  — True if this is a drum note
        start_time: Absolute start time of this audio segment in seconds.
                    Notes before this time or after start_time + duration are
                    silently ignored.
        duration:   Duration of the audio segment in seconds.
                    Combined with n_frames this defines the frame rate.
        programs:   Ordered list of MIDI program numbers to include.
                    Index in this list → index in the instrument dimension.
                    Example: [0, 40] puts Piano at index 0, Violin at index 1.
                    Notes with programs not in this list are ignored.
        n_frames:   Number of frames in the output (must equal T' from the
                    encoder, i.e. enc_hs.shape[1]).
        pitch_min:  Lowest MIDI pitch included (inclusive). Default 21 = A0.
        pitch_max:  Highest MIDI pitch included (inclusive). Default 108 = C8.

    Returns:
        piano_roll: Float tensor of shape (n_frames, n_instruments, n_pitches).
                    Values are 0.0 or 1.0. Cast to bfloat16/float16 after
                    moving to device if memory is a concern.

    Example:
        # 2-second segment, encoder outputs 256 frames, piano + violin
        roll = notes_to_piano_roll(
            notes=track_notes,
            start_time=0.0,
            duration=2.048,
            programs=[0, 40],
            n_frames=256,
        )
        # roll.shape == (256, 2, 88)
    """
    n_pitches = pitch_max - pitch_min + 1
    n_instruments = len(programs)

    # Map MIDI program → instrument index for O(1) lookup
    program_to_idx: Dict[int, int] = {
        prog: idx for idx, prog in enumerate(programs)
    }

    # frames_per_second lets us map a note's onset/offset time to a frame index
    frames_per_second = n_frames / duration

    piano_roll = torch.zeros(n_frames, n_instruments, n_pitches, dtype=torch.float32)

    for note in notes:
        # ── Filter ──────────────────────────────────────────────────────────
        # Skip drum notes (they don't have meaningful pitch information)
        if note.is_drum:
            continue
        # Skip instruments we are not tracking
        if note.program not in program_to_idx:
            continue
        # Skip pitches outside the tracked range
        if note.pitch < pitch_min or note.pitch > pitch_max:
            continue

        # ── Map to tensor indices ────────────────────────────────────────────
        instr_idx = program_to_idx[note.program]
        pitch_idx = note.pitch - pitch_min

        # Convert absolute times to frame indices within this segment
        onset_sec  = note.onset  - start_time
        offset_sec = note.offset - start_time

        onset_frame  = int(onset_sec  * frames_per_second)
        offset_frame = int(offset_sec * frames_per_second)

        # Clamp to valid frame range.
        # Guarantee at least 1 frame is marked active even for very short notes.
        onset_frame  = max(0, min(onset_frame,  n_frames - 1))
        offset_frame = max(onset_frame + 1, min(offset_frame, n_frames))

        piano_roll[onset_frame:offset_frame, instr_idx, pitch_idx] = 1.0

    return piano_roll  # (n_frames, n_instruments, n_pitches)


def batch_notes_to_piano_roll(
    batch_notes,
    start_times: List[float],
    duration: float,
    programs: List[int],
    n_frames: int,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> torch.Tensor:
    """
    Batch version of notes_to_piano_roll.

    Iterates over a batch of note lists and stacks the resulting piano rolls
    into a single tensor, ready to be used as training labels.

    Args:
        batch_notes:  List of note lists, length B.
        start_times:  List of segment start times, length B.
        duration:     Common segment duration in seconds.
        programs:     See notes_to_piano_roll().
        n_frames:     See notes_to_piano_roll(). Must match enc_hs.shape[1].
        pitch_min:    See notes_to_piano_roll().
        pitch_max:    See notes_to_piano_roll().

    Returns:
        piano_rolls: Float tensor of shape (B, n_frames, n_instruments, n_pitches).
    """
    rolls = [
        notes_to_piano_roll(
            notes=notes,
            start_time=start_time,
            duration=duration,
            programs=programs,
            n_frames=n_frames,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
        )
        for notes, start_time in zip(batch_notes, start_times)
    ]

    # Stack along new batch dimension: list of (T', I, P) → (B, T', I, P)
    return torch.stack(rolls, dim=0)


def compute_pos_weight(
    piano_rolls: torch.Tensor,
    epsilon: float = 1e-6,
) -> float:
    """
    Estimate a scalar positive-class weight from a representative set of piano
    roll tensors.

    The weight is n_negative / n_positive, i.e. how many silent pitch-frames
    there are for every active pitch-frame. This value can then be passed as
    pos_weight in FFNNPianoRollDecoder.__init__().

    Use this by running it over your training set (or a large random sample)
    before starting training. Re-running it if you change the instrument list
    or dataset is recommended.

    Args:
        piano_rolls: Float tensor of any shape that ends in
                     (..., n_frames, n_instruments, n_pitches).
                     Typically produced by batch_notes_to_piano_roll().
        epsilon:     Small value to avoid division by zero when the dataset
                     has no active notes (should not happen in practice).

    Returns:
        pos_weight: A single float. Pass this to FFNNPianoRollDecoder as
                    pos_weight=compute_pos_weight(sample_rolls).

    Example:
        all_rolls = [batch_notes_to_piano_roll(...) for batch in train_loader]
        all_rolls = torch.cat(all_rolls, dim=0)
        weight = compute_pos_weight(all_rolls)
        print(f"Recommended pos_weight: {weight:.1f}")
        # Typical values: 20–100 depending on instrument density
    """
    n_positive = piano_rolls.sum().item()
    n_total    = piano_rolls.numel()
    n_negative = n_total - n_positive

    if n_positive < epsilon:
        raise ValueError(
            "No active notes found in the provided piano rolls. "
            "Check that your note lists and program numbers are correct."
        )

    pos_weight = n_negative / (n_positive + epsilon)
    return pos_weight


def piano_roll_to_note_list(
    piano_roll: torch.Tensor,
    programs: List[int],
    start_time: float,
    duration: float,
    pitch_min: int = 21,
    min_duration_frames: int = 1,
) -> List[Tuple[float, float, int, int]]:
    """
    Convert a binary piano roll back to a list of (onset, offset, pitch, program)
    tuples, suitable for evaluation or MIDI export.

    Consecutive active frames for the same pitch/instrument are merged into a
    single note. This is needed during inference to convert FFNN output back
    into the Note format used by YourMT3's evaluation pipeline.

    Args:
        piano_roll:          (n_frames, n_instruments, n_pitches) binary tensor.
                             Typically the output of FFNNPianoRollDecoder.predict().
        programs:            List of MIDI programs in instrument-index order.
                             Must match the programs used to create the labels.
        start_time:          Segment start time in seconds.
        duration:            Segment duration in seconds.
        pitch_min:           Lowest MIDI pitch in the roll.
        min_duration_frames: Notes shorter than this many frames are discarded.
                             Useful to suppress spurious single-frame activations.

    Returns:
        notes: List of (onset_sec, offset_sec, midi_pitch, midi_program) tuples.
    """
    n_frames, n_instruments, n_pitches = piano_roll.shape
    frames_per_second = n_frames / duration
    seconds_per_frame = duration / n_frames

    notes = []

    for instr_idx, program in enumerate(programs):
        for pitch_idx in range(n_pitches):
            midi_pitch = pitch_idx + pitch_min

            # Walk through frames, grouping consecutive active frames into notes
            in_note = False
            note_onset_frame = 0

            for t in range(n_frames):
                active = bool(piano_roll[t, instr_idx, pitch_idx].item())

                if active and not in_note:
                    # Note onset
                    in_note = True
                    note_onset_frame = t

                elif not active and in_note:
                    # Note offset
                    in_note = False
                    note_length_frames = t - note_onset_frame
                    if note_length_frames >= min_duration_frames:
                        onset_sec  = start_time + note_onset_frame * seconds_per_frame
                        offset_sec = start_time + t * seconds_per_frame
                        notes.append((onset_sec, offset_sec, midi_pitch, program))

            # Handle note that extends to the last frame
            if in_note:
                note_length_frames = n_frames - note_onset_frame
                if note_length_frames >= min_duration_frames:
                    onset_sec  = start_time + note_onset_frame * seconds_per_frame
                    offset_sec = start_time + duration
                    notes.append((onset_sec, offset_sec, midi_pitch, program))

    return notes


def compute_n_frames_from_audio_cfg(audio_cfg: dict) -> int:
    """
    Estimate the encoder output time dimension T' from the audio config.

    Uses the torchaudio MelSpectrogram formula with center=True (the default),
    which pads the signal by n_fft//2 on each side before computing frames.
    With center=True: T = floor(input_frames / hop_length) + 1.

    This approximation may differ from the actual enc_hs.shape[1] by ±1 if
    the spectrogram uses different padding. Always verify with the shape
    assertion in _ffnn_forward().

    Args:
        audio_cfg: the audio configuration dict from config/config.py.
    Returns:
        Estimated number of encoder output frames (T').
    """
    return audio_cfg["input_frames"] // audio_cfg["hop_length"] + 1


def piano_roll_tuples_to_notes(
    tuples: List[Tuple[float, float, int, int]],
) -> List[Note]:
    """
    Convert output of piano_roll_to_note_list() to Note objects accepted by
    compute_track_metrics().

    Args:
        tuples: List of (onset_sec, offset_sec, midi_pitch, midi_program)
                as returned by piano_roll_to_note_list().
    Returns:
        List[Note] with velocity=64 (a neutral dummy; metrics use onset/pitch/program).
    """
    return [
        Note(
            is_drum=False,
            program=program,
            onset=onset,
            offset=offset,
            pitch=pitch,
            velocity=64,
        )
        for onset, offset, pitch, program in tuples
    ]
