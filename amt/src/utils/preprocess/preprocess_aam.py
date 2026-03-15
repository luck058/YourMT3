"""preprocess_aam.py

Preprocesses the AAM (AI-generated Accompaniment Music) dataset for YourMT3.

Expected directory structure after prepare_datasets.py runs:
    {data_home}/AAM/
        0001_16k.wav       (downsampled from 0001.flac by prepare_datasets.py)
        0002_16k.wav
        ...
        1000_16k.wav
        0001-1000-midis/   (extracted from 0001-1000-midis.zip)
            0001.mid
            0002.mid
            ...
            1000.mid

If the MIDI files land in a different subdirectory after extraction, update
MIDI_SUBDIR below accordingly.

Writes:
    {data_home}/yourmt3_indexes/aam_{split}_file_list.json
    {data_home}/AAM/{id}_notes.npy
    {data_home}/AAM/{id}_note_events.npy
"""
import os
import json
import glob
from typing import Dict, Tuple

import numpy as np
from utils.audio import get_audio_file_info
from utils.midi import midi2note
from utils.note2event import note2note_event, mix_notes


# Splits: 800 train / 100 val / 100 test (songs 0001–1000)
N_TRAIN = 800
N_VAL = 100


def create_note_event_and_note_from_aam(midi_files: list, song_id: str) -> Tuple[Dict, Dict]:
    """Load and merge all per-instrument MIDI files for one AAM song."""
    merged_notes = []
    duration_sec = 0.0

    for midi_file in midi_files:
        is_drum_track = "Drum" in os.path.basename(midi_file)
        try:
            notes, dur = midi2note(
                midi_file,
                binary_velocity=True,
                ch_9_as_drum=True,
                force_all_drum=is_drum_track,
                force_all_program_to=None,  # preserve GM program from MIDI
                trim_overlap=True,
                fix_offset=True,
                quantize=True,
                verbose=0,
                minimum_offset_sec=0.01,
                drum_offset_sec=0.01,
            )
            merged_notes = mix_notes((merged_notes, notes), True, True, True)
            duration_sec = max(duration_sec, dur)
        except Exception as e:
            print(f"    WARNING: skipping {os.path.basename(midi_file)}: {e}")

    note_events = note2note_event(merged_notes, sort=True, return_activity=True)
    programs = list({n.program for n in merged_notes}) if merged_notes else [0]
    is_drum = [1 if p == 128 else 0 for p in programs]

    notes_dict = {
        "aam_id": song_id,
        "program": programs,
        "is_drum": is_drum,
        "duration_sec": duration_sec,
        "notes": merged_notes,
    }
    note_events_dict = {
        "aam_id": song_id,
        "program": programs,
        "is_drum": is_drum,
        "duration_sec": duration_sec,
        "note_events": note_events,
    }
    return notes_dict, note_events_dict


def preprocess_aam(data_home: str, dataset_name: str = "aam") -> None:
    """
    Preprocess AAM and write yourmt3_indexes file lists.

    Args:
        data_home: root data directory (e.g. /disk/scratch/s2286943/mlp_dataset)
        dataset_name: name used for index file naming (default: 'aam')
    """
    aam_dir = os.path.join(data_home, "AAM")
    output_index_dir = os.path.join(data_home, "yourmt3_indexes")
    os.makedirs(output_index_dir, exist_ok=True)

    # Audio files: named {id}_mix_16k.wav
    audio_files = sorted(glob.glob(os.path.join(aam_dir, "[0-9][0-9][0-9][0-9]_mix_16k.wav")))
    if not audio_files:
        raise FileNotFoundError(
            f"No AAM audio files found at {aam_dir}/*_mix_16k.wav\n"
            "Run prepare_datasets.py first to downsample AAM audio."
        )
    audio_map = {os.path.basename(f)[:4]: f for f in audio_files}

    # MIDI files: multiple per song, named {id}_{InstrumentName}.mid — group by song ID
    midi_map = {}
    for midi_file in sorted(glob.glob(os.path.join(aam_dir, "[0-9][0-9][0-9][0-9]_*.mid"))):
        song_id = os.path.basename(midi_file)[:4]
        midi_map.setdefault(song_id, []).append(midi_file)

    song_ids = sorted(set(audio_map.keys()) & set(midi_map.keys()))
    print(f"Found {len(song_ids)} AAM songs in {aam_dir}")

    train_ids = song_ids[:N_TRAIN]
    val_ids = song_ids[N_TRAIN:N_TRAIN + N_VAL]
    test_ids = song_ids[N_TRAIN + N_VAL:]

    splits = {"train": train_ids, "validation": val_ids, "test": test_ids}

    for split, ids in splits.items():
        file_list = {}
        print(f"\nProcessing {split} split ({len(ids)} songs)...")

        for i, song_id in enumerate(ids):
            audio_file = audio_map[song_id]
            song_midi_files = midi_map[song_id]

            try:
                fs, n_frames, _ = get_audio_file_info(audio_file)
            except Exception as e:
                print(f"  WARNING: skipping {song_id}, bad audio: {e}")
                continue
            if fs != 16000:
                print(f"  WARNING: {song_id} sample rate is {fs}Hz, expected 16000Hz.")

            print(f"  [{i+1}/{len(ids)}] Processing {song_id} ({len(song_midi_files)} MIDI tracks)...")

            try:
                notes, note_events = create_note_event_and_note_from_aam(song_midi_files, song_id)
            except Exception as e:
                print(f"  WARNING: skipping {song_id}, MIDI error: {e}")
                continue

            notes_file = os.path.join(aam_dir, f"{song_id}_notes.npy")
            note_events_file = os.path.join(aam_dir, f"{song_id}_note_events.npy")
            np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
            np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)

            file_list[i] = {
                "aam_id": song_id,
                "n_frames": n_frames,
                "mix_audio_file": audio_file,
                "notes_file": notes_file,
                "note_events_file": note_events_file,
                "midi_file": song_midi_files[0],  # reference MIDI (first instrument)
                "program": notes["program"],
                "is_drum": notes["is_drum"],
            }

        index_file = os.path.join(output_index_dir, f"{dataset_name}_{split}_file_list.json")
        with open(index_file, "w") as f:
            json.dump(file_list, f, indent=4)
        print(f"Saved {index_file} ({len(file_list)} entries)")

    print("\nAAM preprocessing complete.")


if __name__ == "__main__":
    from config.config import shared_cfg
    data_home = shared_cfg["PATH"]["data_home"]
    preprocess_aam(data_home=data_home)
