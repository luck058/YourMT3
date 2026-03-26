"""preprocess_pop909.py

Preprocesses the POP909 dataset for training with YourMT3.

Expected directory structure:
    {data_home}/POP909-Dataset/POP909/
        001/
            001.wav         (mix audio, expected at 16kHz)
            melody.mid
            bridge.mid
            accomp.mid
        002/ ...
        909/ ...

Writes:
    {data_home}/yourmt3_indexes/pop909_{split}_file_list.json
    {data_home}/POP909-Dataset/POP909/{id}/{id}_notes.npy
    {data_home}/POP909-Dataset/POP909/{id}/{id}_note_events.npy
"""
import os
import json
import glob
import random
from typing import Dict, List, Tuple

import numpy as np
from einops import rearrange
from utils.audio import get_audio_file_info, load_audio_file, slice_padded_array
from utils.midi import midi2note
from utils.note2event import note2note_event, mix_notes

SEG_LEN_FRAME = 32767  # must match audio_cfg["input_frames"]


# All three POP909 MIDI tracks are piano (program 0)
POP909_PROGRAM = 0

# Deterministic split: 700 train / 100 val / 109 test (sorted by song id)
N_TRAIN = 700
N_VAL = 100


def create_note_event_and_note_from_pop909(song_dir: str, song_id: str) -> Tuple[Dict, Dict]:
    """Load and merge melody/bridge/accomp MIDI into notes and note_events."""
    midi_files = [
        os.path.join(song_dir, "melody.mid"),
        os.path.join(song_dir, "bridge.mid"),
        os.path.join(song_dir, "accomp.mid"),
    ]

    merged_notes = []
    duration_sec = 0.0

    for mid_file in midi_files:
        if not os.path.exists(mid_file):
            continue
        notes, dur_sec = midi2note(
            mid_file,
            binary_velocity=True,
            ch_9_as_drum=False,
            force_all_drum=False,
            force_all_program_to=POP909_PROGRAM,
            trim_overlap=True,
            fix_offset=True,
            quantize=True,
            verbose=0,
            minimum_offset_sec=0.01,
            drum_offset_sec=0.01,
        )
        merged_notes = mix_notes((merged_notes, notes), True, True, True)
        duration_sec = max(duration_sec, dur_sec)

    note_events = note2note_event(merged_notes, sort=True, return_activity=True)

    notes_dict = {
        "pop909_id": song_id,
        "program": [POP909_PROGRAM],
        "is_drum": [0],
        "duration_sec": duration_sec,
        "notes": merged_notes,
    }
    note_events_dict = {
        "pop909_id": song_id,
        "program": [POP909_PROGRAM],
        "is_drum": [0],
        "duration_sec": duration_sec,
        "note_events": note_events,
    }
    return notes_dict, note_events_dict


def preprocess_pop909(data_home: str, dataset_name: str = "pop909") -> None:
    """
    Preprocess POP909 and write yourmt3_indexes file lists.

    Args:
        data_home: root data directory (e.g. /disk/scratch/s2286943/mlp_dataset)
        dataset_name: name used for index file naming (default: 'pop909')
    """
    pop909_root = os.path.join(data_home, "POP909-Dataset", "POP909")
    output_index_dir = os.path.join(data_home, "yourmt3_indexes")
    os.makedirs(output_index_dir, exist_ok=True)

    # Collect all song IDs (folder names, 001–909)
    song_dirs = sorted(glob.glob(os.path.join(pop909_root, "[0-9][0-9][0-9]")))
    song_ids = [os.path.basename(d) for d in song_dirs]
    print(f"Found {len(song_ids)} songs in {pop909_root}")

    # Deterministic split by sorted index
    train_ids = song_ids[:N_TRAIN]
    val_ids = song_ids[N_TRAIN:N_TRAIN + N_VAL]
    test_ids = song_ids[N_TRAIN + N_VAL:]

    splits = {"train": train_ids, "validation": val_ids, "test": test_ids}

    for split, ids in splits.items():
        file_list = {}
        print(f"\nProcessing {split} split ({len(ids)} songs)...")

        for i, song_id in enumerate(ids):
            song_dir = os.path.join(pop909_root, song_id)
            audio_file = os.path.join(song_dir, f"{song_id}.wav")

            if not os.path.exists(audio_file):
                print(f"  WARNING: audio not found for {song_id}, skipping.")
                continue

            # Get audio info
            try:
                fs, n_frames, n_channels = get_audio_file_info(audio_file)
            except Exception as e:
                print(f"  WARNING: skipping {song_id}, bad audio file: {e}")
                continue
            if fs != 16000:
                print(f"  WARNING: {song_id} sample rate is {fs}Hz, expected 16000Hz. "
                      f"Resample with prepare_datasets.py first.")

            print(f"  [{i+1}/{len(ids)}] Processing {song_id}...")

            try:
                notes, note_events = create_note_event_and_note_from_pop909(song_dir, song_id)
            except Exception as e:
                print(f"  WARNING: skipping {song_id}, MIDI error: {e}")
                continue

            # Save .npy files
            notes_file = os.path.join(song_dir, f"{song_id}_notes.npy")
            note_events_file = os.path.join(song_dir, f"{song_id}_note_events.npy")
            np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
            np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)

            entry = {
                "pop909_id": song_id,
                "n_frames": n_frames,
                "mix_audio_file": audio_file,
                "notes_file": notes_file,
                "note_events_file": note_events_file,
                "midi_file": os.path.join(song_dir, "melody.mid"),
                "program": [POP909_PROGRAM],
                "is_drum": [0],
            }

            if split in ('validation', 'test'):
                audio = load_audio_file(audio_file, dtype=np.int16)
                audio = (audio / 2**15).astype(np.float32)
                if audio.ndim == 2:
                    # Downmix stereo/multi-channel audio to mono before slicing.
                    audio = audio.mean(axis=0)
                audio = audio.reshape(1, -1)
                segs = slice_padded_array(audio, SEG_LEN_FRAME, SEG_LEN_FRAME, pad=True)
                segs = rearrange(segs, 'n t -> n 1 t').astype(np.float32)
                audio_segments_file = os.path.join(song_dir, f"{song_id}_audio_segments.npy")
                np.save(audio_segments_file, segs, fix_imports=False)
                entry['audio_segments_file'] = audio_segments_file

            file_list[i] = entry

        # Write index JSON
        index_file = os.path.join(output_index_dir, f"{dataset_name}_{split}_file_list.json")
        with open(index_file, "w") as f:
            json.dump(file_list, f, indent=4)
        print(f"Saved {index_file} ({len(file_list)} entries)")

    print("\nPOP909 preprocessing complete.")


if __name__ == "__main__":
    from config.config import shared_cfg
    data_home = shared_cfg["PATH"]["data_home"]
    preprocess_pop909(data_home=data_home)
