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
compute_pos_weight_script.py

Run this once before training the FFNNPianoRollDecoder to compute a
data-driven pos_weight for BCEWithLogitsLoss.

Usage (from the amt/src directory):
    python utils/compute_pos_weight_script.py \
        --preset musicnet_mt3_synth_only \
        --n_batches 200

Then update config/config.py:
    model_cfg["decoder"]["ffnn"]["pos_weight"] = <computed_value>

Typical range: 20–100 depending on dataset density.
"""
import argparse
import sys
import os

# Ensure imports resolve from amt/src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config.config import audio_cfg, model_cfg, shared_cfg
from config.data_presets import data_preset_single_cfg, data_preset_multi_cfg
from utils.piano_roll_utils import (
    batch_notes_to_piano_roll,
    compute_pos_weight,
    compute_n_frames_from_audio_cfg,
)
from utils.datasets_helper import create_merged_train_dataset_info
from utils.datasets_train import CachedAudioDataset


def main():
    parser = argparse.ArgumentParser(description="Compute pos_weight for FFNNPianoRollDecoder.")
    parser.add_argument(
        "--preset", type=str, default="musicnet_mt3_synth_only",
        help="Data preset name from config/data_presets.py.",
    )
    parser.add_argument(
        "--n_batches", type=int, default=200,
        help="Number of sub-batches to sample for the estimate.",
    )
    args = parser.parse_args()

    ffnn_cfg = model_cfg["decoder"]["ffnn"]
    programs = list(ffnn_cfg["instruments"].values())
    n_frames = compute_n_frames_from_audio_cfg(audio_cfg)
    segment_duration = audio_cfg["input_frames"] / audio_cfg["sample_rate"]

    print(f"Computing pos_weight for instruments: {list(ffnn_cfg['instruments'].keys())}")
    print(f"  programs={programs}, n_frames={n_frames}, segment_duration={segment_duration:.3f}s")
    print(f"  Sampling {args.n_batches} batches from preset '{args.preset}'")

    # Build file list for the requested preset
    preset_multi = {"presets": [args.preset]}
    train_data_info = create_merged_train_dataset_info(preset_multi)

    piano_roll_cfg = {
        "programs": programs,
        "n_frames": n_frames,
        "pitch_min": ffnn_cfg["pitch_min"],
        "pitch_max": ffnn_cfg["pitch_max"],
    }

    ds = CachedAudioDataset(
        file_list=train_data_info["merged_file_list"],
        seg_len_frame=int(audio_cfg["input_frames"]),
        sub_batch_size=shared_cfg["BSZ"]["train_sub"],
        num_files_cache=None,
        stem_iaug_prob=None,        # no augmentation needed for stats
        stem_xaug_policy=None,
        piano_roll_cfg=piano_roll_cfg,
    )

    all_rolls = []
    indices = np.random.choice(len(ds), size=min(args.n_batches, len(ds)), replace=False)

    for i, idx in enumerate(indices):
        try:
            item = ds[int(idx)]
        except Exception as e:
            print(f"  Warning: skipped index {idx}: {e}")
            continue
        if len(item) > 3:
            all_rolls.append(item[3])  # piano_roll_labels: (sub_b, T', I, P)
        if (i + 1) % 20 == 0:
            print(f"  Collected {i + 1}/{len(indices)} batches ...")

    if not all_rolls:
        print("ERROR: No piano roll labels collected. Check preset and notes files.")
        sys.exit(1)

    combined = torch.cat(all_rolls, dim=0)  # (N, T', I, P)
    weight = compute_pos_weight(combined)

    print(f"\nRecommended pos_weight: {weight:.2f}")
    print(f"\nUpdate config/config.py:")
    print(f"    model_cfg[\"decoder\"][\"ffnn\"][\"pos_weight\"] = {weight:.1f}")


if __name__ == "__main__":
    main()
