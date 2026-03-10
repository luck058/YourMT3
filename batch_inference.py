#!/usr/bin/env python3
"""
Batch Inference Script for YourMT3
Processes multiple audio files and generates MIDI transcriptions
"""

import sys
import os
import argparse
import glob
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'amt/src'))

from model_helper import load_model_checkpoint, transcribe


def get_audio_files(input_dir):
    """Get all audio files from directory recursively."""
    extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    return sorted(audio_files)


def prepare_audio_info(filepath):
    """Prepare minimal audio info - transcribe() handles the real work."""
    return {
        "filepath": filepath,
        "track_name": os.path.splitext(os.path.basename(filepath))[0],
    }


def main():
    parser = argparse.ArgumentParser(description='Batch Inference for YourMT3')

    # Input/Output
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, default='./batch_output',
                        help='Directory to save MIDI files (default: ./batch_output)')
    parser.add_argument('--file-list', type=str, default=None,
                        help='Text file with specific audio files to process (one per line)')

    # Model Configuration
    parser.add_argument('--model-name', type=str,
                        default='YPTF.MoE+Multi (noPS)',
                        choices=['YMT3+', 'YPTF+Single (noPS)', 'YPTF+Multi (PS)',
                                'YPTF.MoE+Multi (noPS)', 'YPTF.MoE+Multi (PS)'],
                        help='Model checkpoint to use')
    parser.add_argument('--project', type=str, default='2024',
                        help='Project name for checkpoint path')
    parser.add_argument('--precision', type=str, default='16',
                        choices=['32', 'bf16-mixed', '16'],
                        help='Floating point precision')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run inference on (default: auto)')

    # Processing Options
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start processing from this file index (for resuming)')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='End processing at this file index')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that already have MIDI output')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Set model arguments based on model name
    if args.model_name == "YMT3+":
        checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
        model_args = [checkpoint, '-p', args.project, '-pr', args.precision]
    elif args.model_name == "YPTF+Single (noPS)":
        checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
        model_args = [checkpoint, '-p', args.project, '-enc', 'perceiver-tf', '-ac', 'spec',
                '-hop', '300', '-atc', '1', '-pr', args.precision]
    elif args.model_name == "YPTF+Multi (PS)":
        checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
        model_args = [checkpoint, '-p', args.project, '-tk', 'mc13_full_plus_256',
                '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
                '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', args.precision]
    elif args.model_name == "YPTF.MoE+Multi (noPS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
        model_args = [checkpoint, '-p', args.project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', args.precision]
    elif args.model_name == "YPTF.MoE+Multi (PS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
        model_args = [checkpoint, '-p', args.project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', args.precision]
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    # Load model
    print(f"Loading model: {args.model_name}")
    print(f"Checkpoint: {checkpoint}")
    model = load_model_checkpoint(args=model_args, device=device)
    if device == 'cpu':
        print("⚠️  Running on CPU (inference will take 5-10 minutes per file)")
    print("Model loaded!")

    # Get list of audio files
    if args.file_list:
        print(f"Reading file list from: {args.file_list}")
        with open(args.file_list, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    else:
        print(f"Scanning for audio files in: {args.input_dir}")
        audio_files = get_audio_files(args.input_dir)

    # Apply start/end indices
    if args.end_idx is not None:
        audio_files = audio_files[args.start_idx:args.end_idx]
    else:
        audio_files = audio_files[args.start_idx:]

    print(f"Found {len(audio_files)} audio files to process")

    # Process each file
    successful = 0
    failed = 0
    skipped = 0

    for idx, audio_file in enumerate(audio_files, start=args.start_idx):
        track_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_midi = os.path.join(args.output_dir, f"{track_name}.mid")

        # Skip if already processed
        if args.skip_existing and os.path.exists(output_midi):
            print(f"[{idx+1}/{len(audio_files)+args.start_idx}] Skipping (exists): {audio_file}")
            skipped += 1
            continue

        print(f"\n[{idx+1}/{len(audio_files)+args.start_idx}] Processing: {audio_file}")

        try:
            # Use absolute paths to avoid path issues
            abs_audio_file = os.path.abspath(audio_file)

            # Prepare audio info
            audio_info = prepare_audio_info(abs_audio_file)

            # Transcribe - change to output directory so transcribe saves there
            original_dir = os.getcwd()
            abs_output_dir = os.path.abspath(args.output_dir)
            os.chdir(abs_output_dir)

            midifile = transcribe(model, audio_info)

            os.chdir(original_dir)

            print(f"  ✓ Success! MIDI saved to: {midifile}")
            successful += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    # Summary
    print("\n" + "="*60)
    print("BATCH INFERENCE SUMMARY")
    print("="*60)
    print(f"Total files processed: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
