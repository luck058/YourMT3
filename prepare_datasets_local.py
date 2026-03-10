import os
import glob
import librosa
import soundfile as sf
import zipfile
import urllib.request

# --- LOCAL CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # YourMT3 directory
DATA_DIR = os.path.join(SCRIPT_DIR, "mlp_dataset_local")  # Extract to YourMT3/mlp_dataset_local
POP909_ZIP = os.path.join(SCRIPT_DIR, "POP909-Dataset.zip")  # Zip file in YourMT3/
TARGET_SR = 16000  # 16 kHz sample rate

def prepare_pop909():
    """Extracts the POP909 zip file from the local directory."""
    pop909_dir = os.path.join(DATA_DIR, "POP909-Dataset")

    print(f"Looking for POP909 zip at: {POP909_ZIP}")

    if os.path.exists(POP909_ZIP):
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)

        # Check if the destination directory already exists
        if not os.path.exists(pop909_dir):
            print(f"Extracting {POP909_ZIP} to {DATA_DIR}...")
            with zipfile.ZipFile(POP909_ZIP, 'r') as zip_ref:
                # List first few files to understand structure
                print("First 5 files in zip:")
                for i, name in enumerate(zip_ref.namelist()[:5]):
                    print(f"  - {name}")

                # Extract
                zip_ref.extractall(DATA_DIR)
            print("POP909 Extraction Complete!")

            # Show what was extracted
            print("\nExtracted contents:")
            for root, dirs, files in os.walk(DATA_DIR):
                level = root.replace(DATA_DIR, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                if level < 2:  # Only show first 2 levels
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:3]:  # Show first 3 files
                        print(f"{subindent}{file}")
                    if len(files) > 3:
                        print(f"{subindent}... and {len(files) - 3} more files")
                if level >= 2:
                    break

        else:
            print(f"POP909 is already extracted at: {pop909_dir}")

        # Search for audio files
        print("\nSearching for audio files...")
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
        audio_files = []
        for ext in audio_extensions:
            found = glob.glob(os.path.join(DATA_DIR, '**', ext), recursive=True)
            if found:
                print(f"  Found {len(found)} {ext} files")
                audio_files.extend(found)

        if audio_files:
            print(f"\nTotal audio files found: {len(audio_files)}")
            print("First 3 audio files:")
            for f in audio_files[:3]:
                print(f"  - {f}")
        else:
            print("\n⚠️  No audio files found! POP909 might only contain MIDI files.")
            # Search for MIDI files
            midi_files = glob.glob(os.path.join(DATA_DIR, '**', '*.mid*'), recursive=True)
            if midi_files:
                print(f"Found {len(midi_files)} MIDI files instead")
                print("First 3 MIDI files:")
                for f in midi_files[:3]:
                    print(f"  - {f}")

    else:
        print(f"Error: Could not find {POP909_ZIP}")
        print(f"\nPlease update POP909_ZIP variable in the script to point to your zip file.")
        print(f"Common locations:")
        print(f"  - ~/Downloads/POP909-Dataset.zip")
        print(f"  - ~/Documents/POP909-Dataset.zip")
        print(f"  - Current directory: {os.path.join(os.path.dirname(__file__), 'POP909-Dataset.zip')}")

def prepare_aam():
    """Downloads, extracts, and downsamples the AAM dataset (LOCAL VERSION - SKIPPED)."""
    print("\n================================================")
    print("AAM dataset download skipped for local testing.")
    print("AAM is several GB and takes a long time to download.")
    print("Only POP909 will be processed locally.")
    print("================================================")

if __name__ == "__main__":
    print("=" * 60)
    print("LOCAL Dataset Preparation Pipeline")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"POP909 zip path: {POP909_ZIP}")
    print("=" * 60)

    prepare_pop909()
    # prepare_aam()  # Skip AAM for local testing

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
