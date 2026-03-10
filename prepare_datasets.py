import os
import glob
import librosa
import soundfile as sf
import zipfile
import urllib.request

# --- CONFIGURATION ---
DATA_DIR = "/disk/scratch/s2286943/mlp_dataset"
TARGET_SR = 16000  # 16 kHz sample rate

def prepare_pop909():
    """Extracts the POP909 zip file from the home directory into scratch."""
    pop909_zip = "/home/s2286943/POP909-Dataset.zip"
    pop909_dir = os.path.join(DATA_DIR, "POP909-Dataset")
    
    if os.path.exists(pop909_zip):
        # We check if the destination directory already exists
        if not os.path.exists(pop909_dir):
            print(f"Extracting {pop909_zip} to {DATA_DIR}...")
            with zipfile.ZipFile(pop909_zip, 'r') as zip_ref:
                # Extracting to DATA_DIR will naturally create the POP909-Dataset folder 
                # assuming the zip was created from the repository root
                zip_ref.extractall(DATA_DIR)
            print("POP909 Extraction Complete!")
        else:
            print("POP909 is already extracted in the scratch directory.")
    else:
        print(f"Error: Could not find {pop909_zip}. Please verify the file path.")

def prepare_aam():
    """Downloads, extracts, and downsamples the AAM dataset."""
    aam_dir = os.path.join(DATA_DIR, "AAM")
    os.makedirs(aam_dir, exist_ok=True)

    # Zenodo Links
    files_to_download = {
        "0001-1000-audio-mixes.zip": "https://zenodo.org/records/5794629/files/0001-1000-audio-mixes.zip?download=1",
        "0001-1000-annotations-v1.1.0.zip": "https://zenodo.org/records/5794629/files/0001-1000-annotations-v1.1.0.zip?download=1",
        "0001-1000-midis.zip": "https://zenodo.org/records/5794629/files/0001-1000-midis.zip?download=1"
    }

    # 1. Download and Extract
    for filename, url in files_to_download.items():
        zip_path = os.path.join(aam_dir, filename)
        
        # Download
        if not os.path.exists(zip_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, zip_path)

        # Extract safely using a flag file so it doesn't loop
        extract_flag = zip_path + ".extracted"
        if not os.path.exists(extract_flag):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(aam_dir)
            # Create the flag file so we know it's done
            open(extract_flag, 'w').close()

    # 2. Downsample Audio Mixes
    flac_files = glob.glob(os.path.join(aam_dir, "*.flac"))
    
    # Only print if there's actually work to do
    if flac_files:
        print(f"Downsampling {len(flac_files)} AAM files to {TARGET_SR}Hz...")
        for flac_file in flac_files:
            wav_file = flac_file.replace(".flac", "_16k.wav")
            if not os.path.exists(wav_file):
                y, sr = librosa.load(flac_file, sr=TARGET_SR, mono=True)
                sf.write(wav_file, y, TARGET_SR)
                # Remove high-res FLAC to save cluster quota
                os.remove(flac_file)
        print("AAM Downsampling Complete!")

if __name__ == "__main__":
    print("Starting dataset preparation pipeline...")
    prepare_pop909()
    prepare_aam()
    print("All datasets prepared successfully!")
