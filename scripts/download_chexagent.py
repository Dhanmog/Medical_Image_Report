"""
Script to download StanfordAIMI/CheXagent-8b model weights from Hugging Face.
Supports resume and progress bar.
"""
import os
import sys
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(BASE_DIR, "models", "05_chexagent_8b")

def download_chexagent():
    print("=" * 60)
    print("  DOWNLOADING StanfordAIMI/CheXagent-8b")
    print(f"  Target directory: {TARGET_DIR}")
    print("=" * 60)
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    snapshot_path = snapshot_download(
        repo_id="StanfordAIMI/CheXagent-8b",
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETED SUCCESSFULLY!")
    print(f"  Files saved at: {snapshot_path}")
    print("=" * 60)

if __name__ == "__main__":
    download_chexagent()
