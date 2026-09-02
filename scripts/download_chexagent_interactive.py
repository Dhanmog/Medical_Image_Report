"""
Interactive downloader for StanfordAIMI/CheXagent-8b with live per-shard progress bar.
"""
import os
import sys
import time
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "StanfordAIMI/CheXagent-8b"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(BASE_DIR, "models", "05_chexagent_8b")

def download_all_files():
    os.makedirs(TARGET_DIR, exist_ok=True)
    api = HfApi()
    
    print("=" * 70)
    print(f"  DOWNLOADING CHEXAGENT-8B MODEL FILES TO:")
    print(f"  {TARGET_DIR}")
    print("=" * 70)
    
    repo_files = api.list_repo_tree(REPO_ID)
    files_to_download = [f for f in repo_files if hasattr(f, "size") and f.size]
    
    total_bytes = sum(f.size for f in files_to_download)
    print(f"Total files: {len(files_to_download)} | Total size: {total_bytes / (1024**3):.2f} GB\n")
    
    for idx, file_info in enumerate(files_to_download, 1):
        filename = file_info.path
        filesize_mb = file_info.size / (1024 * 1024)
        target_path = os.path.join(TARGET_DIR, filename)
        
        if os.path.exists(target_path) and os.path.getsize(target_path) == file_info.size:
            print(f"[{idx}/{len(files_to_download)}] Already downloaded: {filename} ({filesize_mb:.1f} MB)")
            continue
            
        print(f"[{idx}/{len(files_to_download)}] Downloading: {filename} ({filesize_mb:.1f} MB)...")
        start_t = time.time()
        
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        elapsed = time.time() - start_t
        speed_mb = filesize_mb / max(elapsed, 0.001)
        print(f"    -> Done in {elapsed:.1f}s ({speed_mb:.2f} MB/s)")
        
    print("\n" + "=" * 70)
    print("  ALL CHEXAGENT-8B FILES DOWNLOADED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    download_all_files()
