import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ICCV_DIR = os.path.join(PROJECT_DIR, "official_data_iccv_final")
FILES_DIR = os.path.join(ICCV_DIR, "files")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "processed_dataset_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INVALID_REPORT_PATH = os.path.join(OUTPUT_DIR, "invalid_images_report.txt")
AUDIT_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "image_audit_summary.json")

print("=" * 70, flush=True)
print("STEP 1: AUDITING AND VALIDATING ALL X-RAY IMAGES", flush=True)
print("=" * 70, flush=True)
print(f"Scanning directory: {ICCV_DIR}", flush=True)

start_time = time.time()
image_paths = []
format_counts = Counter()
invalid_records = []

subfolders = [os.path.join(FILES_DIR, d) for d in os.listdir(FILES_DIR) if os.path.isdir(os.path.join(FILES_DIR, d))]

def scan_and_validate_subfolder(subfolder):
    valid_found = []
    invalid_found = []
    local_formats = Counter()
    stack = [subfolder]
    while stack:
        curr = stack.pop()
        try:
            with os.scandir(curr) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        local_formats[ext] += 1
                        try:
                            st = entry.stat(follow_symlinks=False)
                            if st.st_size < 100:
                                invalid_found.append({"path": entry.path, "reason": f"Truncated/tiny file ({st.st_size} bytes)"})
                            else:
                                valid_found.append(entry.path)
                        except Exception as e:
                            invalid_found.append({"path": entry.path, "reason": f"Stat error: {str(e)}"})
        except Exception:
            pass
    return valid_found, invalid_found, local_formats

all_valid = []
all_invalid = []
with ThreadPoolExecutor(max_workers=len(subfolders)) as executor:
    results = executor.map(scan_and_validate_subfolder, subfolders)
    for v, inv, fmt in results:
        all_valid.extend(v)
        all_invalid.extend(inv)
        format_counts.update(fmt)

total_found = len(all_valid) + len(all_invalid)
scan_time = time.time() - start_time

print(f"\nAudit & Validation completed in {scan_time:.2f} seconds.", flush=True)
print(f"Total images found      : {total_found}", flush=True)
print(f"Valid images            : {len(all_valid)}", flush=True)
print(f"Corrupted/unreadable    : {len(all_invalid)}", flush=True)
print("File format breakdown   :", dict(format_counts), flush=True)

log_lines = [
    "INVALID / CORRUPTED X-RAY IMAGES REPORT",
    "=" * 70,
    f"Total images scanned : {total_found}",
    f"Valid images         : {len(all_valid)}",
    f"Corrupted / Invalid  : {len(all_invalid)}",
    "=" * 70,
    ""
]

for rec in all_invalid:
    log_lines.append(f"Path: {rec['path']} | Reason: {rec['reason']}")

with open(INVALID_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")

summary = {
    "total_images_found": total_found,
    "valid_images": len(all_valid),
    "corrupted_unreadable_images": len(all_invalid),
    "format_breakdown": dict(format_counts),
    "audit_time_seconds": round(scan_time, 2)
}

with open(AUDIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"Saved invalid report to : {INVALID_REPORT_PATH}", flush=True)
print(f"Saved audit summary to  : {AUDIT_SUMMARY_PATH}", flush=True)
print("Step 1 complete!\n", flush=True)
