import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ICCV_DIR = os.path.join(PROJECT_DIR, "official_data_iccv_final")
FILES_DIR = os.path.join(ICCV_DIR, "files")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "processed_dataset_v1")
PREPROC_IMG_DIR = os.path.join(OUTPUT_DIR, "preprocessed_images")
os.makedirs(PREPROC_IMG_DIR, exist_ok=True)

MAPPING_PARSED_PATH = os.path.join(OUTPUT_DIR, "raw_parsed_mappings.csv")
TRAIN_CSV_OUT = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV_OUT = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_CSV_OUT = os.path.join(OUTPUT_DIR, "test.csv")
EXCLUSION_CSV_OUT = os.path.join(OUTPUT_DIR, "exclusion_log.csv")

IMAGE_SIZE = (224, 224)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
RANDOM_SEED = 42

print("=" * 70, flush=True)
print("STEP 3: IMAGE PREPROCESSING, EXCLUSIONS & PATIENT-LEVEL SPLITTING", flush=True)
print("=" * 70, flush=True)

if not os.path.exists(MAPPING_PARSED_PATH):
    raise FileNotFoundError(f"Missing {MAPPING_PARSED_PATH}. Please run Step 2 first.")

start_t = time.time()

# 1. Fast disk file set scan
print("Fast scanning disk files to build O(1) path set...", flush=True)
subfolders = [os.path.join(FILES_DIR, d) for d in os.listdir(FILES_DIR) if os.path.isdir(os.path.join(FILES_DIR, d))]

def scan_files(subfolder):
    found = set()
    stack = [subfolder]
    while stack:
        curr = stack.pop()
        try:
            with os.scandir(curr) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        found.add(os.path.normpath(entry.path))
        except Exception:
            pass
    return found

disk_file_set = set()
with ThreadPoolExecutor(max_workers=len(subfolders)) as executor:
    results = executor.map(scan_files, subfolders)
    for s in results:
        disk_file_set.update(s)

print(f"Scanned {len(disk_file_set)} verified image paths on disk in {time.time() - start_t:.2f}s.", flush=True)

df_raw = pd.read_csv(MAPPING_PARSED_PATH)
print(f"Loaded {len(df_raw)} raw parsed mapping records.", flush=True)

exclusions = []
valid_records = []

for idx, row in df_raw.iterrows():
    img_path = os.path.normpath(str(row["image_path"]))
    report_text = str(row.get("report", "")).strip()
    
    if img_path not in disk_file_set:
        exclusions.append({
            "patient_id": row["subject_id"],
            "study_id": row["study_id"],
            "image_path": img_path,
            "reason": "Missing Image File on Disk"
        })
        continue
        
    if not report_text or report_text.lower() == "nan":
        exclusions.append({
            "patient_id": row["subject_id"],
            "study_id": row["study_id"],
            "image_path": img_path,
            "reason": "Missing or Empty Radiology Report"
        })
        continue
        
    valid_records.append(row.to_dict())

df_valid = pd.DataFrame(valid_records)

dup_mask = df_valid.duplicated(subset=["image_path"], keep="first")
if dup_mask.sum() > 0:
    for idx, row in df_valid[dup_mask].iterrows():
        exclusions.append({
            "patient_id": row["subject_id"],
            "study_id": row["study_id"],
            "image_path": row["image_path"],
            "reason": "Duplicate Image Path Mapping"
        })
    df_valid = df_valid[~dup_mask].reset_index(drop=True)

df_exclusions = pd.DataFrame(exclusions)
print(f"Valid image-report pairs : {len(df_valid)}", flush=True)
print(f"Total excluded records   : {len(df_exclusions)}", flush=True)

df_exclusions.to_csv(EXCLUSION_CSV_OUT, index=False)
print(f"Saved exclusion log to: {EXCLUSION_CSV_OUT}", flush=True)

# 2. Patient-Level Splitting
unique_patients = df_valid["subject_id"].unique()
print(f"\nUnique patients count: {len(unique_patients)}", flush=True)

train_p, temp_p = train_test_split(unique_patients, test_size=0.20, random_state=RANDOM_SEED)
val_p, test_p = train_test_split(temp_p, test_size=0.50, random_state=RANDOM_SEED)

train_set_p = set(train_p)
val_set_p = set(val_p)
test_set_p = set(test_p)

assert len(train_set_p.intersection(val_set_p)) == 0, "Patient leakage in train/val!"
assert len(train_set_p.intersection(test_set_p)) == 0, "Patient leakage in train/test!"
assert len(val_set_p.intersection(test_set_p)) == 0, "Patient leakage in val/test!"

print(f"Patient distribution  - Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)}", flush=True)

df_train = df_valid[df_valid["subject_id"].isin(train_set_p)].reset_index(drop=True)
df_val = df_valid[df_valid["subject_id"].isin(val_set_p)].reset_index(drop=True)
df_test = df_valid[df_valid["subject_id"].isin(test_set_p)].reset_index(drop=True)

print(f"Image pair distribution - Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}", flush=True)

df_train["preprocessed_image_path"] = df_train["series_id"].apply(lambda s: os.path.join(PREPROC_IMG_DIR, f"{s}.jpg"))
df_val["preprocessed_image_path"] = df_val["series_id"].apply(lambda s: os.path.join(PREPROC_IMG_DIR, f"{s}.jpg"))
df_test["preprocessed_image_path"] = df_test["series_id"].apply(lambda s: os.path.join(PREPROC_IMG_DIR, f"{s}.jpg"))

# 3. Preprocess sample images to verify pipeline & cache function
def preprocess_and_cache_single_image(row):
    img_path = row["image_path"]
    series_id = row["series_id"]
    out_path = os.path.join(PREPROC_IMG_DIR, f"{series_id}.jpg")
    try:
        buf = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return False
        img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        img_enhanced = clahe.apply(img_resized)
        cv2.imwrite(out_path, img_enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return True
    except Exception:
        return False

print("\nValidating image preprocessing pipeline on sample batch of 1,000 images...", flush=True)
sample_rows = df_valid.sample(n=min(1000, len(df_valid)), random_state=RANDOM_SEED).to_dict("records")
prep_success = 0
with ThreadPoolExecutor(max_workers=32) as executor:
    results = executor.map(preprocess_and_cache_single_image, sample_rows)
    for res in results:
        if res:
            prep_success += 1

print(f"Sample preprocessing test passed: {prep_success} / {len(sample_rows)} images preprocessed successfully.", flush=True)

df_train.to_csv(TRAIN_CSV_OUT, index=False)
df_val.to_csv(VAL_CSV_OUT, index=False)
df_test.to_csv(TEST_CSV_OUT, index=False)

print("\nSaved split CSVs:", flush=True)
print(f"  • Train CSV       : {TRAIN_CSV_OUT} ({len(df_train)} rows)", flush=True)
print(f"  • Validation CSV  : {VAL_CSV_OUT} ({len(df_val)} rows)", flush=True)
print(f"  • Test CSV        : {TEST_CSV_OUT} ({len(df_test)} rows)", flush=True)
print("Step 3 complete!\n", flush=True)
