import os
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

PROJECT_DIR = os.getcwd()
PROC_DIR = os.path.join(PROJECT_DIR, 'processed_dataset_v1')

TRAIN_CSV = os.path.join(PROC_DIR, 'train.csv')
VAL_CSV = os.path.join(PROC_DIR, 'validation.csv')
TEST_CSV = os.path.join(PROC_DIR, 'test.csv')

print("=" * 70, flush=True)
print("1. VERIFYING DATASET SPLIT CSVs & DISK FILE EXISTENCE", flush=True)
print("=" * 70, flush=True)

# 1. Existence and readability check
for name, p in [("Train", TRAIN_CSV), ("Validation", VAL_CSV), ("Test", TEST_CSV)]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{name} CSV missing: {p}")
    size = os.path.getsize(p)
    print(f"  • {name} CSV readable ({size:,} bytes): {p}", flush=True)

df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_test = pd.read_csv(TEST_CSV)

print(f"\n2. VERIFYING ROW COUNTS", flush=True)
print(f"  • Train count      : {len(df_train):,} (Expected: 208,769)")
print(f"  • Validation count : {len(df_val):,} (Expected: 26,197)")
print(f"  • Test count       : {len(df_test):,} (Expected: 26,171)")
total_cnt = len(df_train) + len(df_val) + len(df_test)
print(f"  • Total count      : {total_cnt:,} (Expected: 261,137)")

assert len(df_train) == 208769, "Train count mismatch!"
assert len(df_val) == 26197, "Val count mismatch!"
assert len(df_test) == 26171, "Test count mismatch!"
assert total_cnt == 261137, "Total count mismatch!"

# 3. Patient isolation check
print(f"\n3. VERIFYING PATIENT ISOLATION ACROSS SPLITS", flush=True)
p_train = set(df_train["subject_id"].unique())
p_val = set(df_val["subject_id"].unique())
p_test = set(df_test["subject_id"].unique())

ov_tv = len(p_train.intersection(p_val))
ov_tt = len(p_train.intersection(p_test))
ov_vt = len(p_val.intersection(p_test))

print(f"  • Train/Val patient overlap  : {ov_tv}")
print(f"  • Train/Test patient overlap : {ov_tt}")
print(f"  • Val/Test patient overlap   : {ov_vt}")

assert ov_tv == 0 and ov_tt == 0 and ov_vt == 0, "Patient leakage detected!"

# 4. Multi-threaded image path existence check
print(f"\n4. VERIFYING EVERY CSV IMAGE PATH ON DISK", flush=True)
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

def check_row(idx_row):
    idx, row = idx_row
    p = str(row["image_path"])
    rep = str(row.get("report", "")).strip()
    clean_rep = str(row.get("cleaned_report", "")).strip()
    
    path_exists = os.path.exists(p)
    rep_valid = len(rep) > 0 and rep.lower() != "nan" and len(clean_rep) > 0
    return path_exists, rep_valid

all_rows = list(enumerate(df_all.to_dict("records")))

broken_paths = 0
empty_reports = 0

with ThreadPoolExecutor(max_workers=32) as executor:
    results = executor.map(check_row, all_rows)
    for p_ok, r_ok in results:
        if not p_ok:
            broken_paths += 1
        if not r_ok:
            empty_reports += 1

print(f"  • Broken image paths : {broken_paths}")
print(f"  • Empty reports      : {empty_reports}")

assert broken_paths == 0, "Broken image paths found!"
assert empty_reports == 0, "Empty reports found!"

# 5. Duplicates check
print(f"\n5. CHECKING FOR DUPLICATES & INVALID IDs", flush=True)
dup_paths = df_all.duplicated(subset=["image_path"]).sum()
dup_mappings = df_all.duplicated(subset=["image_path", "report"]).sum()

invalid_subj = df_all["subject_id"].isnull().sum()
invalid_study = df_all["study_id"].isnull().sum()

print(f"  • Duplicate image entries     : {dup_paths}")
print(f"  • Duplicate image-report pairs: {dup_mappings}")
print(f"  • Null/invalid subject IDs    : {invalid_subj}")
print(f"  • Null/invalid study IDs      : {invalid_study}")

assert dup_paths == 0, "Duplicate image paths found!"
assert dup_mappings == 0, "Duplicate image-report pairs found!"
assert invalid_subj == 0, "Null subject IDs found!"
assert invalid_study == 0, "Null study IDs found!"

print("\n" + "=" * 70, flush=True)
print("ALL DATASET CHECKS PASSED 100%! DATASET IS PERFECT.", flush=True)
print("=" * 70 + "\n", flush=True)
