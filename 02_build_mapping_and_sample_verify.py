import os
import sys
import ast
import re
import random
import pandas as pd
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ICCV_DIR = os.path.join(PROJECT_DIR, "official_data_iccv_final")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "processed_dataset_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(PROJECT_DIR, "mimic_cxr_aug_train.csv")
VAL_CSV = os.path.join(PROJECT_DIR, "mimic_cxr_aug_validate.csv")
SAMPLE_VERIFY_PATH = os.path.join(OUTPUT_DIR, "sample_verification_50.txt")
MAPPING_PARSED_PATH = os.path.join(OUTPUT_DIR, "raw_parsed_mappings.csv")

START_TOKEN = "<start>"
END_TOKEN = "<end>"

def clean_report(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9.,;:!?\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f"{START_TOKEN} {text} {END_TOKEN}"

def get_view(img_path, ap_list, pa_list, lat_list):
    if img_path in ap_list:
        return "AP"
    if img_path in pa_list:
        return "PA"
    if img_path in lat_list:
        return "LATERAL"
    return "FRONTAL"

print("=" * 70, flush=True)
print("STEP 2: BUILDING EXACT IMAGE-TO-REPORT MAPPING & SAMPLE VERIFICATION", flush=True)
print("=" * 70, flush=True)

def parse_dataset_csv(csv_path):
    print(f"Parsing metadata from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    records = []
    
    missing_images_cnt = 0
    missing_reports_cnt = 0
    empty_reports_cnt = 0
    
    for idx, row in df.iterrows():
        subj_id = row.get("subject_id", f"patient_{idx}")
        
        try:
            imgs = ast.literal_eval(row["image"]) if isinstance(row["image"], str) else []
            texts = ast.literal_eval(row["text"]) if isinstance(row["text"], str) else []
            aps = ast.literal_eval(row["AP"]) if ("AP" in row and isinstance(row["AP"], str)) else []
            pas = ast.literal_eval(row["PA"]) if ("PA" in row and isinstance(row["PA"], str)) else []
            lats = ast.literal_eval(row["Lateral"]) if ("Lateral" in row and isinstance(row["Lateral"], str)) else []
        except Exception as e:
            continue
            
        if not imgs:
            missing_images_cnt += 1
            continue
        if not texts:
            missing_reports_cnt += 1
            continue
            
        study_order = []
        study_img_map = {}
        for img_rel in imgs:
            parts = img_rel.replace("\\", "/").split("/")
            st_id = parts[3] if len(parts) >= 4 else "unknown_study"
            if st_id not in study_img_map:
                study_order.append(st_id)
                study_img_map[st_id] = []
            study_img_map[st_id].append(img_rel)
            
        for s_idx, st_id in enumerate(study_order):
            report_text = texts[s_idx] if s_idx < len(texts) else (texts[0] if len(texts) > 0 else "")
            
            if not report_text or not str(report_text).strip():
                empty_reports_cnt += 1
                continue
                
            cleaned_rep = clean_report(report_text)
            
            for img_rel in study_img_map[st_id]:
                img_name = os.path.basename(img_rel)
                series_or_img_id = os.path.splitext(img_name)[0]
                full_path = os.path.normpath(os.path.join(ICCV_DIR, img_rel.replace("/", os.sep)))
                view_type = get_view(img_rel, aps, pas, lats)
                
                records.append({
                    "subject_id": str(subj_id),
                    "study_id": str(st_id),
                    "series_id": str(series_or_img_id),
                    "image_path": full_path,
                    "rel_image_path": img_rel,
                    "view": view_type,
                    "report": str(report_text).strip(),
                    "cleaned_report": cleaned_rep
                })
                
    res_df = pd.DataFrame(records)
    print(f"Parsed {len(res_df)} image-report pairs from {os.path.basename(csv_path)}", flush=True)
    return res_df

df_train_mapped = parse_dataset_csv(TRAIN_CSV)
df_val_mapped = parse_dataset_csv(VAL_CSV)

df_all_mapped = pd.concat([df_train_mapped, df_val_mapped], ignore_index=True)
print(f"\nTotal parsed raw image-report pairs across train/val metadata: {len(df_all_mapped)}", flush=True)

dup_count = df_all_mapped.duplicated(subset=["image_path"]).sum()
print(f"Duplicate image paths in metadata: {dup_count}", flush=True)

df_all_mapped.to_csv(MAPPING_PARSED_PATH, index=False)
print(f"Saved raw mapped CSV to: {MAPPING_PARSED_PATH}", flush=True)

random.seed(42)
sample_indices = random.sample(range(len(df_all_mapped)), min(50, len(df_all_mapped)))
sample_rows = df_all_mapped.iloc[sample_indices]

sample_log_lines = [
    "MANUAL VERIFICATION OF 50 RANDOM IMAGE-REPORT PAIRS",
    "=" * 80,
    f"Seed: 42 | Total Pairs in Pool: {len(df_all_mapped)}",
    "=" * 80,
    ""
]

for idx, (_, row) in enumerate(sample_rows.iterrows()):
    sample_log_lines.append(f"--- SAMPLE {idx + 1} ---")
    sample_log_lines.append(f"Patient ID (subject_id) : {row['subject_id']}")
    sample_log_lines.append(f"Study ID   (study_id)   : {row['study_id']}")
    sample_log_lines.append(f"Series/Img (series_id)  : {row['series_id']}")
    sample_log_lines.append(f"View                    : {row['view']}")
    sample_log_lines.append(f"Image Path              : {row['image_path']}")
    sample_log_lines.append(f"Raw Report Preview      : {row['report'][:250]}...")
    sample_log_lines.append(f"Cleaned Report Preview  : {row['cleaned_report'][:250]}...")
    sample_log_lines.append("-" * 80)

with open(SAMPLE_VERIFY_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(sample_log_lines) + "\n")

print(f"Saved 50 verification samples log to: {SAMPLE_VERIFY_PATH}", flush=True)
print("Step 2 complete!\n", flush=True)
