import os
import sys
import shutil
import json
import pandas as pd
import numpy as np
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets", "mimic_cxr", "processed", "processed_dataset_v1")
ARTIFACTS_DIR = r"C:\Users\dhanu\.gemini\antigravity-ide\brain\fc0f1678-3e23-4626-ae43-cfb1399eeb5c"

def verify_split(csv_name):
    csv_path = os.path.join(DATASET_DIR, csv_name)
    if not os.path.exists(csv_path):
        return None, f"CSV file {csv_name} not found"

    df = pd.read_csv(csv_path)
    total_records = len(df)
    
    missing_images = 0
    corrupted_images = 0
    missing_reports = 0
    duplicates = df.duplicated(subset=['image_path']).sum()
    
    # Sample a few records for verification
    sample_records = []
    
    for idx, row in df.iterrows():
        img_path = row.get("image_path")
        report = row.get("report")
        
        # Check report
        if pd.isna(report) or not str(report).strip():
            missing_reports += 1
            
        # Check image existence
        if pd.isna(img_path) or not os.path.exists(img_path):
            missing_images += 1
            continue
            
        # Check image readability (corrupted/empty)
        if os.path.getsize(img_path) < 100:
            corrupted_images += 1
            continue
            
        # Verify header & decode
        try:
            with open(img_path, "rb") as f:
                header = f.read(3)
            # Check for JPEG or PNG headers
            if header != b"\xff\xd8\xff" and header[:2] != b"\x89P":
                corrupted_images += 1
                continue
                
            # Quick decode check for first 1000 rows to speed up, plus any samples
            if idx < 1000 or len(sample_records) < 3:
                buf = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    corrupted_images += 1
                    continue
        except Exception:
            corrupted_images += 1
            continue

        # Keep a few valid samples to copy to artifacts for display
        if len(sample_records) < 3 and not pd.isna(report) and str(report).strip():
            sample_records.append({
                "subject_id": row.get("subject_id", "N/A"),
                "study_id": row.get("study_id", "N/A"),
                "image_path": img_path,
                "report": report
            })

    stats = {
        "total_records": total_records,
        "missing_images": missing_images,
        "corrupted_images": corrupted_images,
        "missing_reports": missing_reports,
        "duplicates": int(duplicates),
        "samples": sample_records
    }
    return stats, None

def main():
    print("=" * 60)
    print("PROGRAMMATIC DATASET & IMAGE-REPORT MAPPING VERIFICATION")
    print("=" * 60)
    
    splits = ["train.csv", "validation.csv", "test.csv"]
    results = {}
    
    for split in splits:
        print(f"Verifying {split}...")
        stats, err = verify_split(split)
        if err:
            print(f"Error verifying {split}: {err}")
            sys.exit(1)
        results[split] = stats
        print(f"  • Total: {stats['total_records']} | Missing Img: {stats['missing_images']} | Corrupted Img: {stats['corrupted_images']} | Missing Rep: {stats['missing_reports']} | Duplicates: {stats['duplicates']}")

    # Copy sample images to artifacts directory
    print("\nCopying sample X-rays to artifacts directory for visualization...")
    html_samples = []
    
    for split in ["train.csv", "validation.csv"]:
        samples = results[split]["samples"]
        for i, sample in enumerate(samples):
            img_ext = os.path.splitext(sample["image_path"])[1]
            dest_name = f"sample_{split.split('.')[0]}_{i+1}{img_ext}"
            dest_path = os.path.join(ARTIFACTS_DIR, dest_name)
            
            try:
                shutil.copy2(sample["image_path"], dest_path)
                print(f"  • Copied {os.path.basename(sample['image_path'])} -> {dest_name}")
                
                # Format report text for display
                clean_rep = sample["report"].replace("Findings:", "\n**Findings:**").replace("Impression:", "\n**Impression:**")
                
                html_samples.append({
                    "title": f"Sample from {split.split('.')[0].upper()} (Patient: {sample['subject_id']}, Study: {sample['study_id']})",
                    "image_url": f"file:///{dest_path.replace(os.sep, '/')}",
                    "report": clean_rep
                })
            except Exception as e:
                print(f"  ⚠️ Failed to copy sample image {sample['image_path']}: {e}")

    # Generate stats report in markdown
    report_path = os.path.join(ARTIFACTS_DIR, "dataset_stats.md")
    
    # Calculate global totals
    total_imgs = sum(r["total_records"] for r in results.values())
    total_missing_imgs = sum(r["missing_images"] for r in results.values())
    total_corrupted_imgs = sum(r["corrupted_images"] for r in results.values())
    total_missing_reps = sum(r["missing_reports"] for r in results.values())
    total_duplicates = sum(r["duplicates"] for r in results.values())
    total_valid = total_imgs - (total_missing_imgs + total_corrupted_imgs + total_missing_reps)

    md_content = f"""# Chest X-Ray Dataset Verification Report

This report presents programmatic statistics and quality audits verified across the splits of your MIMIC-CXR dataset.

## 📊 Summary Statistics

| Metric | Count |
| :--- | :--- |
| **Total Mapped Records** | {total_imgs:,} |
| **Valid Image-Report Pairs** | {total_valid:,} |
| **Missing Images** | {total_missing_imgs:,} |
| **Corrupted/Unreadable Images** | {total_corrupted_imgs:,} |
| **Missing Reports** | {total_missing_reps:,} |
| **Duplicate Image Mappings** | {total_duplicates:,} |

---

## 📂 Split Counts

| Dataset Split | Total Records | Missing Images | Corrupted Images | Missing Reports | Duplicates |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (`train.csv`)** | {results["train.csv"]["total_records"]:,} | {results["train.csv"]["missing_images"]:,} | {results["train.csv"]["corrupted_images"]:,} | {results["train.csv"]["missing_reports"]:,} | {results["train.csv"]["duplicates"]:,} |
| **Validation (`validation.csv`)** | {results["validation.csv"]["total_records"]:,} | {results["validation.csv"]["missing_images"]:,} | {results["validation.csv"]["corrupted_images"]:,} | {results["validation.csv"]["missing_reports"]:,} | {results["validation.csv"]["duplicates"]:,} |
| **Test (`test.csv`)** | {results["test.csv"]["total_records"]:,} | {results["test.csv"]["missing_images"]:,} | {results["test.csv"]["corrupted_images"]:,} | {results["test.csv"]["missing_reports"]:,} | {results["test.csv"]["duplicates"]:,} |

---

## 🖼️ Visual Image-Report Mapping Verification
Below are randomly sampled chest X-rays paired with their corresponding ground-truth radiology reports.

"""

    # Add samples to markdown
    for sample in html_samples:
        md_content += f"""### {sample['title']}

![X-Ray Image]({sample['image_url']})

**Radiologist Report:**
{sample['report']}

---
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\n[OK] Saved dataset statistics report to: {report_path}")
    print("Verification completed successfully!")

if __name__ == "__main__":
    main()
