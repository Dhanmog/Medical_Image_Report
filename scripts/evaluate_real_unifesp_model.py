r"""
================================================================================
EVALUATION OF REAL UNIFESP MOBILENETV2 MODEL ON DISJOINT VALIDATION SET
================================================================================
This script loads the trained model artifact:
  models/chest_validator/unifesp_mobilenet_v2_real.keras

And performs evaluation on the disjoint patient-level validation set (349 DICOM images).

Calculates:
  1. Chest-specific Precision, Recall, F1-score, and Confusion Matrix.
  2. Non-Chest Leakage Rate (False Acceptance Rate for Non-Chest).
  3. Chest False Rejection Rate.
  4. Per-Class Precision, Recall, and F1-score for all 22 UNIFESP classes.
  5. Multi-label threshold sweep to identify the optimal chest validation gate threshold.
  6. Saves evaluation reports to JSON and TXT.
================================================================================
"""

import os
import sys
import glob
import json
import time
import pydicom
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, multilabel_confusion_matrix
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

DATASET_DIR = r"C:\XR\archive (2)"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "chest_validator", "unifesp_mobilenet_v2_real.keras")
MAPPING_PATH = os.path.join(PROJECT_DIR, "models", "chest_validator", "unifesp_class_mapping.json")

EVAL_REPORT_JSON = os.path.join(PROJECT_DIR, "models", "chest_validator", "evaluation_report_unifesp_real.json")
EVAL_REPORT_TXT = os.path.join(PROJECT_DIR, "models", "chest_validator", "evaluation_report_unifesp_real.txt")

UNIFESP_22_CLASSES = [
    "Abdomen",        # 0
    "Ankle",          # 1
    "Cervical Spine", # 2
    "Chest",          # 3
    "Clavicle",       # 4
    "Elbow",          # 5
    "Feet",           # 6
    "Finger",         # 7
    "Forearm",        # 8
    "Hand",           # 9
    "Hip",            # 10
    "Knee",           # 11
    "Lower Leg",      # 12
    "Lumbar Spine",   # 13
    "Other",          # 14
    "Pelvis",         # 15
    "Radius",         # 16
    "Shoulder",       # 17
    "Skull",          # 18
    "Thigh",          # 19
    "Thoracic Spine", # 20
    "Wrist"           # 21
]

NUM_CLASSES = len(UNIFESP_22_CLASSES)
CHEST_CLASS_ID = 3  # Index of Chest in UNIFESP taxonomy


def load_dicom_image_array(fpath, target_size=(224, 224)):
    """Reads DICOM file, handles MONOCHROME1, applies CLAHE, resizes to 224x224."""
    ds = pydicom.dcmread(fpath)
    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    if slope != 1.0 or intercept != 0.0:
        img = img * slope + intercept

    photo = str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')).strip().upper()
    if photo == 'MONOCHROME1':
        img = np.max(img) - img

    min_val, max_val = np.min(img), np.max(img)
    if max_val > min_val:
        img_uint8 = ((img - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        img_uint8 = np.zeros(img.shape, dtype=np.uint8)

    img_resized = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)
    img_norm = img_clahe.astype(np.float32) / 255.0

    return np.stack([img_norm, img_norm, img_norm], axis=-1)


def get_validation_data():
    """Indexes DICOM files, parses train.csv, and extracts disjoint validation set."""
    df = pd.read_csv(TRAIN_CSV)
    df['SOPInstanceUID'] = df['SOPInstanceUID'].astype(str).str.strip()
    df['Target'] = df['Target'].astype(str).str.strip()

    dcm_files = glob.glob(os.path.join(TRAIN_DIR, "**", "*.dcm"), recursive=True)
    sop_to_path = {}
    sop_to_patient = {}

    for fpath in dcm_files:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            sop = str(getattr(ds, 'SOPInstanceUID', '')).strip()
            patient_id = str(getattr(ds, 'PatientID', 'UNKNOWN')).strip()
            if sop:
                sop_to_path[sop] = fpath
                sop_to_patient[sop] = patient_id
        except Exception:
            pass

    df['file_path'] = df['SOPInstanceUID'].map(sop_to_path)
    df['PatientID'] = df['SOPInstanceUID'].map(sop_to_patient)

    df_clean = df.dropna(subset=['file_path']).reset_index(drop=True)

    target_matrix = np.zeros((len(df_clean), NUM_CLASSES), dtype=np.float32)
    for idx, raw_target in enumerate(df_clean['Target']):
        label_ids = [int(x) for x in str(raw_target).split() if x.isdigit()]
        for lid in label_ids:
            if 0 <= lid < NUM_CLASSES:
                target_matrix[idx, lid] = 1.0

    # Re-create exact patient-level split (random_state=42)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    groups = df_clean['PatientID'].values
    train_idx, val_idx = next(gss.split(df_clean, target_matrix, groups=groups))

    df_val = df_clean.iloc[val_idx].reset_index(drop=True)
    y_val = target_matrix[val_idx]

    return df_val, y_val


def evaluate_model():
    print("=" * 80)
    print("EVALUATING REAL UNIFESP MOBILENETV2 MODEL ON DISJOINT VALIDATION SET")
    print("=" * 80)
    print("Loading model from:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

    df_val, y_val_true = get_validation_data()
    print(f"Validation dataset size: {len(df_val)} samples across {df_val['PatientID'].nunique()} unique patients.")

    print("\nRunning inference on validation DICOM images...")
    val_preds_list = []
    for idx, row in df_val.iterrows():
        img_3ch = load_dicom_image_array(row['file_path'], target_size=(224, 224))
        inp = np.expand_dims(img_3ch, axis=0)
        pred = model.predict(inp, verbose=0)[0]
        val_preds_list.append(pred)

    y_val_probs = np.array(val_preds_list, dtype=np.float32)

    # --------------------------------------------------------------------------
    # 1. CHEST-SPECIFIC EVALUATION ACROSS THRESHOLDS
    # --------------------------------------------------------------------------
    chest_true = y_val_true[:, CHEST_CLASS_ID].astype(int)
    chest_probs = y_val_probs[:, CHEST_CLASS_ID]

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    threshold_sweep_results = []

    print("\n" + "=" * 80)
    print("1. CHEST VALIDATION GATE THRESHOLD SWEEP ANALYSIS")
    print("=" * 80)
    print(f"{'Thresh':<7} | {'Chest Prec':<11} | {'Chest Rec':<11} | {'Chest F1':<10} | {'Non-Chest Leak %':<18} | {'Chest False Reject %':<20}")
    print("-" * 88)

    best_threshold = 0.50
    best_f1 = 0.0

    for thresh in thresholds:
        chest_pred_b = (chest_probs >= thresh).astype(int)
        
        # Confusion matrix for Chest class
        # TN: Non-chest correctly rejected
        # FP: Non-chest incorrectly accepted as chest (LEAKAGE)
        # FN: Chest incorrectly rejected as non-chest (FALSE REJECTION)
        # TP: Chest correctly accepted as chest
        tn, fp, fn, tp = confusion_matrix(chest_true, chest_pred_b, labels=[0, 1]).ravel()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        non_chest_total = tn + fp
        chest_total = tp + fn
        
        leakage_rate = (fp / non_chest_total) * 100.0 if non_chest_total > 0 else 0.0
        false_rejection_rate = (fn / chest_total) * 100.0 if chest_total > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
        res_dict = {
            "threshold": thresh,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "non_chest_leakage_rate_pct": round(float(leakage_rate), 2),
            "chest_false_rejection_rate_pct": round(float(false_rejection_rate), 2)
        }
        threshold_sweep_results.append(res_dict)
        
        print(f"{thresh:<7.2f} | {prec*100:<10.2f}% | {rec*100:<10.2f}% | {f1*100:<9.2f}% | {leakage_rate:<17.2f}% | {false_rejection_rate:<19.2f}%")

    print(f"\nOptimal Chest Threshold based on Validation F1-Score: {best_threshold:.2f} (F1 = {best_f1*100:.2f}%)")

    # --------------------------------------------------------------------------
    # 2. DETAILED METRICS AT DEFAULT 0.50 AND OPTIMAL THRESHOLDS
    # --------------------------------------------------------------------------
    chest_pred_default = (chest_probs >= 0.50).astype(int)
    tn_50, fp_50, fn_50, tp_50 = confusion_matrix(chest_true, chest_pred_default, labels=[0, 1]).ravel()
    prec_50 = precision_score(chest_true, chest_pred_default, zero_division=0)
    rec_50 = recall_score(chest_true, chest_pred_default, zero_division=0)
    f1_50 = f1_score(chest_true, chest_pred_default, zero_division=0)
    leakage_50 = (fp_50 / (tn_50 + fp_50)) * 100.0 if (tn_50 + fp_50) > 0 else 0.0
    false_reject_50 = (fn_50 / (tp_50 + fn_50)) * 100.0 if (tp_50 + fn_50) > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"2. CHEST-SPECIFIC METRICS AT DEFAULT THRESHOLD (0.50)")
    print("=" * 80)
    print(f"  • Chest Precision            : {prec_50*100:.2f}%")
    print(f"  • Chest Recall (Sensitivity) : {rec_50*100:.2f}%")
    print(f"  • Chest F1-Score             : {f1_50*100:.2f}%")
    print(f"  • Non-Chest Leakage Rate     : {leakage_50:.2f}% ({fp_50} non-chest images accepted out of {tn_50+fp_50})")
    print(f"  • Chest False Rejection Rate : {false_reject_50:.2f}% ({fn_50} chest images rejected out of {tp_50+fn_50})")
    print("\n  Chest Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"    [[{tn_50:<4} (Non-Chest Rejected), {fp_50:<4} (Non-Chest Accepted)],")
    print(f"     [{fn_50:<4} (Chest Rejected),     {tp_50:<4} (Chest Accepted)]]")

    # --------------------------------------------------------------------------
    # 3. PER-CLASS METRICS FOR ALL 22 UNIFESP CLASSES (Threshold = 0.50)
    # --------------------------------------------------------------------------
    y_pred_all_50 = (y_val_probs >= 0.50).astype(int)
    
    per_class_report = classification_report(
        y_val_true, y_pred_all_50,
        target_names=UNIFESP_22_CLASSES,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 80)
    print("3. PER-CLASS EVALUATION METRICS FOR ALL 22 UNIFESP CLASSES (Threshold = 0.50)")
    print("=" * 80)
    print(f"{'ID':<4} | {'UNIFESP Class Name':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 74)

    per_class_summary = {}
    for cid, cname in enumerate(UNIFESP_22_CLASSES):
        metrics = per_class_report[cname]
        p = metrics['precision']
        r = metrics['recall']
        f = metrics['f1-score']
        s = int(metrics['support'])
        per_class_summary[cname] = {
            "class_id": cid,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4),
            "support": s
        }
        print(f"{cid:<4} | {cname:<18} | {p*100:<9.2f}% | {r*100:<9.2f}% | {f*100:<9.2f}% | {s:<8}")

    overall_acc = accuracy_score(y_val_true, y_pred_all_50)
    macro_p = precision_score(y_val_true, y_pred_all_50, average="macro", zero_division=0)
    macro_r = recall_score(y_val_true, y_pred_all_50, average="macro", zero_division=0)
    macro_f1 = f1_score(y_val_true, y_pred_all_50, average="macro", zero_division=0)

    print("-" * 74)
    print(f"OVERALL METRICS across 22 Classes:")
    print(f"  • Overall Binary Accuracy : {overall_acc*100:.2f}%")
    print(f"  • Macro Precision         : {macro_p*100:.2f}%")
    print(f"  • Macro Recall            : {macro_r*100:.2f}%")
    print(f"  • Macro F1-Score          : {macro_f1*100:.2f}%")

    # --------------------------------------------------------------------------
    # 4. EXPLANATION OF MULTI-LABEL METRICS
    # --------------------------------------------------------------------------
    metrics_explanation = (
        "Multi-Label Classification Metric Explanation:\n"
        "1. Binary Accuracy: Percentage of correctly predicted individual label indicators across all 22 classes.\n"
        "2. Precision: Ratio of correctly predicted positive instances to total predicted positive instances (TP / (TP + FP)).\n"
        "3. Recall (Sensitivity): Ratio of correctly predicted positive instances to total actual positive instances (TP / (TP + FN)).\n"
        "4. F1-Score: Harmonic mean of Precision and Recall, balancing false acceptances and false rejections.\n"
        "5. Macro Average: Unweighted mean of metrics computed independently across all 22 classes, giving equal weight to rare anatomical categories."
    )

    # --------------------------------------------------------------------------
    # 5. SAVE EVALUATION REPORTS
    # --------------------------------------------------------------------------
    full_eval_data = {
        "model_evaluated": MODEL_PATH,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_samples_count": len(df_val),
        "validation_unique_patients": int(df_val['PatientID'].nunique()),
        "recommended_chest_threshold": best_threshold,
        "chest_metrics_at_0_50": {
            "precision": round(float(prec_50), 4),
            "recall": round(float(rec_50), 4),
            "f1_score": round(float(f1_50), 4),
            "non_chest_leakage_rate_pct": round(float(leakage_50), 2),
            "chest_false_rejection_rate_pct": round(float(false_reject_50), 2),
            "confusion_matrix": {
                "tn_non_chest_rejected": int(tn_50),
                "fp_non_chest_accepted": int(fp_50),
                "fn_chest_rejected": int(fn_50),
                "tp_chest_accepted": int(tp_50)
            }
        },
        "threshold_sweep_analysis": threshold_sweep_results,
        "per_class_22_metrics": per_class_summary,
        "overall_multi_label_metrics": {
            "binary_accuracy": round(float(overall_acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4)
        },
        "multi_label_metrics_explanation": metrics_explanation
    }

    with open(EVAL_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_eval_data, f, indent=2)
    print(f"\n[ARTIFACT] Saved evaluation report JSON to: {EVAL_REPORT_JSON}")

    # Write formatted TXT report
    with open(EVAL_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("UNIFESP MOBILENETV2 REAL MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Evaluated : {MODEL_PATH}\n")
        f.write(f"Timestamp       : {full_eval_data['evaluation_timestamp']}\n")
        f.write(f"Val Samples     : {len(df_val)} DICOM images ({df_val['PatientID'].nunique()} unique patients)\n\n")
        
        f.write("CHEST-SPECIFIC METRICS (Threshold = 0.50):\n")
        f.write(f"  - Precision            : {prec_50*100:.2f}%\n")
        f.write(f"  - Recall (Sensitivity) : {rec_50*100:.2f}%\n")
        f.write(f"  - F1-Score             : {f1_50*100:.2f}%\n")
        f.write(f"  - Non-Chest Leakage    : {leakage_50:.2f}%\n")
        f.write(f"  - Chest False Rejection: {false_reject_50:.2f}%\n\n")
        
        f.write("CONFUSION MATRIX (Chest vs Non-Chest):\n")
        f.write(f"  TN (Non-Chest Rejected): {tn_50}\n")
        f.write(f"  FP (Non-Chest Accepted): {fp_50}\n")
        f.write(f"  FN (Chest Rejected)    : {fn_50}\n")
        f.write(f"  TP (Chest Accepted)    : {tp_50}\n\n")
        
        f.write("PER-CLASS METRICS FOR ALL 22 CLASSES:\n")
        for cname, m in per_class_summary.items():
            f.write(f"  - {cname:<18}: Prec={m['precision']*100:6.2f}%, Rec={m['recall']*100:6.2f}%, F1={m['f1_score']*100:6.2f}%, Support={m['support']}\n")
            
        f.write("\nRECOMMENDED CHEST GATE THRESHOLD: " + str(best_threshold) + "\n")

    print(f"[ARTIFACT] Saved evaluation report TXT to: {EVAL_REPORT_TXT}")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_model()
