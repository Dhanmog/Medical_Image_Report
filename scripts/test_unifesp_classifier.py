r"""
================================================================================
SMOKE-TEST FOR TRAINED UNIFESP ANATOMICAL BODY-PART CLASSIFIER
================================================================================
This script tests the newly trained UNIFESP MobileNetV2 classifier on:
  1. One sample Chest X-ray from the validation dataset
  2. One sample Non-Chest X-ray from the validation dataset

Model Artifact : models/chest_validator/unifesp_mobilenet_v2_real.keras
Class Mapping  : models/chest_validator/unifesp_class_mapping.json
Decision Threshold: 0.60 (60%) for Chest validation gate

Notice: This functional smoke-test demonstrates single-instance pipeline execution.
It does NOT constitute clinical reliability or diagnostic certification.
================================================================================
"""

import os
import sys
import glob
import json
import pydicom
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GroupShuffleSplit

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

DATASET_DIR = r"C:\XR\archive (2)"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "chest_validator", "unifesp_mobilenet_v2_real.keras")
MAPPING_PATH = os.path.join(PROJECT_DIR, "models", "chest_validator", "unifesp_class_mapping.json")

CHEST_THRESHOLD = 0.60
CHEST_CLASS_ID = 3

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


def load_dicom_image_array(fpath, target_size=(224, 224)):
    """
    Exact training preprocessing pipeline:
    pydicom read -> Rescale Slope/Intercept -> MONOCHROME1 inversion -> CLAHE -> 224x224 RGB [0,1].
    """
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


def run_smoke_test():
    print("=" * 80)
    print("UNIFESP MOBILENETV2 BODY-PART CLASSIFIER SMOKE TEST")
    print("=" * 80)
    print("Loading model from:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model artifact not found at: {MODEL_PATH}")
        sys.exit(1)

    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

    # Load class mapping
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
        print(f"Loaded class mapping: {len(class_mapping)} classes.")
    else:
        class_mapping = {str(i): name for i, name in enumerate(UNIFESP_22_CLASSES)}

    # Index dataset metadata
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

    # Isolate disjoint validation set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    groups = df_clean['PatientID'].values
    _, val_idx = next(gss.split(df_clean, target_matrix, groups=groups))

    df_val = df_clean.iloc[val_idx].reset_index(drop=True)

    # Select 1 Chest sample (Target == '3') and 1 Non-Chest sample (Target != '3')
    chest_samples = df_val[df_val['Target'] == '3'].reset_index(drop=True)
    non_chest_samples = df_val[~df_val['Target'].str.contains('3')].reset_index(drop=True)

    if len(chest_samples) == 0 or len(non_chest_samples) == 0:
        print("[ERROR] Could not isolate validation samples.")
        sys.exit(1)

    sample_chest = chest_samples.iloc[0]
    sample_non_chest = non_chest_samples.iloc[0]

    test_samples = [
        ("SAMPLE 1: GROUND-TRUTH CHEST RADIOGRAPH", sample_chest),
        ("SAMPLE 2: GROUND-TRUTH NON-CHEST RADIOGRAPH", sample_non_chest)
    ]

    for title, row in test_samples:
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)
        print(f"  * SOPInstanceUID : {row['SOPInstanceUID']}")
        print(f"  * PatientID      : {row['PatientID']}")
        print(f"  * Ground Target  : '{row['Target']}'")
        
        # Ground truth class names
        gt_ids = [int(x) for x in row['Target'].split() if x.isdigit()]
        gt_names = [UNIFESP_22_CLASSES[i] for i in gt_ids if i < NUM_CLASSES]
        print(f"  * Ground Classes : {', '.join(gt_names)}")

        # Preprocess DICOM image
        img_3ch = load_dicom_image_array(row['file_path'], target_size=(224, 224))
        inp = np.expand_dims(img_3ch, axis=0)

        # Run model inference
        probs = model.predict(inp, verbose=0)[0]

        # Top predicted classes (threshold = 0.50)
        sorted_indices = np.argsort(probs)[::-1]
        top_preds = [(UNIFESP_22_CLASSES[i], probs[i]) for i in sorted_indices[:5]]

        print("\n  Top-5 Predicted Anatomical Classes:")
        for rank, (cname, prob) in enumerate(top_preds, 1):
            flag = "[POSITIVE]" if prob >= 0.50 else "          "
            print(f"    {rank}. {cname:<18} : {prob*100:6.2f}% {flag}")

        # Chest Validation Gate Decision at threshold 0.60
        chest_prob = probs[CHEST_CLASS_ID]
        is_accepted = chest_prob >= CHEST_THRESHOLD
        decision_str = "ACCEPTED AS CHEST (Passed Gate)" if is_accepted else "REJECTED (Non-Chest Halted)"

        print("\n  Chest Validation Gate Decision (Threshold = 0.60):")
        print(f"    * Chest Class Probability : {chest_prob*100:.2f}%")
        print(f"    * Configured Threshold    : {CHEST_THRESHOLD*100:.0f}%")
        print(f"    * Final Decision          : {decision_str}")

    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("Notice: This smoke-test evaluates 2 individual sample images for functional")
    print("verification and does NOT constitute clinical reliability or diagnostic certification.")


if __name__ == "__main__":
    run_smoke_test()
