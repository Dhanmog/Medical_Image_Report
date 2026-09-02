import os
import sys
import json
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

from app import PositionalEmbedding, TransformerDecoderBlock, WarmupLRSchedule, preprocess_image_bytes, generate_report

MODEL_DIR = os.path.join(PROJECT_DIR, "models", "02_densenet121_mimic_v3")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
VOCAB_PATH = os.path.join(MODEL_DIR, "tokenizer_vocab.json")
VAL_CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "mimic_cxr", "processed", "processed_dataset_v1", "validation.csv")
ARTIFACTS_DIR = r"C:\Users\dhanu\.gemini\antigravity-ide\brain\fc0f1678-3e23-4626-ae43-cfb1399eeb5c"

def main():
    print("=" * 60)
    print("BASELINE MODEL EVALUATION & INFERENCE TEST")
    print("=" * 60)

    # 1. Check if model and vocab files exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(VOCAB_PATH):
        print(f"ERROR: Vocabulary file not found at {VOCAB_PATH}")
        sys.exit(1)

    # 2. Load model and vocab
    print("Loading model and vocabulary...")
    custom_objects = {
        "PositionalEmbedding":            PositionalEmbedding,
        "TransformerDecoderBlock":        TransformerDecoderBlock,
        "Custom>PositionalEmbedding":     PositionalEmbedding,
        "Custom>TransformerDecoderBlock": TransformerDecoderBlock,
        "WarmupLRSchedule":               WarmupLRSchedule,
    }
    
    try:
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print("Model and vocabulary loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model/vocab: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Load validation dataset
    if not os.path.exists(VAL_CSV_PATH):
        print(f"ERROR: Validation CSV not found at {VAL_CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(VAL_CSV_PATH)
    print(f"Loaded validation set: {len(df)} records.")

    # 4. Run inference on 5 different X-rays
    print("\nRunning inference on 5 different chest X-rays...")
    results = []
    
    # We will pick 5 rows that have valid images
    valid_count = 0
    for idx, row in df.iterrows():
        img_path = row.get("image_path")
        report = row.get("report")
        
        if not os.path.exists(img_path) or os.path.getsize(img_path) < 100:
            continue
            
        print(f"Processing image {valid_count + 1}/5: {os.path.basename(img_path)}")
        try:
            # Read image bytes
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                
            preprocessed_img = preprocess_image_bytes(img_bytes)
            
            # Generate report using greedy search (temp = 0.0)
            gen_report = generate_report(model, vocab, preprocessed_img, temperature=0.0)
            
            results.append({
                "patient_id": row.get("subject_id", "N/A"),
                "study_id": row.get("study_id", "N/A"),
                "view": row.get("view", "N/A"),
                "image_name": os.path.basename(img_path),
                "actual_report": report,
                "generated_report": gen_report
            })
            
            valid_count += 1
            if valid_count >= 5:
                break
        except Exception as e:
            print(f"  ⚠️ Error during inference: {e}")

    # 5. Save results to baseline_eval_results.md
    report_path = os.path.join(ARTIFACTS_DIR, "baseline_eval_results.md")
    
    md_content = """# Baseline Model Evaluation Results

This report establishes the performance baseline of the **DenseNet121 v3** model on 5 distinct chest X-rays before making architectural changes.

## ⚙️ Model Configurations
* **Base Encoder:** DenseNet121 (weights initialized from ImageNet)
* **Text Decoder:** Transformer Decoder (3 Blocks, 8 Heads, 512 Embed Dim)
* **Total Parameters:** ~28.44M parameters
* **Best Validation Accuracy (Phase 3):** ~26.49%

---

## 🔍 Generation Testing (5 Different X-Rays)
We evaluate if the model produces image-dependent outputs and compare them against the ground-truth reports.

"""

    for i, res in enumerate(results):
        md_content += f"""### Test Case {i+1}
* **Patient ID:** {res['patient_id']}
* **Study ID:** {res['study_id']}
* **View:** {res['view']}
* **Image File:** `{res['image_name']}`

**Ground Truth Report:**
> {res['actual_report']}

**Model Generated Report (Baseline - Greedy):**
> **{res['generated_report']}**

---
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\n[OK] Baseline evaluation report saved to: {report_path}")
    print("Baseline test completed successfully!")

if __name__ == "__main__":
    main()
