import os
import sys
import numpy as np
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import functions directly from app.py
from app import load_model_and_vocab, preprocess_image_bytes, generate_report

def main():
    print("=" * 70)
    print("VERIFYING APP.PY MODEL LOADING & REPORT GENERATION")
    print("=" * 70)

    model, vocab, err = load_model_and_vocab()
    if err:
        print(f"[FAIL] app.py load_model_and_vocab returned error:\n{err}")
        sys.exit(1)

    print(" [OK] load_model_and_vocab loaded successfully without errors!")
    print(f"  • Vocabulary size : {len(vocab)} tokens")
    print(f"  • Model input shape: {[i.shape for i in model.inputs]}")

    # Test with an image file
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sample_img_path = os.path.join(project_dir, "xray1.jpg")
    if not os.path.exists(sample_img_path):
        iccv_path = os.path.join(project_dir, "official_data_iccv_final", "files", "p10")
        for root, _, files in os.walk(iccv_path):
            for f in files:
                if f.endswith(".jpg") and os.path.getsize(os.path.join(root, f)) > 100:
                    sample_img_path = os.path.join(root, f)
                    break
            if os.path.exists(sample_img_path):
                break

    print(f"Testing image: {sample_img_path}")
    with open(sample_img_path, "rb") as f:
        image_bytes = f.read()

    preprocessed_img = preprocess_image_bytes(image_bytes)
    print(" [OK] Image bytes preprocessed successfully (CLAHE & 224x224 normalization).")

    report_text = generate_report(model, vocab, preprocessed_img)
    print("\nGENERATED DIAGNOSTIC REPORT FROM APP PIPELINE:")
    print(f'"{report_text}"')
    print("\n" + "=" * 70)
    print(" [SUCCESS] APP MODEL LOAD & INFERENCE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
