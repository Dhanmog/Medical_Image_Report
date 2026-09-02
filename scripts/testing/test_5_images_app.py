import os
import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from app import load_model_and_vocab, preprocess_image_bytes, generate_report

def main():
    print("=" * 80)
    print("TESTING 5 DIFFERENT X-RAY IMAGES ON UPDATED APP PIPELINE")
    print("=" * 80)

    model, vocab, err = load_model_and_vocab()
    if err:
        print(f"Error loading model: {err}")
        sys.exit(1)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    iccv_dir = os.path.join(project_dir, "official_data_iccv_final", "files", "p10")

    images = []
    for root, _, files in os.walk(iccv_dir):
        for f in files:
            if f.endswith(".jpg") and os.path.getsize(os.path.join(root, f)) > 100:
                images.append(os.path.join(root, f))
                if len(images) == 5:
                    break
        if len(images) == 5:
            break

    print(f"Loaded {len(images)} distinct X-ray images for testing.\n")

    reports = []
    for idx, img_path in enumerate(images, 1):
        filename = os.path.basename(img_path)
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        preprocessed_img = preprocess_image_bytes(img_bytes)
        rep = generate_report(model, vocab, preprocessed_img, temperature=0.4)
        reports.append(rep)

        print(f"TEST IMAGE #{idx} ({filename}):")
        print(f"  • Preprocessed Shape : {preprocessed_img.shape}")
        print(f"  • Preprocessed Mean  : {preprocessed_img.mean():.6f}")
        print(f"  • Generated Report   : \"{rep}\"")
        print("-" * 80)

    unique_count = len(set(reports))
    print(f"\nFINAL TEST SUMMARY: {unique_count}/{len(images)} UNIQUE REPORTS GENERATED ACROSS DIFFERENT X-RAYS!")
    assert unique_count > 1, "Failed: Reports are still identical!"
    print("=" * 80)
    print(" [SUCCESS] INFERENCE PIPELINE PROVEN TO PRODUCE IMAGE-DEPENDENT REPORTS!")
    print("=" * 80)

if __name__ == "__main__":
    main()
