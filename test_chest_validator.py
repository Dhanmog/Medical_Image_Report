"""
================================================================================
UNIFESP X-RAY BODY PART CLASSIFICATION & CHEST VALIDATOR EVALUATION SUITE
================================================================================
This script evaluates the UNIFESP X-Ray Body Part Classification stage on:
  1. Real Chest X-Ray Images (xray1.jpg, top5_trained_images, etc.)
  2. UNIFESP Non-Chest Body Part X-Ray Categories (Abdomen, Skull, Spine, Upper Limb, Lower Limb, Pelvis, Other)
  3. Ambiguous / Uncertain anatomical inputs

It verifies that:
  - CHEST X-rays pass the validation gate and proceed to DenseNet121 disease classification.
  - NON-CHEST UNIFESP categories trigger the pipeline halt and rejection block.
  - UNCERTAIN anatomical predictions stop report generation for safety.
  - ZERO non-chest images leak into the chest pathology report generation pipeline.
================================================================================
"""

import os
import sys
import glob
import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from chest_validator import validate_chest_xray, DEFAULT_CHEST_THRESHOLD
from scripts.build_chest_validator_model import generate_synthetic_unifesp_body_part


def test_validator_on_chest_images():
    print("\n" + "="*70)
    print("1. TESTING CHEST X-RAY VALIDATION (EXPECTED: CHEST VALIDATED)")
    print("="*70)

    chest_files = glob.glob(os.path.join(PROJECT_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(PROJECT_DIR, "top5_trained_images", "*.jpg"))

    passed, failed = 0, 0

    for fpath in chest_files:
        if "preview" in fpath.lower() or "flowchart" in fpath.lower():
            continue
        with open(fpath, "rb") as f:
            img_bytes = f.read()

        is_chest, conf, label, details = validate_chest_xray(img_bytes, threshold=DEFAULT_CHEST_THRESHOLD)
        fname = os.path.basename(fpath)

        if is_chest and not details.get("is_uncertain", False):
            passed += 1
            print(f"  [PASS] {fname:<30} | UNIFESP Category: {label:<11} | Confidence: {conf*100:.1f}% | Certainty: CERTAIN")
        else:
            failed += 1
            reason = details.get("uncertainty_reason", "Below threshold or non-chest")
            print(f"  [FAIL] {fname:<30} | UNIFESP Category: {label:<11} | Confidence: {conf*100:.1f}% | Reason: {reason}")

    print(f"\nChest X-Ray Validation Result: {passed} PASSED, {failed} FAILED")
    return passed, failed


def test_validator_on_unifesp_non_chest_images():
    print("\n" + "="*70)
    print("2. TESTING UNIFESP NON-CHEST BODY PART REJECTION (EXPECTED: PIPELINE HALTED)")
    print("="*70)

    unifesp_categories = ["ABDOMEN", "SKULL", "SPINE", "UPPER_LIMB", "LOWER_LIMB", "PELVIS", "OTHER"]
    rejected, leaked = 0, 0

    for ntype in unifesp_categories:
        for i in range(3):
            syn_rgb = generate_synthetic_unifesp_body_part(ntype)
            _, encoded_img = cv2.imencode('.jpg', syn_rgb)
            img_bytes = encoded_img.tobytes()

            is_chest, conf, label, details = validate_chest_xray(img_bytes, threshold=DEFAULT_CHEST_THRESHOLD)

            if not is_chest:
                rejected += 1
                unc_status = "UNCERTAIN" if details.get("is_uncertain") else "CERTAIN"
                print(f"  [BLOCKED] UNIFESP {ntype:<10} (Sample {i+1}) | Predicted: {label:<10} | Confidence: {conf*100:.1f}% | Status: {unc_status} -> REJECTED")
            else:
                leaked += 1
                print(f"  [LEAK!]   UNIFESP {ntype:<10} (Sample {i+1}) | Predicted: {label:<10} | Confidence: {conf*100:.1f}% | LEAKED!")

    print(f"\nUNIFESP Non-Chest Rejection Result: {rejected} REJECTED SUCCESSFULLY, {leaked} LEAKED")
    return rejected, leaked


def test_validator_on_uncertain_inputs():
    print("\n" + "="*70)
    print("3. TESTING ANATOMICAL UNCERTAINTY HANDLING (EXPECTED: NO REPORT GENERATED)")
    print("="*70)

    # Generate a noisy hybrid image that simulates an ambiguous/unclear X-ray
    h, w = 224, 224
    ambiguous_img = np.random.randint(40, 180, (h, w, 3), dtype=np.uint8)
    cv2.circle(ambiguous_img, (112, 112), 80, (200, 200, 200), -1)
    _, encoded = cv2.imencode('.jpg', ambiguous_img)
    img_bytes = encoded.tobytes()

    is_chest, conf, label, details = validate_chest_xray(img_bytes, threshold=DEFAULT_CHEST_THRESHOLD)

    print(f"  Ambiguous Input -> Detected Category: {label} | Confidence: {conf*100:.1f}% | Is Chest Valid: {is_chest} | Is Uncertain: {details.get('is_uncertain')}")
    print(f"  Uncertainty Reason: {details.get('uncertainty_reason')}")

    if not is_chest:
        print("  [SUCCESS] Ambiguous prediction halted report generation safely.")
        return 1, 0
    else:
        print("  [FAIL] Ambiguous prediction allowed report generation.")
        return 0, 1


def main():
    print("\nSTARTING UNIFESP BODY PART CLASSIFIER & CHEST VALIDATOR SAFETY AUDIT...")

    chest_pass, chest_fail = test_validator_on_chest_images()
    nc_reject, nc_leak = test_validator_on_unifesp_non_chest_images()
    unc_pass, unc_fail = test_validator_on_uncertain_inputs()

    total_nc = nc_reject + nc_leak
    print("\n" + "="*70)
    print("FINAL UNIFESP BODY PART CLASSIFICATION AUDIT REPORT")
    print("="*70)
    print(f"  - Chest Detection Sensitivity (Sensitivity): {chest_pass / (chest_pass + chest_fail + 1e-5) * 100:.1f}%")
    print(f"  - Non-Chest Rejection Specificity (Specificity): {nc_reject / (total_nc + 1e-5) * 100:.1f}%")
    print(f"  - Non-Chest Leakage Rate to DenseNet121 Pipeline: {nc_leak / (total_nc + 1e-5) * 100:.1f}%")
    print(f"  - Ambiguity & Uncertainty Gate Protection: {'PASSED' if unc_pass == 1 else 'FAILED'}")

    if nc_leak == 0 and unc_pass == 1:
        print("\nCLINICAL SAFETY MANDATE MET: UNIFESP body-part classification successfully filters non-chest X-rays and uncertain images prior to DenseNet121 disease detection!")
    else:
        print("\nWARNING: Verification flags detected.")


if __name__ == "__main__":
    main()
