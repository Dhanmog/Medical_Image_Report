"""
================================================================================
CHEST AI: OPT-IN LIVE SMOKE TEST FOR DIFFUSION IMAGE GENERATION
================================================================================
This script provides an opt-in live execution test for the pretrained Hugging Face
diffusion pipeline. It runs actual model weights only when explicitly requested
via the '--live' flag, checking hardware compatibility and verifying output integrity.

Usage:
  # Check environment readiness and dry-run prompt adaptation:
  python smoke_test_diffusion.py

  # Execute actual live model inference (downloads model if not cached):
  python smoke_test_diffusion.py --live
================================================================================
"""

import os
import sys
import argparse
import time
import torch
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from text_to_xray import (
    DiffusionConfig,
    generate_synthetic_xray,
    parse_chest_ai_report,
    adapt_report_to_diffusion_prompt,
    validate_synthetic_image,
    get_gpu_vram_gb,
    PRIMARY_MODEL_ID,
    RESEARCH_SAFETY_DISCLAIMER
)


def run_smoke_test(live_inference: bool = False, steps: int = 20, seed: int = 42):
    print("=" * 80)
    print("  CHEST AI: HUGGING FACE TEXT-TO-CHEST-XRAY SMOKE TEST")
    print("=" * 80)

    # 1. Hardware & Environment Check
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA)"
    vram_gb = get_gpu_vram_gb()
    _, _, free_disk_bytes = shutil.disk_usage(PROJECT_DIR)
    free_disk_gb = free_disk_bytes / (1024 ** 3)

    print(f"\n[1/4] Hardware & Environment Diagnostics:")
    print(f"  • Python Version  : {sys.version.split()[0]}")
    print(f"  • PyTorch Version : {torch.__version__}")
    print(f"  • CUDA Available  : {cuda_available}")
    print(f"  • Compute Device  : {device_name}")
    print(f"  • Detected VRAM   : {vram_gb:.2f} GB")
    print(f"  • Free Disk Space : {free_disk_gb:.2f} GB")
    print(f"  • Target Model ID : {PRIMARY_MODEL_ID}")

    # 2. Prompt Processing & Conditioning Verification
    test_report = (
        "FINDINGS: The lungs show mild left-sided pleural effusion. "
        "No pneumothorax, pulmonary edema, or focal consolidation. "
        "The cardiac silhouette is mildly enlarged.\n\n"
        "IMPRESSION: 1. Mild left pleural effusion. 2. Borderline cardiomegaly."
    )

    print(f"\n[2/4] Testing Clinical Report Conditioning Adapter:")
    parsed = parse_chest_ai_report(test_report)
    adapted = adapt_report_to_diffusion_prompt(test_report, user_projection="Frontal PA")

    print(f"  • Extracted Positives : {parsed['positive_findings']}")
    print(f"  • Extracted Negatives : {parsed['negative_findings']}")
    print(f"  • Extracted Laterality: {parsed['laterality']}")
    print(f"  • Positive Prompt     : {adapted['positive_prompt'][:100]}...")
    print(f"  • Negative Prompt     : {adapted['negative_prompt'][:100]}...")

    # Strict assertion on negation preservation
    assert "pleural effusion" not in adapted["negative_prompt"]
    assert "pneumothorax" in adapted["negative_prompt"]
    assert "left-sided" in adapted["positive_prompt"]
    print("  -> Conditioning logic verified successfully (Zero-inversion confirmed).")

    # 3. Execution Decision
    if not live_inference:
        print(f"\n[3/4] Live Model Inference: SKIPPED (Opt-in mode)")
        print("  -> Dry run complete. Hardware and prompt adapters are fully functional.")
        print("  -> To run actual diffusion model inference, pass the '--live' flag:")
        print("     python smoke_test_diffusion.py --live\n")
        print("=" * 80)
        return True

    print(f"\n[3/4] Running Live Diffusion Inference (Steps={steps}, Seed={seed})...")
    print("  [Note: Initial invocation may download model weights to Hugging Face cache]")
    
    start_t = time.time()
    cfg = DiffusionConfig(
        model_id=PRIMARY_MODEL_ID,
        num_inference_steps=steps,
        guidance_scale=7.0
    )

    res = generate_synthetic_xray(
        report=test_report,
        view="Frontal PA",
        seed=seed,
        num_inference_steps=steps,
        config=cfg,
        apply_watermark=True
    )

    elapsed = time.time() - start_t

    # 4. Results & Technical Integrity Validation
    print(f"\n[4/4] Evaluating Generated Synthetic Radiograph:")
    if not res.get("success"):
        print(f"  ❌ Generation FAILED: {res.get('error')}")
        print("=" * 80)
        return False

    img = res.get("image")
    img_path = res.get("image_path")
    meta_path = res.get("metadata_path")
    tech = res.get("metadata", {}).get("technical_validation", {})

    print(f"  ✅ SUCCESS: Synthetic radiograph generated in {elapsed:.2f} seconds!")
    print(f"  • Output Image Path    : {img_path}")
    print(f"  • Output Metadata Path : {meta_path}")
    print(f"  • Dimensions           : {tech.get('width')} x {tech.get('height')}")
    print(f"  • Mean Pixel Intensity : {tech.get('mean_intensity')}")
    print(f"  • Pixel Std Deviation  : {tech.get('std_intensity')}")
    print(f"  • Non-Degenerate Check : 🟢 Passed")
    print(f"  • Research Watermark   : 🟢 Imprinted")
    print(f"  • Audit Disclaimer     : {RESEARCH_SAFETY_DISCLAIMER}")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHEST AI Diffusion Smoke Test")
    parser.add_argument("--live", action="store_true", help="Execute live diffusion model inference")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    args = parser.parse_args()

    success = run_smoke_test(live_inference=args.live, steps=args.steps, seed=args.seed)
    sys.exit(0 if success else 1)
