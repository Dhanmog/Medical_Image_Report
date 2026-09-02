"""
================================================================================
STANFORD CHEXAGENT-8B RADIOLOGY REPORT GENERATOR
================================================================================
State-of-the-Art Medical Vision-Language Model for Chest X-Ray Diagnostic Reporting.
Optimized for NVIDIA GeForce RTX 2050 (4GB VRAM) with Auto Device Offloading.
================================================================================
"""

import os
import sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "05_chexagent_8b")
DEFAULT_IMAGE = os.path.join(PROJECT_DIR, "xray1.jpg")

def generate_report(image_path=DEFAULT_IMAGE, prompt="Describe the findings and impression for this chest radiograph."):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None

    print("=" * 70)
    print("  STANFORD CHEXAGENT-8B RADIOLOGY REPORT GENERATION")
    print(f"  Model Path   : {MODEL_DIR}")
    print(f"  Target Image : {image_path}")
    print(f"  Input Prompt : {prompt}")
    print("=" * 70)

    print("\n[1/4] Initializing Processor and Generation Configuration...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(MODEL_DIR)
    generation_config.max_new_tokens = 512
    generation_config.do_sample = False
    generation_config.num_beams = 1

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[2/4] Loading CheXagent-8b Model weights (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True
    )

    print(f"[3/4] Preprocessing input image: {os.path.basename(image_path)}...")
    image = Image.open(image_path).convert("RGB")
    formatted_prompt = f" USER: <s>{prompt} ASSISTANT: <s>"

    inputs = processor(images=[image], text=formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    print("[4/4] Generating clinical diagnostic report...")
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=generation_config)[0]

    raw_response = processor.tokenizer.decode(output, skip_special_tokens=True)
    report_clean = raw_response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw_response else raw_response.strip()

    print("\n" + "=" * 70)
    print("  CLINICAL RADIOLOGY REPORT:")
    print("=" * 70)
    print(report_clean)
    print("=" * 70 + "\n")
    return report_clean

if __name__ == "__main__":
    img_target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    user_prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe the findings and impression for this chest radiograph."
    generate_report(img_target, user_prompt)
