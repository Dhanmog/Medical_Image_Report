"""
================================================================================
CHEST AI: REPORT-TO-SYNTHETIC-CHEST-XRAY DIFFUSION INFERENCE ENGINE
================================================================================
Orchestrates genuine Hugging Face diffusion inference from CHEST AI clinical reports.
Extracts structured prompt conditioning, manages seed reproducibility, validates
output integrity, stamps research watermarks, and writes verifiable audit metadata.
================================================================================
"""

import os
import json
import time
import logging
import torch
from datetime import datetime
from typing import Optional, Dict, Any, Union
from PIL import Image

from .config import DiffusionConfig, default_config
from .prompt_processor import adapt_report_to_diffusion_prompt
from .model_loader import load_diffusion_pipeline
from .image_validator import validate_synthetic_image, apply_synthetic_watermark

logger = logging.getLogger("CHEST_AI.text_to_xray.diffusion_engine")


def generate_synthetic_xray(
    report: Union[str, Dict[str, Any]],
    view: Optional[str] = None,
    seed: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    config: Optional[DiffusionConfig] = None,
    apply_watermark: bool = True
) -> Dict[str, Any]:
    """
    Generates a synthetic chest radiograph conditioned on a clinical CHEST AI report.

    Args:
        report: A string report or dictionary containing 'findings', 'impression', or 'full_text'.
        view: Optional radiograph projection ('Frontal PA', 'Frontal AP', 'Lateral', 'Unknown').
        seed: Optional integer seed for deterministic, reproducible generation.
        num_inference_steps: Number of diffusion denoising steps (defaults to config value, e.g. 25).
        guidance_scale: Classifier-free guidance scale (defaults to config value, e.g. 7.0).
        config: Optional DiffusionConfig instance.
        apply_watermark: If True, adds an in-image research disclaimer banner.

    Returns:
        Dict containing:
          - 'success': bool
          - 'image': PIL.Image (watermarked synthetic radiograph)
          - 'raw_image': PIL.Image (unwatermarked radiograph)
          - 'image_path': Path to saved image file
          - 'metadata_path': Path to saved metadata JSON file
          - 'metadata': Full generation metadata dictionary
          - 'error': Error description string if generation failed
    """
    cfg = config or default_config

    # Step 1: Validate input report
    is_empty_text = isinstance(report, str) and not report.strip()
    is_empty_dict = isinstance(report, dict) and not any(bool(str(v).strip()) for v in report.values() if v is not None)
    if not report or is_empty_text or is_empty_dict:
        return {
            "success": False,
            "image": None,
            "raw_image": None,
            "image_path": None,
            "metadata_path": None,
            "metadata": {},
            "error": "Empty report received. A clinical report text or dictionary is required."
        }

    # Step 2: Convert clinical report to calibrated diffusion prompts
    prompt_bundle = adapt_report_to_diffusion_prompt(report, user_projection=view)
    pos_prompt = prompt_bundle["positive_prompt"]
    neg_prompt = prompt_bundle["negative_prompt"]
    parsed_report = prompt_bundle["parsed_report"]
    projection_used = prompt_bundle["projection_used"]

    # Step 3: Determine generation parameters
    steps = num_inference_steps if num_inference_steps is not None else cfg.num_inference_steps
    guidance = guidance_scale if guidance_scale is not None else cfg.guidance_scale
    gen_seed = seed if seed is not None else cfg.default_seed

    # Step 4: Lazy-load pretrained diffusion pipeline
    pipeline, load_err = load_diffusion_pipeline(cfg)
    if load_err or pipeline is None:
        return {
            "success": False,
            "image": None,
            "raw_image": None,
            "image_path": None,
            "metadata_path": None,
            "metadata": {},
            "error": f"Failed to initialize diffusion pipeline: {load_err}"
        }

    # Step 5: Initialize reproducible random generator
    # Note: When CPU offloading is active in Diffusers, CPU generator is universally compatible
    generator = torch.Generator(device="cpu").manual_seed(gen_seed)

    start_time = time.time()
    logger.info(f"Generating synthetic X-ray (Seed={gen_seed}, Steps={steps}, View={projection_used})...")

    raw_image = None
    try:
        # Run diffusion denoising pass
        with torch.inference_mode():
            output = pipeline(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=cfg.height,
                width=cfg.width,
                generator=generator
            )
            raw_image = output.images[0]

    except torch.cuda.OutOfMemoryError as oom_e:
        err_msg = (
            f"CUDA Out of Memory during diffusion generation: {oom_e}. "
            "Try reducing image dimensions, steps, or enabling CPU execution."
        )
        logger.error(err_msg)
        return {
            "success": False,
            "image": None,
            "raw_image": None,
            "image_path": None,
            "metadata_path": None,
            "metadata": {},
            "error": err_msg
        }
    except Exception as run_e:
        err_msg = f"Diffusion inference encountered an error: {run_e}"
        logger.error(err_msg)
        return {
            "success": False,
            "image": None,
            "raw_image": None,
            "image_path": None,
            "metadata_path": None,
            "metadata": {},
            "error": err_msg
        }

    duration_sec = round(time.time() - start_time, 2)

    # Step 6: Perform technical integrity validation
    is_valid, val_err, tech_metrics = validate_synthetic_image(raw_image)
    if not is_valid:
        return {
            "success": False,
            "image": None,
            "raw_image": raw_image,
            "image_path": None,
            "metadata_path": None,
            "metadata": {"technical_validation": tech_metrics},
            "error": f"Generated image failed technical validation: {val_err}"
        }

    # Step 7: Apply prominent synthetic research watermark
    final_image = apply_synthetic_watermark(raw_image) if apply_watermark else raw_image

    # Step 8: Save synthetic image and audit metadata safely
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = f"synthetic_cxr_{timestamp_str}_s{gen_seed}"
    
    img_save_path = os.path.join(cfg.images_dir, f"{file_id}.png")
    meta_save_path = os.path.join(cfg.metadata_dir, f"{file_id}.json")

    final_image.save(img_save_path, format="PNG")

    metadata = {
        "generation_id": file_id,
        "timestamp": datetime.now().isoformat(),
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "generation_parameters": {
            "seed": gen_seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "height": cfg.height,
            "width": cfg.width,
            "projection": projection_used,
            "duration_seconds": duration_sec
        },
        "conditioning": {
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "source_report_summary": parsed_report.get("full_text", "")[:300],
            "extracted_positive_findings": parsed_report.get("positive_findings", []),
            "extracted_negative_findings": parsed_report.get("negative_findings", []),
            "laterality": parsed_report.get("laterality", {}),
            "severity": parsed_report.get("severity", {}),
            "uncertainty": parsed_report.get("uncertainty", {})
        },
        "technical_validation": tech_metrics,
        "safety_disclaimer": cfg.safety_disclaimer,
        "is_synthetic": True
    }

    with open(meta_save_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    logger.info(f"Synthetic radiograph successfully generated and saved to {img_save_path}")

    return {
        "success": True,
        "image": final_image,
        "raw_image": raw_image,
        "image_path": img_save_path,
        "metadata_path": meta_save_path,
        "metadata": metadata,
        "error": None
    }
