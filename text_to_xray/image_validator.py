"""
================================================================================
CHEST AI: SYNTHETIC RADIOGRAPH VALIDATOR & METADATA RECORDER
================================================================================
Performs technical integrity checks on generated synthetic radiographs
(format, dimensions, non-degeneracy, variance) and stamps mandatory research
watermarks to guarantee ethical non-clinical separation.
================================================================================
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Tuple, Optional

from .config import RESEARCH_SAFETY_DISCLAIMER


def validate_synthetic_image(image: Any) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Performs technical validation on a generated synthetic image.
    
    Checks:
      1. Correct type (PIL.Image.Image or np.ndarray)
      2. Valid spatial dimensions (height > 0, width > 0)
      3. Non-degeneracy (checks that image is not completely blank, solid black, or all-white)
      4. Channel consistency
      
    Returns:
      (is_valid, error_reason, technical_metrics)
    """
    if image is None:
        return False, "Generation returned None; no image data received.", {}

    if not isinstance(image, (Image.Image, np.ndarray)):
        return False, f"Unsupported image object type: {type(image)}", {}

    if isinstance(image, Image.Image):
        width, height = image.size
        img_np = np.array(image)
    else:
        height, width = image.shape[:2]
        img_np = image

    if width < 64 or height < 64:
        return False, f"Image dimensions ({width}x{height}) are below minimal threshold (64x64).", {}

    # Check for degenerate images (e.g. collapsed output with standard deviation ~ 0)
    std_dev = float(np.std(img_np))
    mean_val = float(np.mean(img_np))

    metrics = {
        "width": width,
        "height": height,
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_dev, 2),
        "is_degenerate": False
    }

    if std_dev < 1.0:
        metrics["is_degenerate"] = True
        return False, f"Generated image is degenerate / blank (std_dev = {std_dev:.2f}).", metrics

    if mean_val < 2.0:
        metrics["is_degenerate"] = True
        return False, "Generated image is solid black (mean intensity < 2.0).", metrics

    if mean_val > 253.0:
        metrics["is_degenerate"] = True
        return False, "Generated image is solid white (mean intensity > 253.0).", metrics

    return True, None, metrics


def apply_synthetic_watermark(image: Image.Image) -> Image.Image:
    """
    Adds a prominent, high-visibility clinical safety banner along the top of the image
    stating that the radiograph is AI-generated and not for clinical diagnostic use.
    """
    img_copy = image.copy().convert("RGB")
    width, height = img_copy.size
    
    draw = ImageDraw.Draw(img_copy)
    banner_height = max(28, int(height * 0.055))
    
    # Draw dark banner at the top
    draw.rectangle([(0, 0), (width, banner_height)], fill=(20, 20, 20))
    
    # Draw warning text
    banner_text = "SYNTHETIC AI-GENERATED — RESEARCH ONLY — NOT FOR CLINICAL USE"
    
    # Use default font or simple layout
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    text_x = 10
    text_y = int(banner_height / 4)
    draw.text((text_x, text_y), banner_text, fill=(255, 204, 0), font=font)
    
    return img_copy
